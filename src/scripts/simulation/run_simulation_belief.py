from rl_recsys.agent_modeling.dqn_agent import BeliefTransition
from rl_recsys.user_modeling.features_gen import UniformFeaturesGenerator
from scripts.simulation_imports import *


def optimize_model(batch):
    optimizer.zero_grad()

    (
        prev_state_batch,  # [batch_size, num_item_features]
        prev_obs_buff_batch,
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        reward_batch,  # [batch_size, 1]
        gru_buffer_batch,  # [batch_size, num_item_features]
    ) = batch

    # print(reward_batch)
    state_batch = bf_agent.update_belief(prev_state_batch, prev_obs_buff_batch)
    # Q(s, a): [batch_size, 1]
    q_val = bf_agent.agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )  # type: ignore

    # compute s'
    next_state_batch = bf_agent.update_belief(state_batch, gru_buffer_batch)

    # Q(s', a): [batch_size, 1]
    cand_qtgt_list = []
    for b in range(next_state_batch.shape[0]):
        next_state = next_state_batch[b, :]
        candidates = candidates_batch[b, :, :]

        next_state_rep = next_state.repeat((candidates.shape[0], 1))
        cand_qtgt = bf_agent.agent.compute_q_values(
            next_state_rep, candidates, use_policy_net=False
        )  # type: ignore

        choice_model.score_documents(next_state, candidates)
        # [num_candidates, 1]
        scores_tens = torch.Tensor(choice_model.scores).to(DEVICE).unsqueeze(dim=1)
        scores_tens = torch.softmax(scores_tens, dim=0)

        # max over Q(s', a)
        cand_qtgt_list.append((cand_qtgt * scores_tens).max())

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(dim=1)
    expected_q_values = q_tgt * GAMMA + reward_batch.unsqueeze(dim=1)
    loss = criterion(q_val, expected_q_values)

    # Optimize the model
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/scripts/config.yaml",
        help="Path to the config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    now = datetime.now()
    time_now = now.strftime("%m-%d_%H-%M-%S")

    SEEDS = config["parameters"]["seeds"]["value"]
    for seed in SEEDS:
        pl.seed_everything(seed)

        ######## User related parameters ########
        state_model_cls = config["parameters"]["state_model_cls"]["value"]
        choice_model_cls = config["parameters"]["choice_model_cls"]["value"]
        response_model_cls = config["parameters"]["response_model_cls"]["value"]
        resp_amp_factor = config["parameters"]["resp_amp_factor"]["value"]
        state_update_rate = config["parameters"]["state_update_rate"]["value"]

        ######## Environment related parameters ########
        SLATE_SIZE = config["parameters"]["slate_size"]["value"]
        NUM_CANDIDATES = config["parameters"]["num_candidates"]["value"]
        NUM_USERS = config["parameters"]["num_users"]["value"]
        NUM_ITEM_FEATURES = config["parameters"]["num_item_features"]["value"]
        INTENT_KIND = config["parameters"]["intent_kind"]["value"]
        SONG_PER_SESSION = config["parameters"]["song_per_session"]["value"]

        assert INTENT_KIND in ["observable", "static", "random_state", "random_slate"]
        ######## Training related parameters ########
        BATCH_SIZE = config["parameters"]["batch_size"]["value"]
        GAMMA = config["parameters"]["gamma"]["value"]
        TAU = config["parameters"]["tau"]["value"]
        LR = float(config["parameters"]["lr"]["value"])
        NUM_EPISODES = config["parameters"]["num_episodes"]["value"]
        WARMUP_BATCHES = config["parameters"]["warmup_batches"]["value"]

        DEVICE = config["parameters"]["device"]["value"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)

        ######## Models related parameters ########
        history_model_cls = config["parameters"]["history_model_cls"]["value"]
        belief_model_cls = config["parameters"]["belief_model_cls"]["value"]
        slate_gen_model_cls = config["parameters"]["slate_gen_model_cls"]["value"]
        HIST_LENGTH = config["parameters"]["hist_length"]["value"]

        ######## Models related parameters ########
        SAVE_PATH = Path(os.environ.get("SAVE_PATH"))
        SAVE_PATH = Path.home() / SAVE_PATH
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        ##################################################
        #################### CATALOGUE ###################
        data_df = load_spotify_data()
        doc_catalogue = DocCatalogue(doc_df=data_df, doc_id_column="song_id")

        ################### RETRIEVAL MODEL ###################
        item_features = doc_catalogue.get_all_item_features()
        rec_model = ContentSimilarityRec(item_feature_matrix=item_features)

        ################## USER SAMPLER ###################
        user_feat_gen = UniformFeaturesGenerator()
        intent_feat_gen = UniformFeaturesGenerator()
        state_model_cls = class_name_to_class[state_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]

        state_model_kwgs = {
            "state_update_rate": state_update_rate,
            "intent_gen": intent_feat_gen,
        }
        choice_model_kwgs = {}
        response_model_kwgs = {"amp_factor": resp_amp_factor}

        user_sampler = UserSampler(
            user_feat_gen,
            state_model_cls,
            choice_model_cls,
            response_model_cls,
            state_model_kwargs=state_model_kwgs,
            choice_model_kwargs=choice_model_kwgs,
            response_model_kwargs=response_model_kwgs,
            songs_per_sess=SONG_PER_SESSION,
            device=DEVICE,
        )
        user_sampler.generate_users(num_users=NUM_USERS)

        # TODO: dont really now why needed there we shold use the one associated to the user sampled for the episode
        choice_model = choice_model_cls()

        env = MusicGym(
            user_sampler=user_sampler,
            doc_catalogue=doc_catalogue,
            rec_model=rec_model,
            k=NUM_CANDIDATES,
            device=DEVICE,
        )

        # define Slate Gen model
        slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
        slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

        # defining Belief Agent
        # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
        agent = DQNAgent(
            slate_gen=slate_gen,
            input_size=2 * NUM_ITEM_FEATURES,
            output_size=1,
            tau=TAU,
        )

        # history model
        history_model_cls = class_name_to_class[history_model_cls]
        history_model = history_model_cls(
            num_doc_features=NUM_ITEM_FEATURES,
            hist_length=HIST_LENGTH,
        )
        # belief model
        belief_model_cls = class_name_to_class[belief_model_cls]
        belief_model = belief_model_cls(
            num_doc_features=NUM_ITEM_FEATURES, hist_model=history_model
        )

        bf_agent = BeliefAgent(
            agent=agent,
            belief_model=belief_model,
        ).to(device=DEVICE)

        transition_cls = BeliefTransition

        replay_memory_dataset = ReplayMemoryDataset(
            capacity=10000, transition_cls=transition_cls
        )
        replay_memory_dataloader = DataLoader(
            replay_memory_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=replay_memory_dataset.collate_fn,
            shuffle=False,
        )

        criterion = torch.nn.SmoothL1Loss()
        optimizer = optim.Adam(bf_agent.parameters(), lr=LR)

        is_terminal = False
        # Initialize b_u
        b_u = None

        keys = [
            "ep_reward",
            "ep_avg_reward",
            "loss",
            "best_rl_avg_diff",
            "avg_avd_diff",
            "cum_normalized",
        ]
        save_dict = defaultdict(list)
        save_dict.update({key: [] for key in keys})

        # init wandb
        RUN_NAME = f"{history_model_cls}_SEED_{seed}_{time_now}"
        wandb.init(project="rl_recsys", config=config["parameters"], name=RUN_NAME)

        for i_episode in tqdm(range(NUM_EPISODES)):
            obs_buff = torch.zeros((HIST_LENGTH, NUM_ITEM_FEATURES)).to(DEVICE)
            count = 0

            reward = []
            loss = []
            diff_to_best = []

            env.reset()
            is_terminal = False
            cum_reward = 0

            b_u = torch.Tensor(env.curr_user.features).to(DEVICE)
            # b_u = (torch.rand(NUM_ITEM_FEATURES) * 2 - 1).to(DEVICE)

            # print("init b_u")
            # print(b_u)
            # print("---------------")

            candidate_docs = env.get_candidate_docs()
            candidate_docs_repr = torch.Tensor(
                env.doc_catalogue.get_docs_features(candidate_docs)
            ).to(DEVICE)

            max_sess = []
            avg_sess = []

            prev_b_u = None
            while not is_terminal:
                with torch.no_grad():
                    ##########################################################################
                    max_sess.append(
                        torch.mm(
                            env.curr_user.get_state().unsqueeze(0),
                            candidate_docs_repr.t(),
                        )
                        .squeeze(0)
                        .max()
                    )

                    avg_sess.append(
                        torch.mm(
                            env.curr_user.get_state().unsqueeze(0),
                            candidate_docs_repr.t(),
                        )
                        .squeeze(0)
                        .mean()
                    )
                    ##########################################################################
                    # print(obs_buff)
                    b_u_rep = b_u.repeat((candidate_docs_repr.shape[0], 1))

                    q_val = bf_agent.agent.compute_q_values(
                        state=b_u_rep,
                        candidate_docs_repr=candidate_docs_repr,
                        use_policy_net=True,
                    )  # type: ignore

                    # print("------before------")
                    # print(b_u)
                    # print("---------------")
                    choice_model.score_documents(
                        user_state=b_u, docs_repr=candidate_docs_repr
                    )
                    scores = torch.Tensor(choice_model.scores).to(DEVICE)
                    # scores = torch.exp(scores)
                    # normalize scores for numerical stability
                    scores = scores - torch.max(scores)
                    # scores = torch.softmax(scores, dim=0)

                    # scores = torch.ones_like(scores)
                    q_val = q_val.squeeze()

                    slate = bf_agent.get_action(scores, q_val)
                    # print(slate)

                    selected_doc_feature, response, is_terminal, _, _ = env.step(slate)

                    response_bu = env.curr_user.response_model.generate_response(
                        b_u, selected_doc_feature
                    )

                    reward.append(response)
                    # print("------after------")
                    # print(response)
                    # print(response_bu)
                    # print("-------------------")
                    prev_obs_buff = obs_buff.clone()

                    # fill the GRU buffer
                    if count >= (HIST_LENGTH - 1):
                        # shift the buffer
                        for i in range(HIST_LENGTH - 1):
                            obs_buff[i, :] = obs_buff[i + 1, :]
                        obs_buff[-1, :] = selected_doc_feature
                    else:
                        obs_buff[
                            HIST_LENGTH - (count % HIST_LENGTH) - 1, :
                        ] = selected_doc_feature
                    count += 1

                    if prev_b_u is not None:
                        # push memory
                        replay_memory_dataset.push(
                            transition_cls(
                                prev_b_u,
                                prev_obs_buff,
                                selected_doc_feature,
                                candidate_docs_repr,
                                response,
                                obs_buff,
                            )
                        )

                    prev_b_u = b_u.clone()
                    b_u = bf_agent.update_belief(b_u, obs_buff)
                    # b_u = (torch.rand(NUM_ITEM_FEATURES) * 2 - 1).to(DEVICE)
                    # print("------after------")
                    # print(b_u)
                    # print(env.curr_user.features)
                    # print(env.curr_user.get_state())
                    # print("------==========-------")

            # optimize model
            if len(replay_memory_dataset.memory) >= WARMUP_BATCHES * SONG_PER_SESSION:
                # get a batch of transitions from the replay buffer
                batch = next(iter(replay_memory_dataloader))
                for elem in batch:
                    elem.to(DEVICE)
                batch_loss = optimize_model(batch)
                bf_agent.agent.soft_update_target_network()

                # accumulate loss for each episode
                loss.append(batch_loss)

            ep_avg_reward = torch.mean(torch.tensor(reward))
            ep_cum_reward = torch.sum(torch.tensor(reward))

            loss = torch.mean(torch.tensor(loss))

            ep_max_avg = torch.mean(torch.tensor(max_sess))
            ep_max_cum = torch.sum(torch.tensor(max_sess))

            ep_avg_avg = torch.mean(torch.tensor(avg_sess))
            ep_avg_cum = torch.sum(torch.tensor(avg_sess))

            if ep_max_cum > 0:
                cum_normalized = ep_cum_reward / ep_max_cum
            else:
                cum_normalized = ep_max_cum / ep_cum_reward

            print(
                "Loss: {}\n Avg_Reward: {} - Cum_Rew: {}\n Max_Avg_Reward: {} - Max_Cum_Rew: {}\n Avg_Avg_Reward: {} - Avg_Cum_Rew: {}: - Cumulative_Normalized: {}".format(
                    loss,
                    ep_avg_reward,
                    ep_cum_reward,
                    ep_max_avg,
                    ep_max_cum,
                    ep_avg_avg,
                    ep_avg_cum,
                    cum_normalized,
                )
            )

            log_dit = {
                "avg_reward": ep_avg_reward,
                "cum_reward": ep_cum_reward,
                "max_avg": ep_max_avg,
                "max_cum": ep_max_cum,
                "avg_avg": ep_avg_avg,
                "avg_cum": ep_avg_cum,
                "best_rl_avg_diff": ep_max_avg - ep_avg_reward,
                "best_avg_avg_diff": ep_max_avg - ep_avg_avg,
                "cum_normalized": cum_normalized,
            }

            if len(replay_memory_dataset.memory) >= WARMUP_BATCHES * SONG_PER_SESSION:
                log_dit["loss"] = loss

            wandb.log(log_dit, step=i_episode)

            save_dict["ep_reward"].append(ep_cum_reward)
            save_dict["ep_avg_reward"].append(ep_avg_reward)
            save_dict["loss"].append(loss)
            save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_reward)
            save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
            save_dict["cum_normalized"].append(cum_normalized)

        wandb.finish()
        directory = f"beliefm_{belief_model_cls}_histm_{history_model_cls}"

        directory = directory + "_" + str(seed)

        # Create the directory with the folder name
        path = Path(directory + "_" + time_now)
        save_dir = Path(SAVE_PATH / path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # save config
        source_path = "src/scripts/config.yaml"

        destination_path = save_dir / Path("config.yaml")
        shutil.copy(source_path, destination_path)

        # Save the model
        model_save_name = f"model.pt"
        torch.save(bf_agent, save_dir / Path(model_save_name))

        # save logs dict
        logs_save_name = Path(f"logs_dict.pickle")
        with open(save_dir / logs_save_name, "wb") as f:
            pickle.dump(save_dict, f)