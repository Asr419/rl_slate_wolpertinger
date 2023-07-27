from scripts.simulation_imports import *

DEVICE = "cpu"
print("DEVICE: ", DEVICE)
load_dotenv()
base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))


if __name__ == "__main__":
    SEEDS = [5, 7, 46, 53, 77]
    USER_SEED = 13
    NUM_EPISODES = 200

    for seed in tqdm(SEEDS):
        ALPHA = 0.25
        RUN_BASE_PATH = Path(f"slateq_boredom_{ALPHA}_2000_{seed}")

        parser = argparse.ArgumentParser()
        config_path = base_path / RUN_BASE_PATH / Path("config.yaml")
        parser.add_argument(
            "--config",
            type=str,
            default=config_path,
            help="Path to the config file.",
        )
        args = parser.parse_args()
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        parameters = config["parameters"]

        pl.seed_everything(USER_SEED)
        PATH = base_path / RUN_BASE_PATH / Path("model.pt")
        ######## User related parameters ########
        state_model_cls = parameters["state_model_cls"]
        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]
        resp_amp_factor = parameters["resp_amp_factor"]

        ######## Environment related parameters ########
        SLATE_SIZE = parameters["slate_size"]
        NUM_CANDIDATES = parameters["num_candidates"]
        NUM_USERS = parameters["num_users"]
        NUM_ITEM_FEATURES = parameters["num_item_features"]
        SESS_BUDGET = parameters["sess_budget"]
        NUM_USER_FEATURES = parameters["num_user_features"]
        ALPHA_RESPONSE = parameters["alpha_response"]

        ######## Training related parameters ########
        REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
        BATCH_SIZE = parameters["batch_size"]
        GAMMA = parameters["gamma"]
        TAU = parameters["tau"]
        LR = float(parameters["lr"])
        WARMUP_BATCHES = parameters["warmup_batches"]
        DEVICE = parameters["device"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)
        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]

        ######## Init_wandb ########
        RUN_NAME = f"Topic_GAMMA_{GAMMA}_SEED_{seed}_ALPHA_{ALPHA_RESPONSE}_SLATEQ"
        # wandb.init(project="rl_recsys", config=config["parameters"], name=RUN_NAME)

        ################################################################
        user_feat_gen = UniformFeaturesGenerator()
        state_model_cls = class_name_to_class[state_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]

        state_model_kwgs = {}
        choice_model_kwgs = {}
        response_model_kwgs = {"amp_factor": resp_amp_factor, "alpha": ALPHA_RESPONSE}

        user_sampler = UserSampler(
            user_feat_gen,
            state_model_cls,
            choice_model_cls,
            response_model_cls,
            state_model_kwargs=state_model_kwgs,
            choice_model_kwargs=choice_model_kwgs,
            response_model_kwargs=response_model_kwgs,
            sess_budget=SESS_BUDGET,
            num_user_features=NUM_USER_FEATURES,
        )
        user_sampler.generate_users(num_users=NUM_USERS)

        # TODO: dont really now why needed there we shold use the one associated to the user sampled for the episode
        choice_model = choice_model_cls()
        doc_sampler = DocumentSampler(seed=seed)
        env = SlateGym(
            user_sampler=user_sampler,
            doc_sampler=doc_sampler,
            num_candidates=NUM_CANDIDATES,
            device=DEVICE,
        )

        slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
        slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

        # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
        agent = torch.load(PATH)

        transition_cls = Transition
        replay_memory_dataset = ReplayMemoryDataset(
            capacity=REPLAY_MEMORY_CAPACITY, transition_cls=transition_cls
        )
        replay_memory_dataloader = DataLoader(
            replay_memory_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=replay_memory_dataset.collate_fn,
            shuffle=False,
        )

        ############################## TRAINING ###################################
        save_dict = defaultdict(list)
        is_terminal = False
        for i_episode in tqdm(range(NUM_EPISODES)):
            reward, diff_to_best, quality, time_unit_consumed = [], [], [], []

            env.reset()
            is_terminal = False
            cum_reward = 0

            cdocs_features, cdocs_quality, cdocs_length = env.get_candidate_docs()
            user_state = torch.Tensor(env.curr_user.get_state()).to(DEVICE)

            max_sess, avg_sess = [], []
            while not is_terminal:
                # print("user_state: ", user_state)
                with torch.no_grad():
                    ########################################
                    rewards_candidates = (
                        (1 - ALPHA_RESPONSE)
                        * torch.mm(
                            user_state.unsqueeze(0),
                            cdocs_features.t(),
                        )
                        + ALPHA_RESPONSE * cdocs_quality
                    ).squeeze(0) * resp_amp_factor

                    max_rew = rewards_candidates.max()
                    min_rew = rewards_candidates.min()
                    mean_rew = rewards_candidates.mean()
                    std_rew = rewards_candidates.std()

                    max_sess.append(max_rew)
                    avg_sess.append(mean_rew)
                    ########################################

                    user_state_rep = user_state.repeat((cdocs_features.shape[0], 1))

                    q_val = agent.compute_q_values(
                        state=user_state_rep,
                        candidate_docs_repr=cdocs_features,
                        use_policy_net=True,
                    )  # type: ignore

                    choice_model.score_documents(
                        user_state=user_state, docs_repr=cdocs_features
                    )
                    scores = torch.Tensor(choice_model.scores).to(DEVICE)
                    scores = torch.softmax(scores, dim=0)

                    q_val = q_val.squeeze()
                    slate = agent.get_action(scores, q_val)
                    # print("slate: ", slate)

                    (
                        selected_doc_feature,
                        doc_quality,
                        response,
                        is_terminal,
                        _,
                        _,
                    ) = env.step(slate, cdocs_subset_idx=None)
                    # normalize reward between 0 and 1
                    # response = (response - min_rew) / (max_rew - min_rew)
                    reward.append(response)
                    quality.append(doc_quality)

                    next_user_state = env.curr_user.get_state()
                    # push memory
                    user_state = next_user_state
                    if torch.all(selected_doc_feature == 0):
                        time_unit_consumed.append(-0.5)
                    else:
                        time_unit_consumed.append(4.0)

            ep_quality = torch.mean(torch.tensor(quality))
            sess_length = np.sum(time_unit_consumed)
            ep_avg_reward = torch.mean(torch.tensor(reward))
            ep_cum_reward = torch.sum(torch.tensor(reward))
            ep_max_avg = torch.mean(torch.tensor(max_sess))
            ep_max_cum = torch.sum(torch.tensor(max_sess))
            ep_avg_avg = torch.mean(torch.tensor(avg_sess))
            ep_avg_cum = torch.sum(torch.tensor(avg_sess))
            cum_normalized = (
                ep_cum_reward / ep_max_cum
                if ep_max_cum > 0
                else ep_max_cum / ep_cum_reward
            )

            log_str = (
                f"Avg_Reward: {ep_avg_reward} - Cum_Rew: {ep_cum_reward}\n"
                f"Max_Avg_Reward: {ep_max_avg} - Max_Cum_Rew: {ep_max_cum}\n"
                f"Avg_Avg_Reward: {ep_avg_avg} - Avg_Cum_Rew: {ep_avg_cum}\n"
                f"Cumulative_Normalized: {cum_normalized}"
            )
            print(log_str)
            ###########################################################################
            log_dict = {
                "quality": ep_quality,
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
            # wandb.log(log_dict, step=i_episode)

            ###########################################################################
            save_dict["session_length"].append(sess_length)
            save_dict["ep_cum_reward"].append(ep_cum_reward)
            save_dict["ep_avg_reward"].append(ep_avg_reward)

            save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_reward)
            save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
            save_dict["cum_normalized"].append(cum_normalized)

        for k, v in save_dict.items():
            print(k, len(v))
        # wandb.finish()
        directory = f"serving_slateq_2000"
        save_run(seed=seed, save_dict=save_dict, agent=agent, directory=directory)
