import time

import pandas as pd

from scripts.simulation_imports import *

load_dotenv()
base_path = Path.home() / Path(os.environ.get("SAVE_PATH"))

save_path = os.environ.get("SAVE_PATH")
BASE_LOAD_PATH = Path.home() / save_path

MODEL_SEED = 5
RUN_K = [5, 10, 20]
DEVICE = "cpu"
print("DEVICE: ", DEVICE)

NUM_CANDIDATES = [300, 500, 1000, 2000]
if __name__ == "__main__":
    seed = 37

    model_name_list = []
    num_candidates_list = []
    serving_time_users_list = []

    FOLDER_NAME = f"slateq_boredom_0.25_300_{MODEL_SEED}"
    AGENT_PATH = base_path / FOLDER_NAME / Path("model.pt")
    parser = argparse.ArgumentParser()
    config_path = base_path / FOLDER_NAME / Path("config.yaml")
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

    for num_candidates in NUM_CANDIDATES:
        num_candidates_list.append(num_candidates)
        model_name_list.append(f"SlateQ")

        ######## User related parameters ########
        state_model_cls = parameters["state_model_cls"]
        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]
        resp_amp_factor = parameters["resp_amp_factor"]

        ######## Environment related parameters ########
        SLATE_SIZE = parameters["slate_size"]
        NUM_USERS = 5
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
        NUM_EPISODES = 5
        WARMUP_BATCHES = parameters["warmup_batches"]
        DEVICE = parameters["device"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)
        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]

        ################################################################
        user_feat_gen = UniformFeaturesGenerator()
        state_model_cls = class_name_to_class[state_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]

        state_model_kwgs = {}
        choice_model_kwgs = {}
        response_model_kwgs = {
            "amp_factor": resp_amp_factor,
            "alpha": ALPHA_RESPONSE,
        }

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
            num_candidates=num_candidates,
            device=DEVICE,
        )

        slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
        slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

        # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
        agent = torch.load(AGENT_PATH)

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

        user_serving_time = []
        for i_episode in tqdm(range(NUM_EPISODES)):
            reward, diff_to_best, quality = [], [], []

            env.reset()
            is_terminal = False
            cum_reward = 0

            cdocs_features, cdocs_quality, cdocs_length = env.get_candidate_docs()
            user_state = torch.Tensor(env.curr_user.get_state()).to(DEVICE)

            max_sess, avg_sess = [], []
            serving_time = []
            while not is_terminal:
                start = time.time()
                with torch.no_grad():
                    # user_state_rep = user_state.repeat(
                    #     (cdocs_features.shape[0], 1)
                    # )  # type: ignore

                    q_val_list = []
                    for cdoc in cdocs_features:
                        q_val = agent.compute_q_values(
                            state=user_state.unsqueeze(dim=0),
                            candidate_docs_repr=cdoc.unsqueeze(dim=0),
                            use_policy_net=True,
                        )  # type: ignore
                        q_val_list.append(q_val)
                    q_val = torch.stack(q_val_list).to(DEVICE)

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
                        diversity,
                        selected_position,
                    ) = env.step(slate, cdocs_subset_idx=None)

                    end = time.time()
                    serving_time.append(end - start)

                    reward.append(response)

                    next_user_state = env.curr_user.get_state()
                    # push memory
                    replay_memory_dataset.push(
                        transition_cls(
                            user_state,  # type: ignore
                            selected_doc_feature,
                            cdocs_features,
                            response,
                            next_user_state,  # type: ignore
                        )
                    )
                    user_state = next_user_state
            user_serving_time.append(np.mean(serving_time))

        print("mean serving time: ", np.mean(user_serving_time))
        serving_time_users_list.append(np.mean(user_serving_time))
    # construct a df and save it
    res_df = pd.DataFrame(
        zip(model_name_list, num_candidates_list, serving_time_users_list),
        columns=["model_name", "num_candidates", "serving_time"],
    )
    res_df.to_csv(BASE_LOAD_PATH / "serving_time_slateq.csv", index=False)
    print(res_df)
