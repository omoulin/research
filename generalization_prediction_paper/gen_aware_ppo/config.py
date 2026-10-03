from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Environment selection                                               #
    # ------------------------------------------------------------------ #
    env_name: str = "coinrun-vec"   # "coinrun-vec" | "minigrid-simplecrossing-vec"

    # ------------------------------------------------------------------ #
    # Phase 1 – data collection                                           #
    # ------------------------------------------------------------------ #
    num_base_models: int = 800      # how many PPO models to train
    train_timesteps: int = 1_000_000
    train_num_levels: int = 200     # levels used during training
    train_start_level: int = 0

    # ------------------------------------------------------------------ #
    # Phase 1 – generalization evaluation                                 #
    # ------------------------------------------------------------------ #
    n_eval_episodes: int = 1000
    eval_start_level: int = 10_000  # far above training levels

    # ------------------------------------------------------------------ #
    # PPO shared hyper-parameters                                         #
    # ------------------------------------------------------------------ #
    n_envs: int = 64
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 3
    learning_rate: float = 5e-4
    ent_coef: float = 0.01
    clip_range: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    # ------------------------------------------------------------------ #
    # Phase 2 – predictor                                                 #
    # ------------------------------------------------------------------ #
    predictor_hidden_dim: int = 256
    predictor_epochs: int = 200
    predictor_lr: float = 1e-3
    predictor_val_split: float = 0.2
    predictor_dropout: float = 0.0       # non-zero activates dropout regularisation
    predictor_weight_decay: float = 0.0  # L2 regularisation on predictor weights

    # ------------------------------------------------------------------ #
    # Phase 3 – generalization-aware PPO                                  #
    # ------------------------------------------------------------------ #
    gen_timesteps: int = 1_000_000
    n_phase3_agents: int = 10       # agents trained per approach
    gen_coef: float = 0.005         # weight of the predictor term in PPO loss
    gen_eval_freq: int = 100_000    # timesteps between periodic gen evaluations
    gen_eval_episodes: int = 100    # episodes per periodic gen evaluation

    # ------------------------------------------------------------------ #
    # Paths (overridden in __post_init__ for non-coinrun envs)            #
    # ------------------------------------------------------------------ #
    data_path: str = "data/phase1_dataset.npz"
    gen_scores_path: str = "data/gen_scores.npy"
    predictor_path: str = "data/predictor.pt"
    base_model_dir: str = "data/base_models"

    # ------------------------------------------------------------------ #
    # MiniGrid-specific                                                    #
    # ------------------------------------------------------------------ #
    minigrid_train_num_seeds: int = 500   # training seed pool (≈ num_levels)
    minigrid_eval_seed_start: int = 10_000  # eval seeds start here (unseen)

    # ------------------------------------------------------------------ #
    # Policy (set in __post_init__ based on env_name)                     #
    # ------------------------------------------------------------------ #
    policy_type: str = "CnnPolicy"   # CoinRun (RGB obs); MiniGrid vec switches to MlpPolicy

    # ------------------------------------------------------------------ #
    # Quick-test mode (reduces every expensive number)                    #
    # ------------------------------------------------------------------ #
    quick: bool = False

    def __post_init__(self) -> None:
        if self.env_name == "coinrun-vec":
            d = "data/coinrun_vec"
            self.data_path = f"{d}/phase1_dataset.npz"
            self.gen_scores_path = f"{d}/gen_scores.npy"
            self.predictor_path = f"{d}/predictor.pt"
            self.base_model_dir = f"{d}/base_models"

        if self.env_name == "minigrid-simplecrossing-vec":
            self.policy_type = "MlpPolicy"
            d = "data/simplecrossing_vec"
            self.data_path = f"{d}/phase1_dataset.npz"
            self.gen_scores_path = f"{d}/gen_scores.npy"
            self.predictor_path = f"{d}/predictor.pt"
            self.base_model_dir = f"{d}/base_models"
            self.train_timesteps = 500_000
            self.gen_timesteps = 500_000
            self.n_envs = 32
            self.minigrid_train_num_seeds = 150
            self.predictor_hidden_dim = 64
            self.predictor_dropout = 0.2
            self.predictor_weight_decay = 1e-3
            self.predictor_epochs = 300
            self.gen_coef = 0.01

        if self.quick:
            self.num_base_models = 6
            self.train_timesteps = 100_000
            self.n_eval_episodes = 100
            self.gen_timesteps = 200_000
            self.n_phase3_agents = 2
            self.gen_eval_freq = 20_000
            self.gen_eval_episodes = 20
            self.n_envs = 8
            self.predictor_epochs = 50
