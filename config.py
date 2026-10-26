"""Central configuration for the multi-agent SBCD pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import torch

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    ckpt_dir: Path = ROOT / "checkpoints"
    out_dir: Path = ROOT / "outputs"

    # ---- data --------------------------------------------------------------
    use_support: bool = False      # add support.json users as extra nodes (unlabeled)
    max_nodes: int = 0             # 0 = use all users, else random subsample
    max_tweets: int = 50           # tweets per user used for features

    # ---- semantic agent ----------------------------------------------------
    semantic_backend: str = "hf"               # "hf" (BERT-style) | "tfidf" (offline) | "sbert"
    hf_model: str = "distilbert-base-uncased"
    #   other options: "distilbert-base-multilingual-cased" (multilingual DistilBERT),
    #                  "bert-base-uncased", "vinai/bertweet-base" (English tweets),
    #                  "cardiffnlp/twitter-xlm-roberta-base" (multilingual tweets),
    #                  "sentence-transformers/all-MiniLM-L6-v2" (fast, English)
    hf_pooling: str = "mean"                   # "mean" | "cls"
    hf_tweets_per_user: int = 20               # tweets embedded per user (speed vs. coverage)
    hf_max_len: int = 64                       # tokens per tweet
    hf_batch: int = 64
    hf_cache: bool = True                      # cache per-user embeddings in checkpoints/
    sem_pca_dim: int = 128                     # reduce transformer vector (0 = keep full size)
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sem_dim: int = 64                          # used by the tfidf backend

    # ---- behavioral agent / graph construction ----------------------------
    knn_k: int = 10                # semantic kNN candidate edges
    random_pairs_per_node: int = 2 # random pairs -> supplies dissimilar (negative) edges
    w_sem: float = 0.25            # weights of the 4 behavioural similarities
    w_url: float = 0.25
    w_interest: float = 0.25
    w_social: float = 0.25
    sign_threshold: float = 0.15   # combined similarity < tau  => negative edge
    interest_top_keywords: int = 10

    # ---- trust agent (random walk) ----------------------------------------
    trust_alpha: float = 0.85
    trust_iters: int = 50

    # Dempster-Shafer reliability (0..1) of each evidence source: how much mass is committed
    # to T/U; the rest (1 - r) stays as uncertainty m(Theta)
    ds_reliability: Dict[str, float] = field(default_factory=lambda: {
        "followers_friends": 0.5, "audience": 0.4, "verified": 1.0, "account_age": 0.5,
        "activity_rate": 0.4, "completeness": 0.4, "engagement": 0.3, "graph": 0.5})

    # ---- GBT-CMA -----------------------------------------------------------
    hidden: int = 128
    heads: int = 4
    gnn_layers: int = 2            # signed GNN (message-passing) layers
    gt_layers: int = 2             # graph-transformer layers
    emb_dim: int = 64
    dropout: float = 0.1
    gbt_epochs: int = 150
    gbt_lr: float = 1e-3
    gbt_weight_decay: float = 1e-5
    neg_margin: float = 0.0

    # ---- SBCD --------------------------------------------------------------
    louvain_resolution: float = 1.0
    min_comm_size: int = 5

    # ---- ICR ---------------------------------------------------------------
    icr_max_iter: int = 10
    split_neg_ratio: float = 0.30  # split if internal negative-weight share > this
    merge_cos_thr: float = 0.80    # centroid cosine needed to merge
    reassign_lambda: float = 0.5
    reassign_max_frac: float = 0.20

    # ---- classifier agent: PPO actor-critic that WALKS the graph (Stable-Baselines3) ----
    rl_total_timesteps: int = 80000  # environment steps (one step = one action)
    rl_n_envs: int = 8               # parallel environments
    rl_n_steps: int = 128            # steps per env per rollout (rollout = n_envs * n_steps samples)
    rl_batch_size: int = 128         # PPO minibatch size
    rl_n_epochs: int = 10            # PPO epochs per rollout
    rl_lr: float = 3e-4
    rl_gamma: float = 0.95           # discount (episodes now have several steps)
    rl_gae_lambda: float = 0.95
    rl_clip_range: float = 0.2       # PPO clip epsilon
    rl_hidden: int = 128             # actor and critic: 2 hidden layers each
    rl_entropy: float = 0.01         # exploration bonus (ent_coef)
    rl_value_coef: float = 0.5       # critic loss weight (vf_coef)
    rl_eval_every: int = 2           # evaluate on the selection set every N rollouts
    rl_patience: int = 10            # stop after N evaluations without improvement
    # reward design
    rl_max_moves: int = 5            # moves allowed before the agent must stop
    rl_step_cost: float = -0.02      # every move (actions 0-6)
    rl_invalid_penalty: float = -0.05  # extra penalty when the move is impossible (agent stays)
    rl_timeout_penalty: float = -1.0 # still moving after rl_max_moves -> episode ends with this
    rl_reward_correct: float = 1.0   # stop with the right label for the ORIGIN account
    rl_penalty_fp: float = -1.0      # stop-bot on a human origin
    rl_penalty_fn: float = -1.0      # stop-human on a bot origin (more negative -> favour recall)

    # ---- evaluation: stratified K-fold cross-validation -------------------
    cv_enabled: bool = True
    cv_folds: List[int] = field(default_factory=lambda: [5, 10])

    # ---- run ---------------------------------------------------------------
    mode: str = "train"            # "train" (fit + save ckpts) | "infer" (load ckpts)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_llm_supervisor: bool = False   # optional narrative report via LangChain

    def __post_init__(self):
        for p in (self.data_dir, self.ckpt_dir, self.out_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
