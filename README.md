# TwiBot Multi-Agent: Social Botnet Community Detection (SBCD + ICR)

Multi-agent re-implementation of *Social Botnet Community Detection: A Novel Approach based on
Behavioral Similarity in Twitter Network using Deep Learning* (Lingam et al., ASIA CCS '20).

## Pipeline (LangGraph)
Supervisor -> [Semantic | Behavioral (URL, Interest, Social) | Trust] (parallel) -> GBT-CMA -> SBCD -> ICR -> Classifier -> Evaluation

| Agent | File | Role |
|---|---|---|
| Supervisor | graph/workflow.py | loads data, builds/dispatches the graph |
| Semantic | agents/semantic_agent.py | tweet-content embeddings (TF-IDF+SVD or SBERT) |
| Behavioral | agents/behavioral_agent.py | URL / interest / social-interaction sets, link edges |
| Trust | agents/trust_agent.py | Dempster-Shafer evidence-based trust (profile evidence + random-walk graph evidence) |
| GBT-CMA | agents/gbt_cma_agent.py | signed weighted graph + Cross-Modal Attention -> signed GNN -> Graph Transformer (gated) embeddings |
| SBCD | agents/sbcd_agent.py | initial communities (Louvain on positive similarity edges) |
| ICR | agents/icr_agent.py | Split / Merge / Reassign to minimise signed frustration |
| Classifier | agents/classifier_agent.py | actor-critic deep-RL: Stable-Baselines3 PPO agent that walks the graph (9 actions) on a Gymnasium env bot-vs-human policy + gradient-boosting baseline |
| Evaluation | agents/evaluation_agent.py | modularity, frustration, NMI/ARI/purity, community-based bot F1 |

## Setup
    pip install -r requirements.txt
Put TwiBot-20 `train.json dev.json test.json (support.json)` in `data/`.
No data yet? `python scripts/make_synthetic.py data` creates a tiny fake set for smoke tests.

## Run
    python train.py                       # fit GBT-CMA, save checkpoints, write outputs/
    python run.py --mode infer            # reuse checkpoints
    python evaluate.py                    # recompute metrics from outputs/
    python train.py --set max_nodes=3000 gbt_epochs=200 semantic_backend=sbert use_support=false

All hyper-parameters live in `config.py` and can be overridden with `--set key=value`.
Outputs: `outputs/communities.csv, community_report.csv, metrics.json, embeddings.npy, graph.npz`.

## Notes / honest caveats
- The paper's exact formulas are not reproduced line-by-line; the similarity weights, signing
  threshold, trust walk, SBCD and ICR are faithful-in-spirit implementations. Check them against the paper's equations.
- Labels are used only for evaluation (bot ratio per community is estimated from `train` users only).
- Start with `max_nodes=3000`, then scale. `use_support=true` adds ~217k unlabeled nodes: needs lots of RAM.
- Tune `sign_threshold`, `louvain_resolution`, `min_comm_size` first; they drive community count most.
A “Data” section saying the TwiBot-20 JSON files must be placed in data/ and that they are not included.
The results from your run. Copy the numbers from the report: 3,000 labelled users, 0.839 vs 0.868 CV accuracy for PPO vs gradient boosting, and NMI 0.033 for the communities.
The honest caveats already in the report, for example that the PPO policy almost never walks the graph.
The run commands from the steps I gave earlier.
