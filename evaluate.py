"""Re-compute metrics from saved outputs (outputs/communities.csv + graph.npz) without re-running."""
import json
import numpy as np
import pandas as pd

from agents.evaluation_agent import compute_metrics, dataset_counts
from config import Config
from run import print_metrics

if __name__ == "__main__":
    cfg = Config()
    df = pd.read_csv(cfg.out_dir / "communities.csv")
    g = np.load(cfg.out_dir / "graph.npz")
    stages = {"SBCD (initial)": df["initial"].values, "SBCD+ICR (refined)": df["refined"].values}
    m = compute_metrics(stages, df["label"].values, df["split"].values.astype(str),
                        g["edges"][:, 0], g["edges"][:, 1], g["w"], len(df))
    m = {**dataset_counts(df["label"].values, df["split"].values.astype(str)), **m}
    print_metrics(m)
    json.dump(m, open(cfg.out_dir / "metrics_eval.json", "w"), indent=2)
