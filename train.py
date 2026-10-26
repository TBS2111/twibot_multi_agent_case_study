"""Fit the learnable agents (GBT-CMA) and save checkpoints + outputs."""
import argparse
from config import Config
from graph.workflow import run_pipeline
from run import apply_overrides, print_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. gbt_epochs=200 max_nodes=3000")
    a = ap.parse_args()
    cfg = Config(mode="train")
    apply_overrides(cfg, a.set)
    print_metrics(run_pipeline(cfg)["metrics"])
