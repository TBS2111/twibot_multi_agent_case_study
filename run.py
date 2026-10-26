"""Run the full multi-agent pipeline (loads checkpoints when available)."""
import argparse
from pathlib import Path

from config import Config
from graph.workflow import llm_report, run_pipeline


def apply_overrides(cfg, items):
    for it in items:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, Path):
            v = Path(v)
        elif isinstance(cur, (int, float, str)):
            v = type(cur)(v)
        else:
            v = eval(v)
        setattr(cfg, k, v)
    cfg.__post_init__()


def print_metrics(metrics):
    print("\n" + "=" * 78)
    for stage, m in metrics.items():
        print(f"\n{stage}")
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="infer", choices=["train", "infer"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = Config(mode=a.mode)
    apply_overrides(cfg, a.set)
    st = run_pipeline(cfg)
    print_metrics(st["metrics"])
    if cfg.use_llm_supervisor:
        print("\nLLM report:\n", llm_report(st))
