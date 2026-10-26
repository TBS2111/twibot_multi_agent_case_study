"""Loader for TwiBot-20 style json files (train/dev/test/support)."""
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger(__name__)


def to_num(x, default=0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def to_bool(x) -> float:
    return 1.0 if str(x).strip().lower() == "true" else 0.0


def _ids(lst) -> List[str]:
    return [str(i).strip() for i in lst] if lst else []


def load_split(path: Path, split: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    users = []
    for d in data:
        nb = d.get("neighbor") or {}
        try:
            label = int(d.get("label"))
        except (TypeError, ValueError):
            label = -1
        users.append(dict(
            id=str(d["ID"]).strip(),
            label=label,
            split=split,
            profile=d.get("profile") or {},
            tweets=[t for t in (d.get("tweet") or []) if isinstance(t, str)],
            following=_ids(nb.get("following")),
            followers=_ids(nb.get("follower")),
        ))
    return users


def load_twibot(cfg) -> List[dict]:
    users, seen = [], set()
    names = ["train", "dev", "test"] + (["support"] if cfg.use_support else [])
    for s in names:
        p = Path(cfg.data_dir) / f"{s}.json"
        if not p.exists():
            log.warning("missing %s - skipped", p)
            continue
        for u in load_split(p, s):
            if u["id"] not in seen:
                seen.add(u["id"])
                users.append(u)
    if not users:
        raise FileNotFoundError(f"No json files found in {cfg.data_dir}")
    if cfg.max_nodes and len(users) > cfg.max_nodes:
        rng = np.random.RandomState(cfg.seed)
        keep = np.sort(rng.choice(len(users), cfg.max_nodes, replace=False))
        users = [users[i] for i in keep]
    return users
