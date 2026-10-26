"""Stage 0: load pipeline outputs (communities.csv, graph.npz) + raw TwiBot json into one SQLite file.

Everything downstream (timeline, report, validation) reads ONLY this database, so a run is
reproducible from (outputs/, data/) alone. Row order = pipeline node order, which is how
graph.npz edge indices map back to user ids.
"""
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.dataset import load_split, to_bool, to_num  # noqa: E402

URL_RE = re.compile(r"https?://[^\s]+", re.I)
TAG_RE = re.compile(r"#(\w+)")
MEN_RE = re.compile(r"@(\w+)")
WS_RE = re.compile(r"\s+")

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE users(
  idx INTEGER PRIMARY KEY, id TEXT UNIQUE, screen_name TEXT, label INTEGER, split TEXT,
  community INTEGER, initial_community INTEGER, created_at TEXT,
  followers REAL, friends REAL, statuses REAL, verified INTEGER, default_profile INTEGER);
CREATE TABLE edges(src INTEGER, dst INTEGER, w REAL, sem REAL, url REAL, interest REAL, social REAL);
CREATE TABLE tweets(user_id INTEGER, pos INTEGER, norm TEXT);
CREATE TABLE tweet_urls(user_id INTEGER, url TEXT, n INTEGER);
CREATE TABLE tweet_tags(user_id INTEGER, tag TEXT, n INTEGER);
CREATE INDEX ix_users_c ON users(community);
CREATE INDEX ix_edges_s ON edges(src); CREATE INDEX ix_edges_d ON edges(dst);
CREATE INDEX ix_urls ON tweet_urls(url); CREATE INDEX ix_tags ON tweet_tags(tag);
CREATE INDEX ix_tw_u ON tweets(user_id); CREATE INDEX ix_tw_n ON tweets(norm);
"""


def parse_created(v):
    try:
        ts = pd.to_datetime(str(v).strip(), utc=True)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def norm_text(t: str) -> str:
    return WS_RE.sub(" ", MEN_RE.sub(" ", URL_RE.sub(" ", t))).strip().lower()


def build_db(out_dir, data_dir, db_path, community_col="refined", max_tweets=50):
    out_dir, data_dir, db_path = Path(out_dir), Path(data_dir), Path(db_path)
    df = pd.read_csv(out_dir / "communities.csv", dtype={"id": str})
    g = np.load(out_dir / "graph.npz")
    raw = {}
    for split in ("train", "dev", "test", "support"):
        p = data_dir / f"{split}.json"
        if p.exists():
            for u in load_split(p, split):
                raw.setdefault(u["id"], u)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    missing = 0
    for i, r in enumerate(df.itertuples(index=False)):
        u = raw.get(r.id)
        p = (u or {}).get("profile", {})
        missing += u is None
        con.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            i, r.id, str(p.get("screen_name", "")).strip() or None, int(r.label), str(r.split),
            int(getattr(r, community_col)), int(r.initial), parse_created(p.get("created_at")),
            to_num(p.get("followers_count"), None), to_num(p.get("friends_count"), None),
            to_num(p.get("statuses_count"), None), int(to_bool(p.get("verified"))),
            int(to_bool(p.get("default_profile")))))
        urls, tags = Counter(), Counter()
        for pos, t in enumerate((u or {}).get("tweets", [])[:max_tweets]):
            con.execute("INSERT INTO tweets VALUES (?,?,?)", (i, pos, norm_text(t)))
            urls.update(x.rstrip(".,;:!?)\"'").lower() for x in URL_RE.findall(t))
            tags.update(x.lower() for x in TAG_RE.findall(t))
        con.executemany("INSERT INTO tweet_urls VALUES (?,?,?)", [(i, k, v) for k, v in sorted(urls.items())])
        con.executemany("INSERT INTO tweet_tags VALUES (?,?,?)", [(i, k, v) for k, v in sorted(tags.items())])

    e, w, s = g["edges"], g["w"], g["sims"]
    con.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?,?)",
                    [(int(a), int(b), float(x), *map(float, sim)) for (a, b), x, sim in zip(e, w, s)])
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("community_col", community_col), ("n_users", str(len(df))),
        ("users_without_raw_profile", str(missing)), ("max_tweets", str(max_tweets))])
    con.commit()
    con.close()
    return dict(users=len(df), edges=len(e), users_without_raw_profile=missing)


def connect(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = None
    return con
