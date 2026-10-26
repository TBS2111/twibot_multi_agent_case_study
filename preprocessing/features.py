"""Feature extraction + signed weighted graph construction."""
import re
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocessing.dataset import to_bool, to_num

URL_RE = re.compile(r"https?://[^\s]+", re.I)
TAG_RE = re.compile(r"#(\w+)")
MEN_RE = re.compile(r"@(\w+)")


def l2norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def zscore(x):
    x = np.nan_to_num(x.astype(np.float32))
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def clean_text(t: str) -> str:
    t = URL_RE.sub(" ", t)
    t = MEN_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def user_doc(u, max_tweets) -> str:
    return " ".join(clean_text(t) for t in u["tweets"][:max_tweets])


# --------------------------------------------------------------------------
# Semantic modality
# --------------------------------------------------------------------------
def semantic_embeddings(users: List[dict], cfg) -> np.ndarray:
    docs = [user_doc(u, cfg.max_tweets) for u in users]
    if cfg.semantic_backend == "sbert":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg.sbert_model, device=cfg.device)
        emb = m.encode([d[:2000] or "empty" for d in docs], batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
        return emb.astype(np.float32)
    tfidf = TfidfVectorizer(max_features=30000, stop_words="english",
                            sublinear_tf=True, min_df=2)
    try:
        X = tfidf.fit_transform(docs)
    except ValueError:                       # empty vocabulary
        return np.zeros((len(users), cfg.sem_dim), np.float32)
    n_comp = max(2, min(cfg.sem_dim, X.shape[1] - 1, X.shape[0] - 1))
    emb = TruncatedSVD(n_comp, random_state=cfg.seed).fit_transform(X)
    if n_comp < cfg.sem_dim:
        emb = np.pad(emb, ((0, 0), (0, cfg.sem_dim - n_comp)))
    return l2norm(emb).astype(np.float32)


# --------------------------------------------------------------------------
# Behavioural modality : URL / interest / social item sets
# --------------------------------------------------------------------------
def sets_to_csr(sets):
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(sets):
        for it in s:
            rows.append(i)
            cols.append(vocab.setdefault(it, len(vocab)))
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(sets), max(len(vocab), 1)))


def jaccard_pairs(A: sp.csr_matrix, src, dst, chunk=100000):
    sizes = np.asarray(A.sum(1)).ravel()
    out = np.zeros(len(src), np.float32)
    for s in range(0, len(src), chunk):
        a, b = src[s:s + chunk], dst[s:s + chunk]
        inter = np.asarray(A[a].multiply(A[b]).sum(1)).ravel()
        uni = sizes[a] + sizes[b] - inter
        out[s:s + chunk] = np.where(uni > 0, inter / np.maximum(uni, 1), 0)
    return out


def url_sets(users):
    out = []
    for u in users:
        s = set()
        for t in u["tweets"]:
            for url in URL_RE.findall(t):
                url = url.rstrip(".,;)")
                s.add(url)
                dom = url.split("/")[2] if url.count("/") >= 2 else url
                if "t.co" not in dom:
                    s.add("dom:" + dom)
        out.append(s)
    return out


def interest_sets(users, cfg):
    docs = [user_doc(u, cfg.max_tweets) for u in users]
    tags = [{"#" + h.lower() for t in u["tweets"] for h in TAG_RE.findall(t)} for u in users]
    try:
        tf = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2, sublinear_tf=True)
        X = tf.fit_transform(docs).tocsr()
        vocab = np.array(tf.get_feature_names_out())
        for i in range(X.shape[0]):
            row = X.getrow(i)
            if row.nnz:
                top = row.indices[np.argsort(-row.data)[:cfg.interest_top_keywords]]
                tags[i].update("k:" + w for w in vocab[top])
    except ValueError:
        pass
    return tags


def social_sets(users):
    out = []
    for u in users:
        s = {"n:" + x for x in u["following"]} | {"n:" + x for x in u["followers"]}
        s |= {"m:" + m.lower() for t in u["tweets"] for m in MEN_RE.findall(t)}
        out.append(s)
    return out


def link_edges(users) -> np.ndarray:
    """Directed (i -> j) interaction edges inside the node set: follow + mention."""
    idx = {u["id"]: i for i, u in enumerate(users)}
    name = {str(u["profile"].get("screen_name", "")).strip().lower(): i
            for i, u in enumerate(users) if u["profile"].get("screen_name")}
    e = set()
    for i, u in enumerate(users):
        for x in u["following"]:
            j = idx.get(x)
            if j is not None and j != i:
                e.add((i, j))
        for x in u["followers"]:
            j = idx.get(x)
            if j is not None and j != i:
                e.add((j, i))
        for t in u["tweets"]:
            for m in MEN_RE.findall(t):
                j = name.get(m.lower())
                if j is not None and j != i:
                    e.add((i, j))
    return np.array(sorted(e), dtype=np.int64).reshape(-1, 2)


def behavior_stats(users) -> np.ndarray:
    rows = []
    for u in users:
        tw = u["tweets"]
        n = max(len(tw), 1)
        rows.append([
            np.log1p(len(tw)),
            sum(bool(URL_RE.search(t)) for t in tw) / n,
            sum(bool(TAG_RE.search(t)) for t in tw) / n,
            sum(bool(MEN_RE.search(t)) for t in tw) / n,
            sum(t.startswith("RT @") for t in tw) / n,
            len(set(tw)) / n,
            np.mean([len(t) for t in tw]) / 140 if tw else 0,
            np.mean([len(URL_RE.findall(t)) for t in tw]) if tw else 0,
            np.mean([len(TAG_RE.findall(t)) for t in tw]) if tw else 0,
        ])
    return zscore(np.array(rows))


def profile_features(users) -> np.ndarray:
    import pandas as pd
    rows = []
    for u in users:
        p = u["profile"]
        fol, fri = to_num(p.get("followers_count")), to_num(p.get("friends_count"))
        try:
            age = (pd.Timestamp("2020-06-01", tz="UTC") -
                   pd.to_datetime(str(p.get("created_at")).strip(), utc=True)).days
        except Exception:
            age = 0
        rows.append([
            np.log1p(fol), np.log1p(fri), np.log1p(to_num(p.get("statuses_count"))),
            np.log1p(to_num(p.get("favourites_count"))), np.log1p(to_num(p.get("listed_count"))),
            to_bool(p.get("verified")), to_bool(p.get("default_profile")),
            to_bool(p.get("geo_enabled")), to_bool(p.get("protected")),
            np.log1p(len(str(p.get("description") or "").strip())),
            np.log1p(max(age, 0)), np.log1p(fol) - np.log1p(fri),
            1.0 if str(p.get("url") or "").strip() not in ("", "None") else 0.0,
        ])
    return zscore(np.array(rows))


# --------------------------------------------------------------------------
# Random-walk trust
# --------------------------------------------------------------------------
def random_walk_trust(n, directed, alpha=0.85, iters=50) -> np.ndarray:
    if len(directed) == 0:
        return np.full(n, 0.5, np.float32)
    A = sp.csr_matrix((np.ones(len(directed), np.float32), (directed[:, 0], directed[:, 1])), shape=(n, n))
    A.data[:] = 1.0
    out = np.asarray(A.sum(1)).ravel()
    inv = np.where(out > 0, 1.0 / np.maximum(out, 1), 0)
    P = sp.diags(inv) @ A
    dangling = (out == 0).astype(np.float32)
    r = np.full(n, 1.0 / n, np.float32)
    for _ in range(iters):
        r = alpha * (P.T @ r + dangling @ r / n) + (1 - alpha) / n
    from scipy.stats import rankdata
    return (rankdata(r) / n).astype(np.float32)


# --------------------------------------------------------------------------
# Signed weighted behavioural-similarity graph
# --------------------------------------------------------------------------
def build_signed_graph(cfg, sem_emb, url_mat, int_mat, soc_mat, links, trust):
    n = sem_emb.shape[0]
    rng = np.random.RandomState(cfg.seed)
    cand = []
    # 1) semantic kNN candidates
    k = min(cfg.knn_k + 1, n)
    nn_ = NearestNeighbors(n_neighbors=k, metric="cosine").fit(sem_emb)
    nbr = nn_.kneighbors(sem_emb, return_distance=False)
    cand.append(np.stack([np.repeat(np.arange(n), k), nbr.ravel()], 1))
    # 2) real interaction links
    if len(links):
        cand.append(links)
    # 3) random pairs -> dissimilar/negative edges
    if cfg.random_pairs_per_node:
        m = n * cfg.random_pairs_per_node
        cand.append(rng.randint(0, n, (m, 2)))
    c = np.concatenate(cand)
    c = c[c[:, 0] != c[:, 1]]
    c = np.sort(c, axis=1)
    c = np.unique(c, axis=0)
    src, dst = c[:, 0], c[:, 1]

    direct = set((links[:, 0] * n + links[:, 1]).tolist()) | set((links[:, 1] * n + links[:, 0]).tolist()) \
        if len(links) else set()
    is_link = np.array([(a * n + b) in direct for a, b in zip(src, dst)], np.float32)

    s_sem = np.clip((sem_emb[src] * sem_emb[dst]).sum(1), 0, 1)
    s_url = jaccard_pairs(url_mat, src, dst)
    s_int = jaccard_pairs(int_mat, src, dst)
    s_soc = np.maximum(jaccard_pairs(soc_mat, src, dst), 0.5 * is_link + 0.5 * jaccard_pairs(soc_mat, src, dst))
    sims = np.stack([s_sem, s_url, s_int, s_soc], 1).astype(np.float32)

    combined = sims @ np.array([cfg.w_sem, cfg.w_url, cfg.w_interest, cfg.w_social], np.float32)
    tau = cfg.sign_threshold
    t_edge = np.sqrt(trust[src] * trust[dst])
    w = (combined - tau) / max(tau, 1 - tau) * (0.5 + 0.5 * t_edge)
    return np.stack([src, dst], 1).astype(np.int64), w.astype(np.float32), sims
