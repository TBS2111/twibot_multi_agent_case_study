"""Semantic agent: tweet-content embeddings.

backend "hf"            -> HuggingFace transformer (BERT, BERTweet, XLM-R, MiniLM, ...).
                           Every tweet is embedded (mean/CLS pooling), then averaged per user.
backend "tfidf"/"sbert" -> original behaviour (offline TF-IDF+SVD / sentence-transformers).
If the HF model cannot be loaded (no internet, wrong name) it falls back to TF-IDF and says so.
"""
import re

import numpy as np
import torch
from sklearn.decomposition import PCA

from preprocessing.features import MEN_RE, URL_RE, l2norm, semantic_embeddings


def _norm_tweet(t: str) -> str:
    """Light normalisation (keeps case; BERTweet-style placeholders)."""
    t = URL_RE.sub("http", t)
    t = MEN_RE.sub("@user", t)
    return re.sub(r"\s+", " ", t).strip()


def _pick_device(cfg) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"                      # Apple-silicon MacBooks
    return "cpu"


@torch.no_grad()
def _embed_texts(texts, cfg, device):
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.hf_model)
    model = AutoModel.from_pretrained(cfg.hf_model).to(device).eval()
    order = np.argsort([len(t) for t in texts])          # similar lengths per batch = faster
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    for s in tqdm(range(0, len(texts), cfg.hf_batch), desc=f"[semantic] {cfg.hf_model}", unit="batch"):
        idx = order[s:s + cfg.hf_batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=cfg.hf_max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        if cfg.hf_pooling == "cls":
            v = h[:, 0]
        else:                                            # masked mean pooling
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[idx] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
    return out


def hf_user_embeddings(users, cfg):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(cfg.hf_model)).strip("_")[-60:]
    cache_path = cfg.ckpt_dir / f"sem_cache_{slug}_{cfg.hf_tweets_per_user}_{cfg.hf_max_len}_{cfg.hf_pooling}.npz"
    cache = {}
    if cfg.hf_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["emb"]))

    todo = [i for i, u in enumerate(users) if u["id"] not in cache]
    if todo:
        per_user = []
        for i in todo:
            ts = [_norm_tweet(t) for t in users[i]["tweets"][:cfg.hf_tweets_per_user]]
            per_user.append([t for t in ts if len(t) >= 3])
        uniq = {}                                        # identical tweets (bots!) embedded once
        for ts in per_user:
            for t in ts:
                uniq.setdefault(t, len(uniq))
        if not uniq and not cache:
            raise ValueError("no tweets to embed")
        if uniq:
            E = _embed_texts(list(uniq), cfg, _pick_device(cfg))
            dim = E.shape[1]
        else:
            dim = len(next(iter(cache.values())))
        for i, ts in zip(todo, per_user):
            cache[users[i]["id"]] = E[[uniq[t] for t in ts]].mean(0) if ts else np.zeros(dim, np.float32)
        if cfg.hf_cache:
            ids = np.array(list(cache.keys()))
            np.savez(cache_path, ids=ids, emb=np.stack([cache[k] for k in ids]))

    emb = np.stack([cache[u["id"]] for u in users]).astype(np.float32)
    # Transformer embeddings are anisotropic (all cosines high) -> centre before normalising,
    # otherwise every pair looks "similar" and the signed graph loses its negative edges.
    emb = emb - emb.mean(0, keepdims=True)
    if cfg.sem_pca_dim and emb.shape[1] > cfg.sem_pca_dim and len(users) > cfg.sem_pca_dim:
        emb = PCA(cfg.sem_pca_dim, random_state=cfg.seed).fit_transform(emb)
    return l2norm(emb).astype(np.float32)


class SemanticAgent:
    """Tweet-content similarity viewpoint -> per-user semantic embedding."""
    name = "semantic"

    def run(self, state):
        cfg, users = state["cfg"], state["users"]
        log = []
        if cfg.semantic_backend == "hf":
            try:
                emb = hf_user_embeddings(users, cfg)
                log.append(f"[semantic] HF model '{cfg.hf_model}' ({cfg.hf_pooling} pooling)")
            except Exception as e:                       # offline / bad model name / no transformers
                log.append(f"[semantic] WARNING: HF backend failed ({type(e).__name__}: {e}) "
                           f"-> falling back to TF-IDF")
                emb = semantic_embeddings(users, cfg)
        else:
            emb = semantic_embeddings(users, cfg)
        rng = np.random.RandomState(cfg.seed)
        a, b = rng.randint(0, len(users), (2, 2000))
        cos = (emb[a] * emb[b]).sum(1)
        log.append(f"[semantic] embeddings {emb.shape}; random-pair cosine mean={cos.mean():.3f} "
                   f"p90={np.percentile(cos, 90):.3f}")
        return {"sem_emb": emb, "log": log}
