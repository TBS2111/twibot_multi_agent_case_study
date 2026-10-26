"""ICR: Iterative Community Refinement  (Split -> Merge -> Reassign).
Objective: minimise signed frustration = |negative weight inside| + positive weight between."""
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from agents.sbcd_agent import absorb_small, relabel


def frustration(labels, src, dst, w):
    same = labels[src] == labels[dst]
    return float(np.abs(w[same & (w < 0)]).sum() + w[(~same) & (w > 0)].sum())


def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class ICRAgent:
    name = "icr"

    def split(self, labels, emb, src, dst, w, cfg):
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) < 2 * cfg.min_comm_size:
                continue
            inside = (labels[src] == c) & (labels[dst] == c)
            tot = np.abs(w[inside]).sum()
            neg = np.abs(w[inside & (w < 0)]).sum()
            if tot <= 0 or neg / tot <= cfg.split_neg_ratio:
                continue
            km = KMeans(2, n_init=5, random_state=cfg.seed).fit(emb[idx])
            if min(np.bincount(km.labels_)) < cfg.min_comm_size:
                continue
            new = labels.copy()
            new[idx[km.labels_ == 1]] = labels.max() + 1
            if frustration(new, src, dst, w) < frustration(labels, src, dst, w):
                labels = new
        return labels

    def merge(self, labels, emb, Wsym, cfg):
        labels = relabel(labels)
        n, K = len(labels), labels.max() + 1
        H = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
        B = (H.T @ Wsym @ H).toarray()
        cent = _norm(np.stack([emb[labels == c].mean(0) for c in range(K)]))
        cosc = cent @ cent.T
        cands = sorted(((B[a, b], a, b) for a in range(K) for b in range(a + 1, K)
                        if B[a, b] > 0 and cosc[a, b] > cfg.merge_cos_thr), reverse=True)
        used, mp = set(), np.arange(K)
        for _, a, b in cands:
            if a in used or b in used:
                continue
            mp[b] = a
            used |= {a, b}
        return relabel(mp[labels])

    def reassign(self, labels, emb, Wsym, src, dst, w, cfg):
        labels = relabel(labels)
        n, K = len(labels), labels.max() + 1
        H = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
        A = (Wsym @ H).toarray()
        deg = np.asarray(abs(Wsym).sum(1)).ravel()[:, None] + 1e-8
        cent = _norm(np.stack([emb[labels == c].mean(0) for c in range(K)]))
        score = A / deg + cfg.reassign_lambda * (_norm(emb) @ cent.T)
        ar = np.arange(n)
        best = score.argmax(1)
        gain = score[ar, best] - score[ar, labels]
        move = (best != labels) & (gain > 1e-3)
        if not move.any():
            return labels
        cap = int(cfg.reassign_max_frac * n)
        if move.sum() > cap:
            thr = np.sort(gain[move])[-cap]
            move &= gain >= thr
        new = labels.copy()
        new[move] = best[move]
        return new if frustration(new, src, dst, w) <= frustration(labels, src, dst, w) else labels

    def run(self, state):
        cfg, emb = state["cfg"], state["emb"]
        n, edges, w = state["n_nodes"], state["edges"], state["edge_w"]
        src, dst = edges[:, 0], edges[:, 1]
        W = sp.coo_matrix((w, (src, dst)), shape=(n, n))
        Wsym = (W + W.T).tocsr()
        labels = relabel(state["initial_labels"])
        trace = [dict(iter=0, k=int(labels.max() + 1), frustration=frustration(labels, src, dst, w))]
        for it in range(1, cfg.icr_max_iter + 1):
            prev = labels.copy()
            labels = relabel(self.split(labels, emb, src, dst, w, cfg))
            labels = self.merge(labels, emb, Wsym, cfg)
            labels = self.reassign(labels, emb, Wsym, src, dst, w, cfg)
            labels = absorb_small(labels, emb, cfg.min_comm_size)
            trace.append(dict(iter=it, k=int(labels.max() + 1), frustration=frustration(labels, src, dst, w)))
            if labels.shape == prev.shape and (relabel(labels) == relabel(prev)).all():
                break
        msg = " | ".join(f"it{t['iter']}: k={t['k']} F={t['frustration']:.1f}" for t in trace)
        return {"refined_labels": labels, "icr_trace": trace, "log": [f"[icr] {msg}"]}
