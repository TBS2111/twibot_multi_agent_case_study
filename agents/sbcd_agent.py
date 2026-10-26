"""SBCD: Social Botnet Community Detection on the signed behavioural-similarity graph."""
import networkx as nx
import numpy as np


def relabel(l):
    return np.unique(l, return_inverse=True)[1]


def absorb_small(labels, emb, min_size):
    """Assign members of tiny communities to the nearest large community (by centroid)."""
    labels = relabel(labels)
    sizes = np.bincount(labels)
    big = np.where(sizes >= min_size)[0]
    if len(big) == 0:
        return np.zeros_like(labels)
    small = ~np.isin(labels, big)
    if small.any():
        cent = np.stack([emb[labels == c].mean(0) for c in big])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
        e = emb[small] / (np.linalg.norm(emb[small], axis=1, keepdims=True) + 1e-8)
        labels[small] = big[np.argmax(e @ cent.T, 1)]
    return relabel(labels)


class SBCDAgent:
    name = "sbcd"

    def run(self, state):
        cfg, emb = state["cfg"], state["emb"]
        n, edges, w = state["n_nodes"], state["edges"], state["edge_w"]
        src, dst = edges[:, 0], edges[:, 1]
        cos = (emb[src] * emb[dst]).sum(1)
        pos = w > 0
        ew = w[pos] * np.clip((1 + cos[pos]) / 2, 1e-3, 1)       # similarity-weighted positive edges
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_weighted_edges_from(zip(src[pos].tolist(), dst[pos].tolist(), ew.astype(float).tolist()))
        parts = nx.community.louvain_communities(G, weight="weight",
                                                 resolution=cfg.louvain_resolution, seed=cfg.seed)
        labels = np.zeros(n, dtype=np.int64)
        for c, nodes in enumerate(parts):
            labels[list(nodes)] = c
        labels = absorb_small(labels, emb, cfg.min_comm_size)
        return {"initial_labels": labels,
                "log": [f"[sbcd] louvain -> {len(parts)} raw, {labels.max() + 1} after absorbing small"]}
