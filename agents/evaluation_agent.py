"""Evaluation: dataset class counts, community metrics (modularity, signed frustration, NMI/ARI/purity)
and stratified K-fold cross-validation (k=5 and k=10, reported separately)."""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score, precision_score, recall_score)


def modularity(labels, src, dst, w, n):
    p = w > 0
    A = sp.coo_matrix((w[p], (src[p], dst[p])), shape=(n, n))
    A = (A + A.T).tocsr()
    two_m = A.sum()
    if two_m <= 0:
        return 0.0
    K = labels.max() + 1
    H = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
    inner = np.asarray((H.T @ A @ H).diagonal())
    deg = np.asarray(H.T @ np.asarray(A.sum(1)).ravel())
    return float((inner / two_m - (deg / two_m) ** 2).sum())


def frustration(labels, src, dst, w):
    same = labels[src] == labels[dst]
    return float(np.abs(w[same & (w < 0)]).sum() + w[(~same) & (w > 0)].sum())


def purity(labels, y):
    return sum(np.bincount(y[labels == c]).max() for c in np.unique(labels)) / len(y)


def compute_metrics(stages, y, splits, src, dst, w, n):
    out = {}
    for name, lab in stages.items():
        if lab is None:
            continue
        m = dict(n_communities=int(lab.max() + 1),
                 largest=int(np.bincount(lab).max()),
                 modularity=modularity(lab, src, dst, w, n),
                 frustration=frustration(lab, src, dst, w))
        lm = y >= 0
        if lm.sum() > 1:
            m.update(NMI=float(normalized_mutual_info_score(y[lm], lab[lm])),
                     ARI=float(adjusted_rand_score(y[lm], lab[lm])),
                     purity=float(purity(lab[lm], y[lm])))
            # community-level bot prediction: bot ratio estimated from TRAIN members only
            tr = (splits == "train") & lm
            ev = (splits != "train") & lm
            if tr.sum() and ev.sum():
                prior = y[tr].mean()
                ratio = np.full(lab.max() + 1, prior)
                for c in np.unique(lab[tr]):
                    ratio[c] = y[tr & (lab == c)].mean()
                pred = (ratio[lab[ev]] >= 0.5).astype(int)
                m.update(acc=float(accuracy_score(y[ev], pred)),
                         precision=float(precision_score(y[ev], pred, zero_division=0)),
                         recall=float(recall_score(y[ev], pred, zero_division=0)),
                         f1=float(f1_score(y[ev], pred, zero_division=0)))
        out[name] = m
    return out


def community_report(users, lab, y):
    df = pd.DataFrame({"id": [u["id"] for u in users], "community": lab, "label": y})
    rep = df.groupby("community").agg(size=("id", "size"),
                                      labeled=("label", lambda s: int((s >= 0).sum())),
                                      bot_ratio=("label", lambda s: float(s[s >= 0].mean()) if (s >= 0).any() else np.nan))
    return rep.sort_values("size", ascending=False)

from agents.classifier_agent import _scores, community_features, neighbor_mean
from models.actor_critic import WalkGraph, train_ppo, walk_predict

CV_KEYS = ["acc", "precision", "recall", "f1", "auc"]


# --------------------------------------------------------------------------
# dataset class counts
# --------------------------------------------------------------------------
def dataset_counts(y, splits):
    lab = y >= 0
    n_lab, bots, hum = int(lab.sum()), int((y == 1).sum()), int((y == 0).sum())
    out = {"DATASET | labeled users (all splits)": dict(
        labeled=n_lab, bots=bots, humans=hum,
        bot_pct=100.0 * bots / max(n_lab, 1), human_pct=100.0 * hum / max(n_lab, 1),
        unlabeled=int((~lab).sum()))}
    for sname in ("train", "dev", "test", "support"):
        m = splits == sname
        if m.any():
            out[f"DATASET | {sname}"] = dict(users=int(m.sum()), bots=int((m & (y == 1)).sum()),
                                             humans=int((m & (y == 0)).sum()), unlabeled=int((m & ~lab).sum()))
    return out


# --------------------------------------------------------------------------
# stratified K-fold cross-validation (all labeled users)
# --------------------------------------------------------------------------
def _static_features(state):
    neigh = neighbor_mean(state["emb"], state["edges"], state["edge_w"], state["n_nodes"])
    return np.concatenate([state["trust_feat"], state["beh_feat"], state["sem_emb"],
                           state["emb"], neigh], 1).astype(np.float32)


def cross_validate(state, static, k, G):
    """Every fold: community bot-ratio feature, scaler, policy selection set and models use the
    TRAINING fold only (no leakage). Unsupervised agents (GBT-CMA, SBCD, ICR, trust) never see labels."""
    cfg, y = state["cfg"], state["labels_true"]
    idx = np.where(y >= 0)[0]
    skf = StratifiedKFold(k, shuffle=True, random_state=cfg.seed)
    folds = []
    for f, (a, b) in enumerate(skf.split(idx, y[idx]), 1):
        tr, te = idx[a], idx[b]
        X = np.concatenate([static, community_features(state["refined_labels"], y, tr, seed=cfg.seed)], 1)
        perm = np.random.RandomState(cfg.seed + f).permutation(tr)
        ks = max(10, int(0.15 * len(tr)))
        sel, fit = perm[:ks], perm[ks:]
        mean, std = X[fit].mean(0), X[fit].std(0) + 1e-6
        Xs = (X - mean) / std
        model, _ = train_ppo(Xs, y, fit, sel, G, cfg)
        pa, _ = walk_predict(model, Xs, G, te, cfg)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                            l2_regularization=1.0, random_state=cfg.seed).fit(X[tr], y[tr])
        pg = gb.predict_proba(X[te])[:, 1]
        folds.append(dict(fold=f, n_train=len(tr), n_test=len(te),
                          bots_test=int((y[te] == 1).sum()), humans_test=int((y[te] == 0).sum()),
                          ac=_scores(y[te], (pa >= 0.5).astype(int), pa),
                          gb=_scores(y[te], (pg >= 0.5).astype(int), pg)))
    return folds


def summarize(folds, model):
    mean = {m: float(np.mean([f[model][m] for f in folds])) for m in CV_KEYS if m in folds[0][model]}
    std = {m: float(np.std([f[model][m] for f in folds])) for m in mean}
    return mean, std


def format_cv(k, folds, y):
    lab = y >= 0
    lines = [f"[evaluation] ===== {k}-fold stratified cross-validation on {int(lab.sum())} labeled users "
             f"(bots={int((y == 1).sum())}, humans={int((y == 0).sum())}) ====="]
    for name, key in (("PPO GRAPH-WALK ACTOR-CRITIC (Stable-Baselines3)", "ac"), ("GRADIENT BOOSTING (baseline)", "gb")):
        lines.append(f"  {name}")
        lines.append("    fold | test(bot/human) |  acc    prec   recall   f1     auc")
        for f in folds:
            m = f[key]
            lines.append(f"    {f['fold']:>4} | {f['n_test']:>5} ({f['bots_test']}/{f['humans_test']}) | "
                         f"{m['acc']:.4f} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f} {m.get('auc', float('nan')):.4f}")
        mean, std = summarize(folds, key)
        lines.append("    mean |                 | " + " ".join(f"{mean[m]:.4f}" for m in CV_KEYS if m in mean))
        lines.append("    std  |                 | " + " ".join(f"{std[m]:.4f}" for m in CV_KEYS if m in std))
    return "\n".join(lines)


class EvaluationAgent:
    name = "evaluation"

    def run(self, state):
        cfg, users = state["cfg"], state["users"]
        e, y, splits = state["edges"], state["labels_true"], state["splits"]
        log = []

        counts = dataset_counts(y, splits)
        c = counts["DATASET | labeled users (all splits)"]
        log.append(f"[evaluation] labeled data: {c['labeled']} users -> BOTS={c['bots']} ({c['bot_pct']:.1f}%), "
                   f"HUMANS={c['humans']} ({c['human_pct']:.1f}%), unlabeled={c['unlabeled']}")
        for kname, v in counts.items():
            if kname != "DATASET | labeled users (all splits)":
                log.append(f"[evaluation]   {kname.split('| ')[1]:8s} users={v['users']:6d}  bots={v['bots']:6d}  "
                           f"humans={v['humans']:6d}  unlabeled={v['unlabeled']}")

        stages = {"SBCD (initial)": state.get("initial_labels"),
                  "SBCD+ICR (refined)": state.get("refined_labels")}
        metrics = dict(counts)
        metrics.update(compute_metrics(stages, y, splits, e[:, 0], e[:, 1], state["edge_w"], state["n_nodes"]))
        metrics.update(state.get("clf_metrics", {}))

        out = cfg.out_dir
        rows = []
        if cfg.cv_enabled and (y >= 0).sum() >= 50:
            static = _static_features(state)
            G = WalkGraph(state["n_nodes"], e, state["edge_w"], state["trust"], state["refined_labels"])
            min_class = int(min((y == 0).sum(), (y == 1).sum()))
            for k in cfg.cv_folds:
                if min_class < k:
                    log.append(f"[evaluation] skipped {k}-fold CV: smallest class has only {min_class} users")
                    continue
                folds = cross_validate(state, static, k, G)
                log.append(format_cv(k, folds, y))
                for key, label in (("ac", "PPO graph-walk actor-critic"), ("gb", "gradient boosting")):
                    mean, std = summarize(folds, key)
                    metrics[f"CV k={k} | {label} | mean"] = mean
                    metrics[f"CV k={k} | {label} | std"] = std
                    for f in folds:
                        rows.append(dict(k=k, fold=f["fold"], model=label, n_test=f["n_test"],
                                         bots_test=f["bots_test"], humans_test=f["humans_test"], **f[key]))
            if rows:
                pd.DataFrame(rows).to_csv(out / "cv_folds.csv", index=False)

        pd.DataFrame({"id": [u["id"] for u in users], "label": y, "split": splits,
                      "initial": state["initial_labels"], "refined": state["refined_labels"]}
                     ).to_csv(out / "communities.csv", index=False)
        community_report(users, state["refined_labels"], y).to_csv(out / "community_report.csv")
        np.save(out / "embeddings.npy", state["emb"])
        np.savez(out / "graph.npz", edges=e, w=state["edge_w"], sims=state["edge_sims"])
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.append(f"[evaluation] saved results to {out}")
        return {"metrics": metrics, "log": log}
