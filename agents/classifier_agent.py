"""Classifier agent: bot-vs-human detection with an ACTOR-CRITIC deep-RL policy = Stable-Baselines3 PPO.
The agent WALKS the signed graph (9 actions: 7 moves between accounts + stop-human / stop-bot) before deciding
about its origin account; reward = step cost per move, +1 / -1 for the final decision (see models/actor_critic.py).
Features come from the other agents (D-S trust + profile, behaviour, semantic, graph/community).
Policy is trained on `train`, the best epoch is picked on `dev` (selection reward), metrics are
reported on `test`. A Gradient-Boosting baseline on the same features is reported for comparison."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

from models.actor_critic import ACTION_NAMES, WalkGraph, train_ppo, walk_predict   # SB3 PPO graph-walk actor-critic


def neighbor_mean(emb, edges, w, n):
    """Mean embedding of positively-linked neighbours (graph context)."""
    p = w > 0
    A = sp.coo_matrix((w[p], (edges[p, 0], edges[p, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(1)).ravel()[:, None] + 1e-8
    return np.asarray((A @ emb) / deg, dtype=np.float32)


def community_features(labels, y, train_idx, n_splits=5, seed=0):
    """Community bot-ratio + size. Train nodes get out-of-fold ratios (no label leakage)."""
    K = labels.max() + 1
    prior = y[train_idx].mean()
    size = np.bincount(labels, minlength=K)

    def ratio(idx):
        pos = np.bincount(labels[idx], weights=y[idx], minlength=K)
        cnt = np.bincount(labels[idx], minlength=K)
        return (pos + 2 * prior) / (cnt + 2)

    feat = np.zeros((len(labels), 2), np.float32)
    feat[:, 0] = ratio(train_idx)[labels]
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(train_idx):
        feat[train_idx[b], 0] = ratio(train_idx[a])[labels[train_idx[b]]]
    feat[:, 1] = np.log1p(size[labels])
    return feat


def _scores(y, pred, prob):
    m = dict(acc=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))
    try:
        m["auc"] = roc_auc_score(y, prob)
    except ValueError:
        pass
    return {k: float(v) for k, v in m.items()}


class ClassifierAgent:
    name = "classifier"

    def run(self, state):
        cfg = state["cfg"]
        y, sp_, n = state["labels_true"], state["splits"], state["n_nodes"]
        lab = y >= 0
        tr = np.where((sp_ == "train") & lab)[0]
        dev = np.where((sp_ == "dev") & lab)[0]
        test = np.where((sp_ == "test") & lab)[0]
        ev = test if len(test) else dev
        if len(tr) < 20 or len(ev) < 5 or len(np.unique(y[tr])) < 2:
            return {"clf_metrics": {}, "log": ["[classifier] skipped: need labeled train + dev/test users"]}

        # policy-selection set: dev if it is separate from the eval set, else hold out 15% of train
        rng = np.random.RandomState(cfg.seed)
        if len(test) and len(dev) >= 20:
            fit, sel = tr, dev
        else:
            perm = rng.permutation(tr)
            k = max(10, int(0.15 * len(tr)))
            sel, fit = perm[:k], perm[k:]

        comm = community_features(state["refined_labels"], y, tr, seed=cfg.seed)
        neigh = neighbor_mean(state["emb"], state["edges"], state["edge_w"], n)
        blocks = [("D-S trust + profile", state["trust_feat"]),
                  ("+ behavior", state["beh_feat"]),
                  ("+ semantic", state["sem_emb"]),
                  ("+ graph (GBT-CMA, neighbours, communities)", np.concatenate([state["emb"], neigh, comm], 1))]

        G = WalkGraph(n, state["edges"], state["edge_w"], state["trust"], state["refined_labels"])   # label-free
        base = max(y[ev].mean(), 1 - y[ev].mean())
        log = [f"[classifier] PPO graph-walk actor-critic (Stable-Baselines3) | fit={len(fit)} select={len(sel)} eval={len(ev)} "
               f"majority-baseline acc={base:.3f}"]
        metrics, cur = {}, []
        model = mean = std = None
        for name, Xb in blocks:
            cur.append(Xb)
            Xa = np.concatenate(cur, 1).astype(np.float32)
            mean, std = Xa[fit].mean(0), Xa[fit].std(0) + 1e-6          # scale with fit-set statistics only
            Xs = (Xa - mean) / std
            model, info = train_ppo(Xs, y, fit, sel, G, cfg)
            prob, walk = walk_predict(model, Xs, G, ev, cfg)
            m = _scores(y[ev], (prob >= 0.5).astype(int), prob)
            metrics[f"BOT DETECTION | {name} (PPO graph-walk actor-critic)"] = m
            log.append(f"[classifier] {name:45s} acc={m['acc']:.3f} f1={m['f1']:.3f} auc={m.get('auc', float('nan')):.3f}"
                       f"  | avg moves={walk['avg_moves']:.2f} (best policy at {info['best_epoch']}k of {info['epochs_run']}k steps)")

        log.append(f"[classifier] RL setup: algo={info['algo']} (Stable-Baselines3 {info['sb3']}, MlpPolicy actor-critic) | "
                   f"obs_dim={info['obs_dim']} (3 x {Xa.shape[1]} features + extras) | actions=9 ({', '.join(ACTION_NAMES)}) | "
                   f"reward: step={cfg.rl_step_cost:+g}, stop correct={cfg.rl_reward_correct:+g}, FP={cfg.rl_penalty_fp:+g}, "
                   f"FN={cfg.rl_penalty_fn:+g}, timeout={cfg.rl_timeout_penalty:+g}, max moves={cfg.rl_max_moves} | "
                   f"actor params={info['actor_params']:,}, critic params={info['critic_params']:,}")
        metrics["RL WALK | policy behaviour on eval set (full features)"] = {
            "avg_moves": walk["avg_moves"], "pct_no_move": walk["pct_no_move"],
            **{f"use_{nm}": float(u) for nm, u in zip(ACTION_NAMES, walk["usage"])}}
        log.append("[classifier] action usage on eval set: " + ", ".join(
            f"{nm}={100 * u:.0f}%" for nm, u in zip(ACTION_NAMES, walk["usage"])))

        # comparison baseline: gradient boosting on the same (full) feature set
        Xf = np.concatenate(cur, 1)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                            l2_regularization=1.0, random_state=cfg.seed).fit(Xf[tr], y[tr])
        pg = gb.predict_proba(Xf[ev])[:, 1]
        mg = _scores(y[ev], (pg >= 0.5).astype(int), pg)
        metrics["BASELINE | Gradient Boosting (all features)"] = mg
        log.append(f"[classifier] baseline GradientBoosting (all features)          acc={mg['acc']:.3f} "
                   f"f1={mg['f1']:.3f} auc={mg.get('auc', float('nan')):.3f}")

        # final policy: predictions for every user + checkpoint
        prob_all, _ = walk_predict(model, Xs, G, np.arange(n), cfg)
        pd.DataFrame({"id": [u["id"] for u in state["users"]], "split": sp_, "label": y,
                      "bot_prob": prob_all, "pred": (prob_all >= 0.5).astype(int),
                      "community": state["refined_labels"]}).to_csv(cfg.out_dir / "bot_predictions.csv", index=False)
        model.save(str(cfg.ckpt_dir / "classifier_ppo"))
        np.savez(cfg.ckpt_dir / "classifier_scaler.npz", mean=mean, std=std)
        return {"clf_metrics": metrics, "log": log}
