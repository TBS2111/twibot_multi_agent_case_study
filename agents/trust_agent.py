"""Trust agent: Dempster-Shafer evidence-based trust model.

Frame of discernment  Theta = {T (trustworthy), U (untrustworthy)}.
Every evidence source gives a mass function  m = (m(T), m(U), m(Theta)):
    m(Theta) = "I don't know"  -> missing / weak evidence adds uncertainty, not wrong certainty.
Sources (all unsupervised, no labels): followers/friends ratio, audience size, verified flag,
account age, posting rate, profile completeness, engagement, and graph evidence (random walk).
Masses are fused with Dempster's rule of combination.
Output per user:  Bel(T)=m(T) | Pl(T)=m(T)+m(Theta) | uncertainty=m(Theta) | conflict | trust=BetP(T)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing import features as F
from preprocessing.dataset import to_bool, to_num

SOURCES = ["followers_friends", "audience", "verified", "account_age",
           "activity_rate", "completeness", "engagement", "graph"]
REF_DATE = pd.Timestamp("2020-06-01", tz="UTC")          # TwiBot-20 crawl time


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_mass(score, r):
    """score in [0,1] (1 = trustworthy), r = reliability of the source. NaN -> total ignorance."""
    s = np.nan_to_num(score, nan=0.5)
    mT, mU = r * s, r * (1 - s)
    mO = 1 - r * np.ones_like(s)
    miss = np.isnan(score)
    return np.where(miss, 0, mT), np.where(miss, 0, mU), np.where(miss, 1, mO)


def dempster(m1, m2, eps=1e-6):
    """Dempster's rule on the 2-element frame. Returns fused masses and (1-K) normaliser."""
    T1, U1, O1 = m1
    T2, U2, O2 = m2
    K = T1 * U2 + U1 * T2                                 # conflict between the two sources
    norm = 1 - K
    bad = norm < eps                                      # total conflict -> fall back to ignorance
    d = np.where(bad, 1.0, norm)
    T = np.where(bad, 0, (T1 * T2 + T1 * O2 + O1 * T2) / d)
    U = np.where(bad, 0, (U1 * U2 + U1 * O2 + O1 * U2) / d)
    O = np.where(bad, 1, O1 * O2 / d)
    return (T, U, O), np.where(bad, eps, norm)


def combine_all(masses):
    """Fold Dempster's rule over all sources; also return overall conflict 1 - prod(1-K)."""
    fused, keep = masses[0], np.ones_like(masses[0][0])
    for m in masses[1:]:
        fused, norm = dempster(fused, m)
        keep = keep * norm
    return fused, 1 - keep


def _col(users, key):
    return np.array([to_num(u["profile"].get(key)) for u in users], dtype=np.float64)


def _age_days(users):
    out = []
    for u in users:
        try:
            out.append(max((REF_DATE - pd.to_datetime(str(u["profile"].get("created_at")).strip(), utc=True)).days, 0))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=np.float64)


def profile_evidence(users, rel):
    """Map profile fields to Dempster-Shafer mass functions."""
    fol, fri = _col(users, "followers_count"), _col(users, "friends_count")
    stat, fav, lst = _col(users, "statuses_count"), _col(users, "favourites_count"), _col(users, "listed_count")
    age = _age_days(users)
    ver = np.array([to_bool(u["profile"].get("verified")) for u in users])

    ratio = np.log1p(fol) - np.log1p(fri)                 # followers vs following (friends)
    s_ratio = sigmoid(0.8 * (ratio + 1.0))
    s_aud = sigmoid(0.6 * (np.log1p(fol) - 4.0))          # ~55 followers = neutral
    s_age = sigmoid(1.0 * (np.log1p(age) - np.log(180.0)))  # NaN stays NaN
    rate = stat / (age + 1.0)                             # tweets per day
    s_rate = 1 - sigmoid(1.5 * (np.log1p(rate) - np.log(50.0)))
    comp = []
    for u in users:
        p = u["profile"]
        has = lambda k: 1.0 if str(p.get(k) or "").strip() not in ("", "None") else 0.0
        comp.append(np.mean([has("description"), has("url"), has("location"),
                             1 - to_bool(p.get("default_profile")), 1 - to_bool(p.get("default_profile_image"))]))
    s_comp = np.array(comp)
    s_eng = sigmoid(0.5 * (np.log1p(fav) + 2 * np.log1p(lst) - 4.0))

    scores = {"followers_friends": s_ratio, "audience": s_aud, "account_age": s_age,
              "activity_rate": s_rate, "completeness": s_comp, "engagement": s_eng}
    masses = {k: soft_mass(v, rel[k]) for k, v in scores.items()}
    r = rel["verified"]                                   # verified = strong evidence FOR trust;
    masses["verified"] = (np.where(ver > 0, 0.9 * r, 0.0),   # unverified = only weak evidence against
                          np.where(ver > 0, 0.0, 0.2 * r),
                          np.where(ver > 0, 1 - 0.9 * r, 1 - 0.2 * r))
    scores["verified"] = np.where(ver > 0, 1.0, 0.3)
    return scores, masses


class TrustAgent:
    name = "trust"

    def run(self, state):
        users, cfg = state["users"], state["cfg"]
        n = len(users)
        rel = dict(cfg.ds_reliability)
        scores, masses = profile_evidence(users, rel)

        # graph evidence: random-walk percentile; isolated users give no evidence (ignorance)
        links = F.link_edges(users)
        rw = F.random_walk_trust(n, links, cfg.trust_alpha, cfg.trust_iters).astype(np.float64)
        deg = np.zeros(n)
        if len(links):
            np.add.at(deg, links[:, 0], 1)
            np.add.at(deg, links[:, 1], 1)
        rw = np.where(deg > 0, rw, np.nan)
        scores["graph"] = rw
        masses["graph"] = soft_mass(rw, rel["graph"])

        (mT, mU, mO), conflict = combine_all([masses[k] for k in SOURCES])
        belief, plaus, unc = mT, mT + mO, mO
        trust = (mT + 0.5 * mO).astype(np.float32)        # pignistic probability BetP(T)

        src = np.stack([np.nan_to_num(scores[k], nan=0.5) for k in SOURCES], 1)
        feat = np.concatenate([np.stack([trust, belief, plaus, unc, conflict], 1), src,
                               F.profile_features(users)], 1).astype(np.float32)
        pd.DataFrame({"id": [u["id"] for u in users], "trust": trust, "belief": belief,
                      "plausibility": plaus, "uncertainty": unc, "conflict": conflict,
                      **{f"s_{k}": np.nan_to_num(scores[k], nan=0.5) for k in SOURCES}}
                     ).to_csv(cfg.out_dir / "trust_ds.csv", index=False)

        log = [f"[trust] Dempster-Shafer: trust mean={trust.mean():.3f} std={trust.std():.3f} | "
               f"uncertainty mean={unc.mean():.3f} | conflict mean={conflict.mean():.3f}"]
        y = state.get("labels_true")
        if y is not None and (y >= 0).sum() > 10 and len(np.unique(y[y >= 0])) == 2:
            auc = roc_auc_score(y[y >= 0], 1 - trust[y >= 0])
            log.append(f"[trust] diagnostic only (labels NOT used): AUC of (1 - trust) for bot = {auc:.3f}")
        return {"trust": trust, "trust_feat": feat, "log": log}
