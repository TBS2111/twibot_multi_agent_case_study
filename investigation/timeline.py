"""Stage 1: correlate events and build a timeline. Deterministic code + SQL only.

Every output is an Evidence record: a typed bag of facts plus the name of the SQL that made it.
Same DB + same Query + same Params  =>  byte-identical evidence (see `digest`).

IMPORTANT LIMIT: TwiBot-20 tweets carry no timestamps, so the only real time axis is account
creation (profile.created_at). Shared URLs / hashtags / templates are structural evidence.
"""
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from investigation.query_parser import Query


@dataclass
class Params:
    min_accounts: int = 3        # an item is "shared" if >= this many distinct accounts use it
    top_k: int = 3               # items kept per kind
    burst_min: int = 3           # accounts created on one day to count as a burst
    window_days: int = 7         # creation-concentration window
    min_template_len: int = 20   # chars of normalised tweet text to count as a template


@dataclass
class Evidence:
    id: str
    kind: str
    community: int
    ts: Optional[str]            # ISO timestamp if the fact is time-anchored, else None
    facts: Dict[str, Any]
    source: str                  # key into SQL (auditability)


SQL = {
    "select": """SELECT community, COUNT(*) n, SUM(label>=0) labeled,
                        AVG(CASE WHEN label>=0 THEN label END) ratio
                 FROM users GROUP BY community""",
    "summary": """SELECT COUNT(*), SUM(label>=0), SUM(label=1), SUM(label=0)
                  FROM users WHERE community=?""",
    "internal": """SELECT COUNT(*), SUM(e.w>0), SUM(e.w<0), AVG(e.w), AVG(e.sem), AVG(e.url),
                          AVG(e.interest), AVG(e.social), SUM(CASE WHEN e.w>0 THEN e.w ELSE 0 END)
                   FROM edges e JOIN users a ON a.idx=e.src JOIN users b ON b.idx=e.dst
                   WHERE a.community=? AND b.community=?""",
    "boundary": """SELECT SUM(e.w>0), SUM(CASE WHEN e.w>0 THEN e.w ELSE 0 END)
                   FROM edges e JOIN users a ON a.idx=e.src JOIN users b ON b.idx=e.dst
                   WHERE (a.community=?) <> (b.community=?) AND (a.community=? OR b.community=?)""",
    "created": """SELECT created_at FROM users WHERE community=? AND created_at IS NOT NULL
                  ORDER BY created_at, id""",
    "burst": """SELECT substr(created_at,1,10) d, COUNT(*) n,
                       (SELECT COUNT(*) FROM users x WHERE substr(x.created_at,1,10)=substr(u.created_at,1,10))
                FROM users u WHERE community=? AND created_at IS NOT NULL
                GROUP BY d HAVING n>=? ORDER BY n DESC, d LIMIT ?""",
    "shared_url": """SELECT t.url, COUNT(DISTINCT t.user_id) a, SUM(t.n) p,
                            (SELECT COUNT(DISTINCT user_id) FROM tweet_urls x WHERE x.url=t.url)
                     FROM tweet_urls t JOIN users u ON u.idx=t.user_id WHERE u.community=?
                     GROUP BY t.url HAVING a>=? ORDER BY a DESC, p DESC, t.url LIMIT ?""",
    "shared_hashtag": """SELECT t.tag, COUNT(DISTINCT t.user_id) a, SUM(t.n) p,
                                (SELECT COUNT(DISTINCT user_id) FROM tweet_tags x WHERE x.tag=t.tag)
                         FROM tweet_tags t JOIN users u ON u.idx=t.user_id WHERE u.community=?
                         GROUP BY t.tag HAVING a>=? ORDER BY a DESC, p DESC, t.tag LIMIT ?""",
    "template_tweet": """SELECT t.norm, COUNT(DISTINCT t.user_id) a, COUNT(*) p,
                                (SELECT COUNT(DISTINCT user_id) FROM tweets x WHERE x.norm=t.norm)
                         FROM tweets t JOIN users u ON u.idx=t.user_id
                         WHERE u.community=? AND length(t.norm)>=?
                         GROUP BY t.norm HAVING a>=? ORDER BY a DESC, p DESC, t.norm LIMIT ?""",
    "profile": """SELECT followers, friends, statuses, default_profile, verified
                  FROM users WHERE community=? ORDER BY idx""",
    "profile_all": "SELECT followers, friends, statuses FROM users ORDER BY idx",
    "sample": "SELECT id FROM users WHERE community=? ORDER BY id LIMIT 5",
}


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def select_communities(con, q: Query) -> List[int]:
    if q.community is not None:
        row = con.execute("SELECT 1 FROM users WHERE community=? LIMIT 1", (q.community,)).fetchone()
        if not row:
            raise ValueError(f"community {q.community} does not exist in this database")
        return [q.community]
    rows = [r for r in con.execute(SQL["select"]).fetchall() if r[1] >= q.min_size]
    if q.min_bot_ratio is not None:
        rows = [r for r in rows if r[3] is not None and r[3] >= q.min_bot_ratio]
    if q.selector == "largest":
        rows.sort(key=lambda r: (-r[1], r[0]))
    else:
        rows = [r for r in rows if r[3] is not None]
        rows.sort(key=lambda r: (-r[3], -r[1], r[0]))
    return [r[0] for r in rows[:q.top_n]]


def _in_window(day, q):
    return (q.since is None or day >= q.since) and (q.until is None or day <= q.until)


def collect(con, q: Query, p: Params = Params(), communities=None) -> List[Evidence]:
    ev: List[Evidence] = []

    def add(kind, c, facts, ts=None):
        ev.append(Evidence("", kind, c, ts, facts, kind))

    gl = con.execute(SQL["profile_all"]).fetchall()
    base = {k: _median([r[i] for r in gl]) for i, k in enumerate(("followers", "friends", "statuses"))}

    for c in (communities if communities is not None else select_communities(con, q)):
        n, lab, bots, hum = con.execute(SQL["summary"], (c,)).fetchone()
        add("summary", c, dict(size=n, labeled=lab or 0, bots=bots or 0, humans=hum or 0,
                               bot_ratio=_r(bots / lab) if lab else None))

        ie, ip, ineg, imean, s1, s2, s3, s4, ipw = con.execute(SQL["internal"], (c, c)).fetchone()
        bp, bpw = con.execute(SQL["boundary"], (c, c, c, c)).fetchone()
        ipw, bpw = ipw or 0.0, bpw or 0.0
        add("cohesion", c, dict(internal_edges=ie, internal_positive=ip or 0, internal_negative=ineg or 0,
                                mean_internal_weight=_r(imean), sim_semantic=_r(s1), sim_url=_r(s2),
                                sim_interest=_r(s3), sim_social=_r(s4), boundary_positive=bp or 0,
                                cohesion=_r(ipw / (ipw + bpw)) if ipw + bpw > 0 else None))

        created = [r[0] for r in con.execute(SQL["created"], (c,)).fetchall()]
        if created:
            d0, d1 = date.fromisoformat(created[0][:10]), date.fromisoformat(created[-1][:10])
            add("creation_window", c, dict(first=created[0], last=created[-1], dated_accounts=len(created),
                                           span_days=(d1 - d0).days), ts=created[0])
            days = [date.fromisoformat(x[:10]) for x in created]
            best, start, j = (0, None), None, 0
            for i in range(len(days)):
                while (days[i] - days[j]).days >= p.window_days:
                    j += 1
                if i - j + 1 > best[0]:
                    best = (i - j + 1, days[j].isoformat())
            add("creation_concentration", c, dict(window_days=p.window_days, max_accounts=best[0],
                                                  window_start=best[1], share=_r(best[0] / len(days))),
                ts=best[1] + "T00:00:00")
            for d, k, g_ in con.execute(SQL["burst"], (c, p.burst_min, 50)).fetchall():
                if _in_window(d, q):
                    add("creation_burst", c, dict(day=d, accounts=k, global_accounts=g_), ts=d + "T00:00:00")

        for kind, args in (("shared_url", (c, p.min_accounts, p.top_k)),
                           ("shared_hashtag", (c, p.min_accounts, p.top_k)),
                           ("template_tweet", (c, p.min_template_len, p.min_accounts, p.top_k))):
            for item, a, posts, glob in con.execute(SQL[kind], args).fetchall():
                add(kind, c, dict(item=item.replace("`", "'")[:90], accounts=a, posts=posts, global_accounts=glob))

        prof = con.execute(SQL["profile"], (c,)).fetchall()
        add("profile", c, dict(
            median_followers=_median([r[0] for r in prof]), median_friends=_median([r[1] for r in prof]),
            median_statuses=_median([r[2] for r in prof]), global_median_followers=base["followers"],
            global_median_friends=base["friends"], global_median_statuses=base["statuses"],
            default_profile_accounts=sum(r[3] or 0 for r in prof), verified_accounts=sum(r[4] or 0 for r in prof)))
        add("sample_accounts", c, dict(ids=[r[0] for r in con.execute(SQL["sample"], (c,)).fetchall()]))

    for i, e in enumerate(ev, 1):
        e.id = f"E{i}"
    return ev


def digest(evidence: List[Evidence]) -> str:
    blob = json.dumps([asdict(e) for e in evidence], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def evidence_from_json(items) -> List[Evidence]:
    return [Evidence(**d) for d in items]


def build_timeline(evidence: List[Evidence]):
    """Time-anchored evidence only, chronological, ties broken by kind then id (stable)."""
    anchored = [e for e in evidence if e.ts]
    return sorted(anchored, key=lambda e: (e.ts, e.kind, int(e.id[1:])))
