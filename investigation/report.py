"""Stage 4: write the narrative report. Template by default (deterministic); LLM rewrite is optional.

Format contract (what the validator enforces):
  * every bullet in Findings / Timeline ends with citations like [E3] or [E3][E7]
  * every number, date and `backticked` string in a bullet must exist in the cited evidence
  * Background bullets cite notes as [K:<note id>]; nothing from Background counts as evidence
"""
from collections import defaultdict

from investigation.timeline import build_timeline

INDICATOR_SHARE = 0.5      # creation concentration share that counts as an indicator
INDICATOR_COHESION = 0.8

EMPTY_TIMELINE = "_No time-anchored evidence matched._"

LIMITS = [
    "Tweets in TwiBot-20 carry no timestamps, so the timeline uses account creation dates only.",
    "Communities come from similarity clustering; similar behaviour is not proof of coordination.",
    "Cohesion is the share of positive edge weight touching the community that stays inside it.",
    "Background notes are general guidance and say nothing about this dataset.",
    "Labels are the dataset's ground truth and exist only for a subset of accounts.",
]


def f(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.2f}"
    return str(v)


def pct(v):
    return f"{v * 100:.0f}%"


def _refs(*ids):
    return "".join(f"[{i}]" for i in ids)


def _in_window(ts, q):
    d = ts[:10]
    return (q.since is None or d >= q.since) and (q.until is None or d <= q.until)


def findings(evidence):
    by_c = defaultdict(list)
    for e in evidence:
        by_c[e.community].append(e)
    out = []
    for c, items in by_c.items():
        out.append(f"\n### Community {c}")
        ind = []                                     # (label, evidence id)
        for e in items:
            k, x, r = e.kind, e.facts, _refs(e.id)
            if k == "summary":
                tail = (f"bot ratio {f(x['bot_ratio'])}" if x["bot_ratio"] is not None
                        else "no labelled members, so no bot ratio")
                out.append(f"- Community {c} has {x['size']} accounts; {x['labeled']} carry labels "
                           f"({x['bots']} bots, {x['humans']} humans), {tail}. {r}")
            elif k == "cohesion":
                out.append(f"- Similarity graph: {x['internal_edges']} internal edges ({x['internal_positive']} positive, "
                           f"{x['internal_negative']} negative); mean internal weight {f(x['mean_internal_weight'])}; "
                           f"mean similarity semantic {f(x['sim_semantic'])}, url {f(x['sim_url'])}, "
                           f"interest {f(x['sim_interest'])}, social {f(x['sim_social'])}; "
                           f"{x['boundary_positive']} positive edges leave the community; cohesion {f(x['cohesion'])}. {r}")
                if x["cohesion"] is not None and x["cohesion"] >= INDICATOR_COHESION:
                    ind.append(("high internal cohesion", e.id))
            elif k == "creation_window":
                out.append(f"- Accounts were created between {x['first']} and {x['last']} "
                           f"({x['dated_accounts']} dated accounts, span {x['span_days']} days). {r}")
            elif k == "creation_concentration":
                out.append(f"- At most {x['max_accounts']} accounts were created within any {x['window_days']}-day window "
                           f"starting {x['window_start']} ({pct(x['share'])} of dated accounts). {r}")
                if x["share"] >= INDICATOR_SHARE:
                    ind.append(("concentrated account creation", e.id))
            elif k == "shared_url":
                out.append(f"- Link `{x['item']}` was posted by {x['accounts']} accounts of this community in "
                           f"{x['posts']} posts; {x['global_accounts']} accounts use it across the dataset. {r}")
                ind.append(("shared links", e.id))
            elif k == "shared_hashtag":
                out.append(f"- Hashtag `{x['item']}` was used by {x['accounts']} accounts of this community in "
                           f"{x['posts']} posts; {x['global_accounts']} accounts use it across the dataset. {r}")
                ind.append(("shared hashtags", e.id))
            elif k == "template_tweet":
                out.append(f"- Near-identical tweet text `{x['item']}` appears in {x['posts']} posts from "
                           f"{x['accounts']} accounts ({x['global_accounts']} accounts overall). {r}")
                ind.append(("repeated tweet text", e.id))
            elif k == "profile":
                out.append(f"- Median followers {f(x['median_followers'])} (dataset {f(x['global_median_followers'])}), "
                           f"friends {f(x['median_friends'])} (dataset {f(x['global_median_friends'])}), "
                           f"statuses {f(x['median_statuses'])} (dataset {f(x['global_median_statuses'])}); "
                           f"{x['default_profile_accounts']} accounts use the default profile and "
                           f"{x['verified_accounts']} are verified. {r}")
            elif k == "sample_accounts":
                out.append("- Example accounts: " + ", ".join(f"`{i}`" for i in x["ids"]) + f". {r}")
        summary_id = next(e.id for e in items if e.kind == "summary")
        if ind:
            labels = list(dict.fromkeys(l for l, _ in ind))
            ids = list(dict.fromkeys(i for _, i in ind))
            out.append(f"- Similarity signals observed: {'; '.join(labels)}. "
                       f"These are signals to verify, not proof of coordination. {_refs(*ids)}")
        else:
            out.append(f"- No similarity signal passed its threshold for this community. {_refs(summary_id)}")
    return out


def timeline_lines(evidence, q):
    events = []
    for e in build_timeline(evidence):
        x, c, r = e.facts, e.community, _refs(e.id)
        if e.kind == "creation_window":
            events += [(x["first"], 0, f"- {x['first']} — first account created in community {c}. {r}"),
                       (x["last"], 1, f"- {x['last']} — last account created in community {c}. {r}")]
        elif e.kind == "creation_concentration":
            events.append((e.ts, 2, f"- {x['window_start']} — start of the busiest {x['window_days']}-day creation "
                                    f"window in community {c}: {x['max_accounts']} accounts. {r}"))
        elif e.kind == "creation_burst":
            events.append((e.ts, 3, f"- {x['day']} — {x['accounts']} accounts of community {c} were created on this "
                                    f"day; {x['global_accounts']} accounts dataset-wide. {r}"))
    events = [ev for ev in events if _in_window(ev[0], q)]
    events.sort(key=lambda t: (t[0] if "T" in t[0] else t[0] + "T00:00:00", t[1], t[2]))
    return [t[2] for t in events] or [EMPTY_TIMELINE]


def header(q, digest):
    return ["# Bot community investigation", "",
            f"**Question:** {q.raw}", f"**Parsed query:** `{q.to_json()}`",
            f"**Evidence digest:** `{digest}`", ""]


def tail(notes):
    L = ["", "## Background (not evidence about this dataset)", ""]
    L += [f"- **{n['id']}** ({n['source']}) — {n['title']}. {n['text']} [K:{n['id']}]" for n in notes]
    L += ["", "## Limits", ""] + [f"- {s}" for s in LIMITS]
    return L


def render_template(q, evidence, notes, digest):
    L = header(q, digest) + ["## Findings"] + findings(evidence)
    L += ["", "## Timeline", ""] + timeline_lines(evidence, q) + tail(notes)
    return "\n".join(L) + "\n"


def assemble(q, digest, body, notes):
    """LLM writes Findings + Timeline only; header, Background and Limits are always deterministic."""
    return "\n".join(header(q, digest)) + "\n" + body.strip() + "\n" + "\n".join(tail(notes)) + "\n"


def llm_rewrite(template_md, model="claude-sonnet-4-6"):
    """Optional readability pass. Output is untrusted until validate() passes."""
    from langchain_anthropic import ChatAnthropic
    prompt = ("Improve the wording of this investigation report for a security analyst. Rules: keep every "
              "section heading; keep one bullet per fact; keep every number, date, `backticked` string, "
              "[E#] and [K:id] citation exactly as written; do not add facts, numbers, technique ids or "
              "conclusions; do not remove the Limits section. Return only the markdown.\n\n" + template_md)
    out = ChatAnthropic(model=model, max_tokens=3000).invoke(prompt).content
    return out if isinstance(out, str) else "".join(b.get("text", "") for b in out)
