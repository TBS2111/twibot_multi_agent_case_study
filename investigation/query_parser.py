"""Stage 3: turn a free-text question into structured parameters.

Rule-based first (deterministic). An LLM parser is optional and its JSON is re-validated by
Query.from_dict, so a bad LLM answer can never inject unknown fields or out-of-range values.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

DATE = r"(\d{4}-\d{2}-\d{2})"


@dataclass
class Query:
    raw: str = ""
    community: Optional[int] = None        # explicit community id
    selector: str = "most_bot"             # most_bot | largest
    top_n: int = 1
    min_bot_ratio: Optional[float] = None  # 0..1, over labeled members
    min_size: int = 5
    since: Optional[str] = None            # YYYY-MM-DD, applied to timeline events
    until: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict, raw: str = ""):
        q = cls(raw=raw)
        if d.get("community") is not None:
            q.community = int(d["community"])
        if d.get("selector") in ("most_bot", "largest"):
            q.selector = d["selector"]
        q.top_n = max(1, min(20, int(d.get("top_n", 1))))
        if d.get("min_bot_ratio") is not None:
            r = float(d["min_bot_ratio"])
            q.min_bot_ratio = r / 100 if r > 1 else r
            if not 0 <= q.min_bot_ratio <= 1:
                raise ValueError("min_bot_ratio out of range")
        q.min_size = max(1, int(d.get("min_size", 5)))
        for k in ("since", "until"):
            v = d.get(k)
            if v is not None:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(v)):
                    raise ValueError(f"{k} must be YYYY-MM-DD")
                setattr(q, k, str(v))
        return q

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def parse_rules(text: str) -> Query:
    t = text.lower()
    d, notes = {}, []
    m = re.search(r"(?:community|cluster|group)\s*#?\s*(\d+)", t)
    if m:
        d["community"] = int(m.group(1))
    if re.search(r"\b(largest|biggest|major)\b", t):
        d["selector"] = "largest"
    m = re.search(r"\btop\s+(\d+)\b", t) or re.search(r"\b(\d+)\s+(?:largest|biggest|most|communities|clusters)", t)
    if m:
        d["top_n"] = int(m.group(1))
    m = re.search(r"bot\s*(?:ratio|share|rate)?\s*(?:above|over|at least|>=|>|greater than)\s*(\d+(?:\.\d+)?)\s*(%?)", t)
    if m:
        v = float(m.group(1))
        d["min_bot_ratio"] = v / 100 if (m.group(2) or v > 1) else v
    m = re.search(r"between\s+" + DATE + r"\s+and\s+" + DATE, t)
    if m:
        d["since"], d["until"] = m.groups()
    else:
        m = re.search(r"(?:since|after|from)\s+" + DATE, t)
        if m:
            d["since"] = m.group(1)
        m = re.search(r"(?:until|before|to)\s+" + DATE, t)
        if m:
            d["until"] = m.group(1)
        m = re.search(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", t)
        if m and "since" not in d:
            d["since"], d["until"] = f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    if d.get("selector") == "largest" and "min_bot_ratio" not in d and re.search(r"\bbot", t):
        d["min_bot_ratio"] = 0.5                  # "largest bot community" means bot-majority, not just largest
        notes.append("'bot' mentioned with 'largest': restricted to communities with bot ratio >= 0.5")
    if not d:
        notes.append("no filters recognised; defaulting to the community with the highest bot ratio")
    q = Query.from_dict(d, raw=text)
    q.notes = notes
    return q


def parse_llm(text: str, model="claude-sonnet-4-6") -> Query:
    """Optional convenience parser. Falls back to rules on ANY problem."""
    try:
        from langchain_anthropic import ChatAnthropic
        prompt = ("Convert the question into JSON with only these optional keys: community (int), "
                  "selector ('most_bot'|'largest'), top_n (int), min_bot_ratio (0-1), min_size (int), "
                  "since/until (YYYY-MM-DD). Output JSON only.\nQuestion: " + text)
        out = ChatAnthropic(model=model, max_tokens=200).invoke(prompt).content
        out = out if isinstance(out, str) else "".join(b.get("text", "") for b in out)
        d = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        return Query.from_dict(d, raw=text)
    except Exception:
        q = parse_rules(text)
        q.notes.append("LLM parser failed or unavailable; used rules")
        return q


def parse(text: str, use_llm: bool = False) -> Query:
    return parse_llm(text) if use_llm else parse_rules(text)
