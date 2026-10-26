"""Stage 5: validate the report against the evidence. Pure deterministic code, no model calls.

A report passes only if every claim traces to evidence. This gate applies equally to the template
report and to any LLM-rewritten report. Checks:

  1. digest      header digest == digest recomputed from the evidence records
  2. citations   every Findings/Timeline bullet cites existing [E#]; Background never cited as evidence
  3. numbers     every number in a bullet appears (within rounding) in the cited evidence
  4. dates       every date/timestamp in a bullet appears in the cited evidence
  5. strings     every `backticked` string appears in the cited evidence
  6. communities every "community N" mention is a community the cited evidence is about
  7. technique   every ATT&CK id (Txxxx[.xxx]) is one that retrieval actually returned
  8. background background bullets cite retrieved notes and keep the note title
  9. chronology  Timeline bullets are in non-decreasing time order
 10. tone        no overclaiming words; required sections present; every evidence community has a heading
"""
import re
from dataclasses import dataclass
from typing import List

from investigation.report import EMPTY_TIMELINE
from investigation.timeline import digest as evidence_digest

CITE_E = re.compile(r"\[E(\d+)\]")
CITE_K = re.compile(r"\[K:([A-Za-z0-9._-]+)\]")
ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?")
TICK = re.compile(r"`([^`]+)`")
TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
COMM = re.compile(r"\b[Cc]ommunity\s+(\d+)\b")
NUM = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(%?)")
OVERCLAIM = re.compile(r"\bproves?\b|\bproven\b|\bdefinitely\b|\bcertainly\b|\bconfirmed bot(?:net)?\b|"
                       r"\bis a botnet\b|\battributed to\b", re.I)
RISKY = re.compile(r"https?://|www\.|\]\(|<[A-Za-z/!]", re.I)
DIGEST = re.compile(r"\*\*Evidence digest:\*\*\s*`([0-9a-f]+)`")
SECTIONS = {"findings": "findings", "timeline": "timeline", "background": "background", "limits": "limits"}


@dataclass
class Issue:
    line: int
    code: str
    msg: str

    def __str__(self):
        return f"line {self.line}: [{self.code}] {self.msg}"


def _flatten(x, nums, strs):
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        nums.add(float(x))
    elif isinstance(x, str):
        strs.add(x)
    elif isinstance(x, dict):
        for v in x.values():
            _flatten(v, nums, strs)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _flatten(v, nums, strs)


def _num_ok(tok, is_pct, nums):
    v = float(tok.replace(",", ""))
    if is_pct:
        return any(abs(n * 100 - v) <= 0.51 or abs(n - v) <= 0.51 for n in nums)
    return any(abs(n - v) <= 0.0051 for n in nums)


def _norm_ts(s):
    return s if "T" in s else s + "T00:00:00"


def validate(report: str, evidence, notes, claimed_digest=None) -> List[Issue]:
    issues: List[Issue] = []
    ev = {e.id: e for e in evidence}
    note_ids = {n["id"]: n for n in notes}
    comms = {e.community for e in evidence}

    m = DIGEST.search(report)
    actual = evidence_digest(evidence)
    if not m:
        issues.append(Issue(0, "digest", "evidence digest header missing"))
    elif m.group(1) != actual:
        issues.append(Issue(0, "digest", f"header digest {m.group(1)} != recomputed {actual}"))
    if claimed_digest and claimed_digest != actual:
        issues.append(Issue(0, "digest", "stored evidence does not match its recorded digest"))

    section, seen_sections, headed, last_ts = None, set(), set(), None
    for no, raw in enumerate(report.splitlines(), 1):
        line = raw.strip()
        if line.startswith("## "):
            name = line[3:].lower().split(" ")[0]
            section = SECTIONS.get(name, name)
            seen_sections.add(section)
            continue
        if not line:
            continue
        for t in TID.findall(line):
            if t not in note_ids:
                issues.append(Issue(no, "technique", f"{t} was not returned by retrieval"))
        for c in COMM.findall(line):
            if int(c) not in comms:
                issues.append(Issue(no, "community", f"community {c} is not in the evidence"))
        if line.startswith("### "):
            mm = COMM.search(line)
            if mm:
                headed.add(int(mm.group(1)))
            continue

        if section in ("findings", "timeline"):
            if line == EMPTY_TIMELINE:
                continue
            if not line.startswith("- "):
                issues.append(Issue(no, "format", "only cited bullets are allowed in this section"))
                continue
            if OVERCLAIM.search(line):
                issues.append(Issue(no, "tone", "overclaiming wording"))
            if CITE_K.search(line):
                issues.append(Issue(no, "citation", "background note cited as evidence"))
            cites = CITE_E.findall(line)
            if not cites:
                issues.append(Issue(no, "citation", "bullet has no evidence citation"))
                continue
            cited = []
            for c in cites:
                if f"E{c}" not in ev:
                    issues.append(Issue(no, "citation", f"E{c} does not exist"))
                else:
                    cited.append(ev[f"E{c}"])
            if not cited:
                continue
            nums, strs = set(), set()
            for e in cited:
                _flatten(e.facts, nums, strs)
            text = CITE_E.sub("", line[2:])
            for q in TICK.findall(text):
                if not any(q in s for s in strs):
                    issues.append(Issue(no, "string", f"`{q}` not found in cited evidence"))
            text = TICK.sub(" ", text)
            if RISKY.search(text):
                issues.append(Issue(no, "format", "links/markup are only allowed inside backticks"))
            if len(line) > 600:
                issues.append(Issue(no, "format", "bullet too long"))
            for d in ISO.findall(text):
                if not any(d in s for s in strs):
                    issues.append(Issue(no, "date", f"{d} not found in cited evidence"))
            text = ISO.sub(" ", text)
            allowed = {e.community for e in cited}
            for c in COMM.findall(text):
                if int(c) not in allowed:
                    issues.append(Issue(no, "community", f"community {c} is not what the cited evidence is about"))
            text = COMM.sub(" ", text)
            for tok, pc in NUM.findall(text):
                if not _num_ok(tok, bool(pc), nums):
                    issues.append(Issue(no, "number", f"{tok}{pc} not supported by cited evidence"))
            if section == "timeline":
                lead = ISO.match(line[2:].strip())
                if lead:
                    ts = _norm_ts(lead.group(0))
                    if last_ts and ts < last_ts:
                        issues.append(Issue(no, "chronology", "timeline is out of order"))
                    last_ts = ts
                else:
                    issues.append(Issue(no, "format", "timeline bullet must start with a date"))

        elif section == "background":
            if not line.startswith("- "):
                issues.append(Issue(no, "format", "only note bullets are allowed in Background"))
                continue
            if CITE_E.search(line):
                issues.append(Issue(no, "background", "background must not cite evidence ids"))
            ks = CITE_K.findall(line)
            if not ks:
                issues.append(Issue(no, "background", "background bullet cites no note"))
            for k in ks:
                if k not in note_ids:
                    issues.append(Issue(no, "background", f"note {k} was not retrieved"))
                elif note_ids[k]["title"] not in line:
                    issues.append(Issue(no, "background", f"note {k} title altered"))

    for need in ("findings", "timeline", "background", "limits"):
        if need not in seen_sections:
            issues.append(Issue(0, "format", f"missing section: {need}"))
    for c in sorted(comms - headed):
        issues.append(Issue(0, "completeness", f"no heading for community {c}"))
    return issues
