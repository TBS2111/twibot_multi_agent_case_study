"""LangGraph workflow for the investigation agents (same style as graph/workflow.py).

    START -> query -> evidence -> knowledge -> report -> validator --(ok or fallback used)--> END
                                                  ^                 |
                                                  +---(issues)------+     revise loop, bounded

If langgraph is not installed, run_sequential() follows the same routing.
"""
import json
import logging
import operator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, List, TypedDict

from investigation.agents import AGENTS

log = logging.getLogger("investigation")


class InvState(TypedDict, total=False):
    question: str
    db: str
    llm: Any                    # object with .complete(system, user, max_tokens) or None
    max_revisions: int
    log: Annotated[List[str], operator.add]
    query: Any
    params: Any
    communities: list
    evidence: list
    digest: str
    notes: list
    kb_keywords: Any
    draft: str
    revisions: int
    used_fallback: bool
    issues: list
    ok: bool
    report_md: Any


def route_after_validator(st) -> str:
    return "end" if st.get("ok") or st.get("used_fallback") else "report"


def _wrap(agent):
    def node(st):
        out = agent.run(st)
        for line in out.get("log", []):
            log.info(line)
        return out
    return node


def build_graph():
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(InvState)
    for a in AGENTS:
        g.add_node(a.name, _wrap(a))
    g.add_edge(START, "query")
    g.add_edge("query", "evidence")
    g.add_edge("evidence", "knowledge")
    g.add_edge("knowledge", "report")
    g.add_edge("report", "validator")
    g.add_conditional_edges("validator", route_after_validator, {"report": "report", "end": END})
    return g.compile()


def run_sequential(state):
    agents = {a.name: a for a in AGENTS}
    node = "query"
    while node:
        out = _wrap(agents[node])(state)
        state["log"] = state.get("log", []) + out.pop("log", [])
        state.update(out)
        node = {"query": "evidence", "evidence": "knowledge", "knowledge": "report", "report": "validator",
                "validator": None if route_after_validator(state) == "end" else "report"}[node]
    return state


def write_outputs(st, report_dir):
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "report.md" if st["ok"] else "report.REJECTED.md"
    (out / name).write_text(st["draft"], encoding="utf-8")
    (out / "report.evidence.json").write_text(json.dumps(dict(
        query=asdict(st["query"]), params=asdict(st["params"]), digest=st["digest"],
        communities=st["communities"], evidence=[asdict(e) for e in st["evidence"]], notes=st["notes"],
        agent=dict(kb_keywords=st.get("kb_keywords"), revisions=st["revisions"],
                   template_fallback=st["used_fallback"], log=st["log"])), indent=2, sort_keys=True),
        encoding="utf-8")
    return out / name


def run_investigation(question, db, report_dir, llm=None, max_revisions=3):
    state = {"question": question, "db": str(db), "llm": llm, "max_revisions": max_revisions, "log": []}
    try:
        st = build_graph().invoke(state, {"recursion_limit": 50})
    except ImportError:
        log.warning("langgraph not installed - running agents sequentially")
        st = run_sequential(state)
    path = write_outputs(st, report_dir)
    return st, path
