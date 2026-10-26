"""LangGraph workflow:  Supervisor -> [Semantic | Behavioral | Trust] -> GBT-CMA -> SBCD -> ICR -> Classifier -> Evaluation"""
import logging
import time

import numpy as np

from agents.behavioral_agent import BehavioralAgent
from agents.classifier_agent import ClassifierAgent
from agents.evaluation_agent import EvaluationAgent
from agents.gbt_cma_agent import GBTCMAAgent
from agents.icr_agent import ICRAgent
from agents.sbcd_agent import SBCDAgent
from agents.semantic_agent import SemanticAgent
from agents.trust_agent import TrustAgent
from graph.state import PipelineState
from preprocessing.dataset import load_twibot

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("workflow")


class SupervisorAgent:
    """Loads data, defines the plan; downstream agents are dispatched by the graph edges."""
    name = "supervisor"

    def run(self, state):
        cfg = state["cfg"]
        users = load_twibot(cfg)
        y = np.array([u["label"] for u in users])
        sp_ = np.array([u["split"] for u in users])
        plan = "semantic||behavioral||trust -> gbt_cma -> sbcd -> icr -> classifier -> evaluation"
        return {"users": users, "n_nodes": len(users), "labels_true": y, "splits": sp_,
                "log": [f"[supervisor] {len(users)} users "
                        f"(bots={int((y == 1).sum())}, humans={int((y == 0).sum())}, unlabeled={int((y < 0).sum())})",
                        f"[supervisor] plan: {plan}"]}


def _wrap(agent):
    def node(state):
        t = time.time()
        out = agent.run(state)
        dt = time.time() - t
        for line in out.get("log", []):
            log.info(line)
        out["log"] = out.get("log", []) + [f"[{agent.name}] done in {dt:.1f}s"]
        log.info("[%s] done in %.1fs", agent.name, dt)
        return out
    return node


AGENTS = [SupervisorAgent(), SemanticAgent(), BehavioralAgent(), TrustAgent(), GBTCMAAgent(),
          SBCDAgent(), ICRAgent(), ClassifierAgent(), EvaluationAgent()]


def build_graph():
    from langgraph.graph import END, START, StateGraph
    try:
        from langchain_core.runnables import RunnableLambda
        mk = lambda a: RunnableLambda(_wrap(a), name=a.name)       # agents as LangChain Runnables
    except ImportError:
        mk = _wrap
    g = StateGraph(PipelineState)
    for a in AGENTS:
        g.add_node(a.name, mk(a))
    g.add_edge(START, "supervisor")
    for n in ("semantic", "behavioral", "trust"):        # parallel fan-out
        g.add_edge("supervisor", n)
    g.add_edge(["semantic", "behavioral", "trust"], "gbt_cma")   # join
    g.add_edge("gbt_cma", "sbcd")
    g.add_edge("sbcd", "icr")
    g.add_edge("icr", "classifier")
    g.add_edge("classifier", "evaluation")
    g.add_edge("evaluation", END)
    return g.compile()


def run_sequential(state):
    """Fallback when langgraph is not installed."""
    for a in AGENTS:
        out = _wrap(a)(state)
        out_log = out.pop("log")
        state.update(out)
        state["log"] = state.get("log", []) + out_log
    return state


def run_pipeline(cfg):
    state = {"cfg": cfg, "log": []}
    try:
        return build_graph().invoke(state)
    except ImportError:
        log.warning("langgraph not installed - running agents sequentially")
        return run_sequential(state)


def llm_report(state):
    """Optional: LangChain LLM turns metrics into a short narrative (cfg.use_llm_supervisor)."""
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=600)
    prompt = ("Summarise these social-botnet community detection results for a researcher "
              "in <=150 words, noting which stage helped most:\n" + str(state["metrics"]))
    return llm.invoke(prompt).content
