"""Shared state passed between agents in the LangGraph workflow."""
import operator
from typing import Annotated, Any, Dict, List, TypedDict


class PipelineState(TypedDict, total=False):
    cfg: Any
    log: Annotated[List[str], operator.add]     # parallel-safe

    # supervisor
    users: List[dict]
    n_nodes: int
    labels_true: Any        # np.ndarray, -1 = unknown
    splits: Any             # np.ndarray[str]

    # semantic agent
    sem_emb: Any
    # behavioral agent
    url_mat: Any
    interest_mat: Any
    social_mat: Any
    link_edges: Any
    beh_feat: Any
    # trust agent
    trust: Any
    trust_feat: Any

    # GBT-CMA
    edges: Any              # (E,2) undirected, i<j
    edge_w: Any             # (E,) signed weights
    edge_sims: Any          # (E,4) sem/url/interest/social
    emb: Any                # (N, emb_dim) graph embeddings

    # SBCD / ICR
    initial_labels: Any
    refined_labels: Any
    icr_trace: List[dict]

    # classifier
    clf_metrics: Dict[str, Any]

    # evaluation
    metrics: Dict[str, Any]
