"""Stage 3 skeleton: the node-graph controller (see docs/graph-controller.md).

Data structures and the search-loop shape. Not yet wired to the episode
runner — the linear controller (engine.py) remains the production path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# node types
N_EXTRACTION, N_CANDIDATE, N_CRITIQUE = "extraction", "candidate", "critique"


@dataclass
class Node:
    nid: int
    kind: str                      # extraction | candidate | critique
    content: str                   # the artifact text (answer, extraction...)
    answer: float | None           # parsed answer if kind == candidate
    parent: int | None
    entropy: float | None          # mean logit entropy at creation
    b_correct: float = 0.5         # calibrated private belief
    tokens: int = 0


@dataclass
class EpisodeGraph:
    nodes: list[Node] = field(default_factory=list)

    def add(self, **kw) -> Node:
        node = Node(nid=len(self.nodes), **kw)
        self.nodes.append(node)
        return node

    def candidates(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == N_CANDIDATE]

    def best(self) -> Node | None:
        cands = self.candidates()
        return max(cands, key=lambda n: n.b_correct) if cands else None

    def siblings(self, node: Node) -> list[Node]:
        return [n for n in self.nodes
                if n.parent == node.parent and n.nid != node.nid
                and n.kind == node.kind]


class GraphController:
    """Best-first (node, operator) search with EFE priority.

    Operators, their learned transition/cost counts, and the belief-update
    rules are the persistent model; the graph is per-episode.
    """

    def __init__(self, operators: dict, lam: float = 0.0002,
                 rng: np.random.Generator | None = None):
        self.operators = operators      # name -> callable(graph, node) -> Node
        self.lam = lam
        self.rng = rng or np.random.default_rng(0)
        # TODO(stage3): Dirichlet counts per (operator, context bucket)

    def node_belief(self, graph: EpisodeGraph, node: Node) -> float:
        """Calibrated b(correct) from entropy + sibling agreement.
        TODO(stage3): replace heuristic with learned likelihood counts;
        handle correlated-evidence double counting for shared parents."""
        raise NotImplementedError

    def efe(self, graph: EpisodeGraph, node: Node, op: str) -> float:
        """Expected free energy of applying op to node.
        TODO(stage3): expected improvement of best submittable node
        + info gain + operator novelty - lam * expected cost."""
        raise NotImplementedError

    def run_episode(self, task, agent, max_nodes: int = 12):
        """Search until no (node, operator) pair has positive EFE.
        TODO(stage3): implement; return best candidate + trace for learning."""
        raise NotImplementedError
