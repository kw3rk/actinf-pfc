# Graph controller (Stage 3 design)

Generalize the linear episode to a graph search. Status: design + skeleton
(`pfc/graph.py`); not yet wired to the runner.

## Formalism

- **Node** = one LLM artifact: a candidate answer, an extraction, a critique.
  Each node carries a private belief b(correct) updated from cheap signals:
  logit entropy at creation, agreement with sibling nodes (same parent,
  independent generation), verdicts targeted at it.
- **Edge operator** = a prompt template applied to a node (or node pair):
  solve-from-extraction, redo-under-pressure, verify, extract, merge.
  Operators carry the *persistent* learned model: per-(operator, context)
  transition counts (does applying this operator to a doubted node yield a
  better node?) and cost counts. This is the B-matrix generalized to a graph
  — "what seems to work," measured.
- **Search** = best-first over (node, operator) pairs, priority = expected
  free energy: expected utility improvement of the best submittable node
  + information gain about node beliefs + Dirichlet novelty of the operator,
  minus λ-priced expected cost.
- **Termination** = no (node, operator) pair has positive EFE → submit the
  node with the highest b(correct). Inquiry ends when nothing more is worth
  asking — a principled stop, not a step cap.

## Why this beats the linear controller

- **Backtracking**: the linear episode can only rework the latest candidate;
  the graph can branch from any node — including returning to a clean
  extraction after a poisoned solve lineage.
- **Calibrated value function**: Tree/Graph-of-Thoughts searches with LLM
  self-evaluation as the value function (measured on our model: AUC 0.734).
  Node beliefs here come from calibrated signals (entropy AUC 0.876,
  agreement likelihoods from counts).
- **Persistent vs transient split**: calibration tables stay small and grow
  across episodes; the node graph is per-episode and discarded. Learning
  never scales with graph size.

## Open questions

- Sibling agreement bookkeeping: agreement is pairwise; belief updates must
  avoid double-counting correlated evidence from the same parent.
- Operator context featurization: start with (node type, entropy bucket,
  depth); resist expansion until counts justify it.
- Mock validation environment: tasks with multi-step structure where an
  intermediate artifact can be silently poisoned — the case where
  backtracking provably beats linear rework.
