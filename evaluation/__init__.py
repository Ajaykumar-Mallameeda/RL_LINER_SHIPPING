"""
P15 — Common Evaluation Layer for LSNDP.

Method-agnostic evaluation framework supporting RL, GA, MILP, and
reference solutions under a single objective and MCF evaluation.
"""

from .candidate import CandidateService, CandidateSolution
from .evaluator import CommonEvaluator, EvaluationResult
from .adapters import rl_result_to_candidate, reference_solution_to_candidate

__all__ = [
    "CandidateService",
    "CandidateSolution",
    "CommonEvaluator",
    "EvaluationResult",
    "rl_result_to_candidate",
    "reference_solution_to_candidate",
]
