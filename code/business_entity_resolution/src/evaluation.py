import numpy as np
from typing import Dict, Set

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    - True Singleton (0 true matches): 1.0 if pred is empty, 0.0 if any false match is predicted.
    - Non-Singleton: (1.25 * P * R) / (0.25 * P + R)
    """
    # 1. Singleton Case
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    # 2. Non-Singleton Case
    tp = len(true_matches.intersection(pred_matches))
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = 0.25  # F_0.5 -> beta^2 = 0.5^2 = 0.25
    f05 = ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)
    return float(f05)

def evaluate_predictions_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """
    Computes macro-averaged F_0.5 across all Source 1 entities in ground truth.
    """
    scores = [
        compute_entity_f05(true_matches, predictions.get(s1_id, set()))
        for s1_id, true_matches in ground_truth.items()
    ]
    return float(np.mean(scores)) if scores else 0.0

def verify_evaluator_against_worked_example() -> bool:
    """
    Sanity checks evaluator against PDF Page 6 worked example:
    - Predicts: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    - Ground Truth: S1-00001 -> [S2-00047, S3-00812]
    - Precision: 2/3, Recall: 2/2 = 1.0 -> Expected F_0.5 = 0.714
    """
    gt = {"S1-00001": {"S2-00047", "S3-00812"}}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    score = compute_entity_f05(gt["S1-00001"], pred["S1-00001"])
    passed = abs(score - 0.7142857) < 1e-4
    return passed
