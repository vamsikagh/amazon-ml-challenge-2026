import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.calibration import CalibratedClassifierCV
from .evaluation import compute_entity_f05, evaluate_predictions_macro_f05

class EntityResolutionClassifier:
    """
    Precision-Heavy Calibrated GBDT Classifier designed to hit >0.988 Leaderboard macro F_0.5.
    """
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05):
        if HAS_LIGHTGBM:
            base_model = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                min_child_samples=5,
                verbosity=-1,
                verbose=-1
            )
        else:
            base_model = HistGradientBoostingClassifier(
                max_iter=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            
        self.model = CalibratedClassifierCV(estimator=base_model, cv=3, method='isotonic')
        self.best_threshold = 0.50

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weights: np.ndarray = None):
        """Fits calibrated gradient boosting ensemble."""
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated match probability scores."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold_f05(
        self, 
        val_df: pd.DataFrame, 
        val_ground_truth: Dict[str, Set[str]]
    ) -> Tuple[float, float]:
        """
        Fine-grained sweep over calibrated probabilities to maximize per-entity macro F_0.5.
        """
        feature_cols = [c for c in val_df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'prob']]
        probs = self.predict_proba(val_df[feature_cols])
        val_df['prob'] = probs
        
        best_tau = 0.50
        best_score = -1.0
        
        # Fine-grained step size of 0.005 for high precision placement
        for tau in np.arange(0.30, 0.90, 0.005):
            preds = {}
            matched = val_df[val_df['prob'] >= tau]
            for row in matched.itertuples():
                s1_id = row.source1_entity_id
                cand_id = row.candidate_entity_id
                if s1_id not in preds:
                    preds[s1_id] = set()
                preds[s1_id].add(cand_id)
                
            macro_f05 = evaluate_predictions_macro_f05(val_ground_truth, preds)
            if macro_f05 > best_score:
                best_score = macro_f05
                best_tau = tau
                
        self.best_threshold = float(best_tau)
        return float(best_score), float(best_tau)
