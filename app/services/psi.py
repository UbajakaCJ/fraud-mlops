"""
psi.py — Population Stability Index utility
Used by both fraud_drift_monitor_dag.py and the test suite directly.
"""
import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

    Thresholds (standard):
      PSI < 0.10  → stable
      PSI 0.10–0.20 → minor shift, monitor
      PSI > 0.20  → significant shift, retrain
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    def bucket_pct(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=breakpoints)
        pct = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    e_pct = bucket_pct(expected)
    a_pct = bucket_pct(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
