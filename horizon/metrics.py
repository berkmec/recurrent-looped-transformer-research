"""U1 metrics: effective horizon, flips, three-seed summaries."""
import math
import statistics

import numpy as np

T_975 = {1: 12.706204736, 2: 4.302652729, 3: 3.182446305}


def effective_horizon(accuracy_by_length, threshold):
    """Largest tested length such that every tested length up to it reaches the threshold; 0 if the first fails."""
    horizon = 0
    for length in sorted(accuracy_by_length):
        if accuracy_by_length[length] >= threshold:
            horizon = length
        else:
            break
    return horizon


def flip_counts(base_pred, new_pred, target):
    was = base_pred == target
    now = new_pred == target
    return dict(correct_to_wrong=int(np.sum(was & ~now)), wrong_to_correct=int(np.sum(~was & now)),
                prediction_changes=int(np.sum(base_pred != new_pred)), n=int(len(target)))


def seed_summary(values):
    n = len(values)
    if n < 2:
        return dict(values=values, mean=values[0] if values else None, t95=None, seeds=n)
    mean = statistics.mean(values)
    half = T_975[n - 1] * statistics.stdev(values) / math.sqrt(n)
    return dict(values=values, mean=mean, min=min(values), max=max(values), t95=[mean - half, mean + half], seeds=n, df=n - 1)
