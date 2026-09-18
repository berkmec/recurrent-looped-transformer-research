"""C2/C7/F2 statistical and reporting contracts: Holm step-down, one-sided zero-error bound, full-log summaries, full job manifests."""
import math


def holm(pvalues, alpha=0.05):
    """Holm step-down: sorted p_i rejected while p_i <= alpha / (m - i). Returns {key: rejected}. Arithmetic only; no claim of independence."""
    if not 0 < alpha < 1 or any(not 0. <= p <= 1. for p in pvalues.values()):
        raise ValueError('alpha in (0,1) and probabilities in [0,1] required')
    ordered = sorted(pvalues, key=pvalues.get)
    result = {k: False for k in ordered}
    for rank, key in enumerate(ordered):
        if pvalues[key] <= alpha / (len(ordered) - rank):
            result[key] = True
        else:
            break
    return result


def zero_error_upper_bound(n, confidence=0.95):
    """One-sided upper bound on the error rate given ZERO errors among n independent accepts: 1 - (1-confidence)^(1/n).
    Not valid for thresholds selected after the test, dependent outputs, nonzero errors or unobserved distributions."""
    if n < 1 or not 0 < confidence < 1:
        raise ValueError('n >= 1 and confidence in (0,1) required')
    return -math.expm1(math.log1p(-confidence) / n)


def summarize_records(records, expected_ids):
    """Consume ALL records; reject duplicates/unexpected ids; report missing ids, prediction histogram, coverage and accepted risk."""
    expected = set(expected_ids); seen = set(); counts = {}; accepted = wrong = 0
    for row in records:
        key = row['id']
        if key not in expected or key in seen:
            raise ValueError(f'duplicate or unexpected record id {key!r}')
        seen.add(key)
        label = str(row['prediction']); counts[label] = counts.get(label, 0) + 1
        if row.get('accepted', False):
            accepted += 1; wrong += int(not row['correct'])
    return dict(expected=len(expected), processed=len(seen), missing=sorted(expected - seen), prediction_counts=counts,
                coverage=accepted / len(expected) if expected else None, accepted_risk=wrong / accepted if accepted else None,
                status='COMPLETE' if seen == expected else 'INCOMPLETE')


def manifest_jobs(problem_ids, arms, seeds):
    """Full Cartesian product problem x arm x seed; empty or duplicated dimensions are refused so no cell can vanish silently."""
    if not problem_ids or not arms or not seeds:
        raise ValueError('empty manifest dimension')
    if len(set(problem_ids)) != len(problem_ids) or len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise ValueError('duplicate manifest dimension')
    return [dict(problem=p, arm=a, seed=s) for p in problem_ids for a in arms for s in seeds]
