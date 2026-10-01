"""Paired evidence summaries; no model fitting, policy selection or target repair.

Arrays use [training_seed, example]. Examples shared across models remain a
single sampling axis. Crossed bootstrap samples seed rows and example columns
independently, using the same sampled columns for every sampled seed. With only
three training seeds these intervals are descriptive, not strong population
evidence. They must accompany the individual seed effects.
"""
from __future__ import annotations
import math
import numpy as np


def _paired(reference, candidate):
    a, b = np.asarray(reference,dtype=np.float64), np.asarray(candidate,dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1:
        raise ValueError('equal nonempty [seed,example] arrays required')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('finite nonnegative losses required')
    return a,b


def _relative(reference, candidate):
    if reference==0:
        return 0. if candidate==0 else None
    return float(1-candidate/reference)


def paired_summary(reference, candidate, bootstrap_reps=2000, seed=170043):
    a,b=_paired(reference,candidate)
    if not isinstance(bootstrap_reps,int) or bootstrap_reps<1:
        raise ValueError('positive integer bootstrap_reps required')
    effect=a-b; seeds,examples=effect.shape
    rng=np.random.default_rng(seed)
    draws=np.empty(bootstrap_reps,dtype=np.float64)
    for index in range(bootstrap_reps):
        rows=rng.integers(seeds,size=seeds)
        columns=rng.integers(examples,size=examples)
        draws[index]=effect[np.ix_(rows,columns)].mean()
    return {'reference_mean':float(a.mean()),'candidate_mean':float(b.mean()),
            'mean_improvement':float(effect.mean()),
            'relative_improvement':_relative(float(a.mean()),float(b.mean())),
            'ci95_improvement':np.quantile(draws,[.025,.975]).tolist(),
            'per_seed_improvement':effect.mean(axis=1).tolist(),
            'training_seeds':seeds,'examples':examples,
            'model_example_predictions':seeds*examples,
            'bootstrap_reps':bootstrap_reps,'bootstrap_seed':seed,
            'uncertainty_method':'paired crossed seed/example percentile bootstrap',
            'limitation':'Few training seeds; interval is descriptive and does not establish broad statistical reliability.'}


def revision_gate(reference_final_loss,candidate_final_loss,affected,
                  minimum_affected_gain=.1,unaffected_max_ratio=1.05):
    a,b=_paired(reference_final_loss,candidate_final_loss)
    mask=np.asarray(affected)
    if mask.dtype!=np.bool_ or mask.ndim!=1 or len(mask)!=a.shape[1] or not mask.any() or mask.all():
        raise ValueError('boolean affected mask must contain both strata')
    if not 0<=minimum_affected_gain<1 or not np.isfinite(unaffected_max_ratio) or unaffected_max_ratio<1:
        raise ValueError('invalid predeclared gain/protection thresholds')
    ar,ac=float(a[:,mask].mean()),float(b[:,mask].mean())
    ur,uc=float(a[:,~mask].mean()),float(b[:,~mask].mean())
    relative=_relative(ar,ac)
    seed_improvements=(a[:,mask]-b[:,mask]).mean(axis=1)
    all_seeds=bool((seed_improvements>0).all())
    unaffected_protected=bool(uc<=unaffected_max_ratio*ur)
    overall_improved=bool(b.mean()<a.mean())
    passed=bool(relative is not None and relative>=minimum_affected_gain
                and all_seeds and unaffected_protected and overall_improved)
    return {'passed':passed,'affected_reference_final_loss':ar,'affected_candidate_final_loss':ac,
            'affected_relative_improvement':relative,'all_seeds_affected_improved':all_seeds,
            'per_seed_affected_improvement':seed_improvements.tolist(),
            'unaffected_reference_final_loss':ur,'unaffected_candidate_final_loss':uc,
            'unaffected_final_protected':unaffected_protected,
            'overall_relative_improvement':_relative(float(a.mean()),float(b.mean())),
            'overall_improved':overall_improved,'minimum_affected_gain':minimum_affected_gain,
            'unaffected_max_ratio':unaffected_max_ratio,
            'note':'Final-answer protection is distinct from low revision drift. This gate is a predeclared pilot criterion, not a significance test.'}


def _wilson(errors,count):
    if count==0:
        return None
    z=1.959963984540054; p=errors/count; den=1+z*z/count
    center=(p+z*z/(2*count))/den
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [max(0.,center-radius),min(1.,center+radius)]


def coverage_risk(accepted,absolute_error,tolerance=.1):
    mask=np.asarray(accepted); error=np.asarray(absolute_error,dtype=np.float64)
    if (mask.dtype!=np.bool_ or mask.ndim!=1 or error.shape!=mask.shape or len(mask)==0
            or not np.isfinite(error).all() or (error<0).any() or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('finite nonnegative error and nonempty boolean acceptance arrays required')
    count=int(mask.sum()); failures=int((error[mask]>tolerance).sum())
    return {'examples':len(mask),'accepted_count':count,'coverage':float(mask.mean()),
            'accepted_failures':failures,'accepted_error_rate':failures/count if count else None,
            'accepted_error_rate_ci95':_wilson(failures,count),'tolerance':tolerance,
            'interval_assumption':'Wilson interval for independent examples within one fixed model; do not flatten training seeds.'}
