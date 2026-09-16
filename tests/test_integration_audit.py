"""Audit safeguards: independently derived cases, not learned-model assertions."""
import importlib
import numpy as np
import pytest


def module():
    try:
        return importlib.import_module('integration.audit_stats')
    except ModuleNotFoundError:
        pytest.fail('paired audit and protection gates not implemented')


def test_exact_constant_paired_effect_has_point_interval():
    audit = module()
    ref = np.array([[3.,4.,5.],[10.,11.,12.],[20.,21.,22.]])
    result = audit.paired_summary(ref, ref-2, bootstrap_reps=100, seed=19)
    assert result['mean_improvement'] == 2
    assert result['ci95_improvement'] == [2.,2.]
    assert result['examples'] == 3
    assert result['training_seeds'] == 3


def test_correlated_seed_predictions_not_counted_as_more_examples():
    audit = module()
    result = audit.paired_summary(np.ones((3,4)),np.zeros((3,4)),bootstrap_reps=80,seed=2)
    assert result['examples'] == 4
    assert result['model_example_predictions'] == 12
    assert len(result['per_seed_improvement']) == 3


def test_unaffected_final_error_blocks_misleading_revision_gain():
    audit = module()
    ref = np.ones((3,4))
    candidate = np.array([[.5,.5,1.2,1.2]]*3)
    gate = audit.revision_gate(ref,candidate,np.array([True,True,False,False]))
    assert gate['affected_relative_improvement'] == pytest.approx(.5)
    assert not gate['unaffected_final_protected']
    assert not gate['passed']


def test_revision_gate_requires_every_seed_to_improve():
    audit = module()
    ref=np.ones((3,4)); candidate=ref.copy()
    candidate[:,:2]=np.array([[.1],[.1],[1.1]])
    gate=audit.revision_gate(ref,candidate,np.array([True,True,False,False]))
    assert gate['affected_relative_improvement']>.1
    assert not gate['all_seeds_affected_improved']
    assert not gate['passed']


def test_valid_revision_gate_passes_both_final_error_checks():
    audit = module()
    ref=np.ones((3,4)); candidate=np.array([[.7,.7,.98,.98]]*3)
    gate=audit.revision_gate(ref,candidate,np.array([True,True,False,False]))
    assert gate['passed']
    assert gate['overall_relative_improvement']==pytest.approx(.16)


@pytest.mark.parametrize('bad',[np.array([[np.nan,1.]]),np.array([[np.inf,1.]]),np.array([1.,2.])])
def test_invalid_loss_evidence_is_rejected(bad):
    audit=module()
    with pytest.raises(ValueError):
        audit.paired_summary(bad,bad,bootstrap_reps=20)


def test_no_acceptances_do_not_become_zero_risk_evidence():
    audit=module()
    result=audit.coverage_risk(np.zeros(5,dtype=bool),np.array([0.,1.,0.,1.,0.]),tolerance=.1)
    assert result['coverage']==0
    assert result['accepted_error_rate'] is None
    assert result['accepted_error_rate_ci95'] is None


def test_acceptance_risk_counts_conditional_errors_and_coverage():
    audit=module()
    result=audit.coverage_risk(np.array([True,False,True,False]),np.array([.2,3.,.05,0.]),tolerance=.1)
    assert result['coverage']==.5
    assert result['accepted_error_rate']==.5
    assert result['accepted_count']==2
    lo,hi=result['accepted_error_rate_ci95']
    assert 0<lo<.5<hi<1
