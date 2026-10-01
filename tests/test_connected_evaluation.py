import importlib
import math
import pytest


def api():
    try:return importlib.import_module('integration.evaluate_system')
    except ModuleNotFoundError:pytest.fail('source-relative connected evaluation is not implemented')


def test_independent_labels_preserve_inverse_evidence_after_revision():
    p=api()
    records=[dict(kind='edge',child='x1',parent='x0',value=.25),
             dict(kind='root',child='x1',parent=None,value=.75),
             dict(kind='query',child='x0',parent=None,value=None),
             dict(kind='revision',child='x1',parent=None,value=1.25),
             dict(kind='query',child='x0',parent=None,value=None)]
    labels=p.source_labels(records)
    assert [x['source_index'] for x in labels]==[2,4]
    assert [x['answer'] for x in labels]==pytest.approx([.5,1.])


def test_ambiguous_and_inconsistent_source_is_not_silently_labeled():
    p=api()
    with pytest.raises(ValueError):p.source_labels([dict(kind='query',child='x0',parent=None,value=None)])
    with pytest.raises(ValueError):p.source_labels([
        dict(kind='root',child='x0',parent=None,value=0.),
        dict(kind='root',child='x1',parent=None,value=0.),
        dict(kind='edge',child='x1',parent='x0',value=1.),
        dict(kind='query',child='x0',parent=None,value=None)])


def test_missing_or_misparsed_query_remains_in_intended_denominator():
    p=api();case=p.make_cases(8,990001,'iid')[0]
    out={'outputs':[],'parsed':[]}
    scored=p.score_episode(case,out)
    assert len(scored['queries'])==3
    assert not any(x['source_correct'] for x in scored['queries'])
    assert all(x['status']=='MISSING_QUERY_OUTPUT' for x in scored['queries'])
    assert scored['final']['squared_error'] is None


def test_certificate_for_wrong_requested_entity_counts_as_source_error():
    p=api();case=p.make_cases(8,990001,'iid')[0];q=case['queries'][-1]
    out={'outputs':[dict(source_index=q['source_index'],prediction=q['answer'],
                        query='different_entity',status='VERIFIED_PARSED_CONSTRAINTS')], 'parsed':[]}
    scored=p.score_episode(case,out)['final']
    assert scored['accepted'] and not scored['source_correct'] and scored['accepted_source_failure']


def test_dataset_is_balanced_and_text_roundtrip_values_match_source():
    p=api();cases=p.make_cases(16,990002,'depth8')
    assert sum(c['affected'] for c in cases)==8
    assert sum(c['inverse'] for c in cases)==8
    assert all(c['depth']==8 for c in cases)
    for case in cases:
        assert len(case['records'])==14
        assert all(math.isfinite(q['answer']) for q in case['queries'])
        assert len(p.render_case(case,'seen'))==len(p.render_case(case,'heldout'))==14
        assert len({e['child'] for e in case['records']})<=10


def test_run_refuses_partial_outputs_and_source_drift(tmp_path):
    p=api();p.assert_fresh_results(tmp_path)
    (tmp_path/'execution.jsonl').write_text('partial')
    with pytest.raises(FileExistsError):p.assert_fresh_results(tmp_path)
    f=tmp_path/'frozen.py';f.write_text('original')
    hashes={str(f):p.sha(f)};p.check_sources(hashes)
    f.write_text('modified')
    with pytest.raises(ValueError):p.check_sources(hashes)


def test_summary_zero_coverage_has_unknown_risk_and_no_missing_zero_fill():
    p=api();case=p.make_cases(8,990001,'iid')[0]
    scored=p.score_episode(case,{'outputs':[],'parsed':[]})
    summary=p.summarize_scores([scored])
    assert summary['final']['coverage']==0.
    assert summary['final']['accepted_error_rate'] is None
    assert summary['final']['mse_emitted'] is None
    assert summary['final']['accuracy_all_intended']==0.


def test_certified_spurious_query_counts_as_source_alignment_failure():
    p=api();case=p.make_cases(8,990001,'iid')[0]
    result={'outputs':[dict(source_index=0,prediction=0.,query='x0',status='VERIFIED_PARSED_CONSTRAINTS')], 'parsed':[]}
    score=p.score_episode(case,result)
    assert score['spurious_accepted_outputs']==1
    summary=p.summarize_scores([score])
    assert summary['all_certificate_outputs']['count']==1
    assert summary['all_certificate_outputs']['source_failures']==1
    assert summary['all_certificate_outputs']['source_failure_rate']==1.
