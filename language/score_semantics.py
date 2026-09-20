"""D1 scoring: relation/answer F1 against the human decomposition, source binding, accepted-wrong risk, coverage."""
import math
import re
import unicodedata


def normalize(text):
    if text is None:
        return ''
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'\b(the|a|an)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def answer_match(pred, gold, aliases=()):
    p = normalize(pred)
    if not p:
        return False
    return any(p == normalize(g) for g in [gold, *aliases] if g is not None)


def relation_match(pred, gold):
    return normalize(pred) == normalize(gold)


def score_example(pred, gold, expect_unresolved=()):
    """pred: dict(status, hops=[dict(step, relation, answer, source, depends_on, unresolved?)]); gold: dict(hops=[...])."""
    gold_hops = {h['step']: h for h in gold['hops']}
    pred_hops = {} if pred.get('status') == 'UNRESOLVED' else {h.get('step'): h for h in pred.get('hops', []) if isinstance(h, dict)}
    tp = fp = fn = 0; source_correct = 0; accepted = 0; accepted_wrong = 0; unresolved = 0; controls_ok = 0
    for step, g in gold_hops.items():
        p = pred_hops.get(step)
        if p is None or p.get('unresolved') or p.get('answer') in (None, '', 'UNRESOLVED'):
            unresolved += 1; fn += 1
            if step in expect_unresolved:
                controls_ok += 1
            continue
        accepted += 1
        ok = g.get('answer') is not None and relation_match(p.get('relation'), g['relation']) and answer_match(p.get('answer'), g['answer'], g.get('aliases', ()))
        if ok:
            tp += 1
            if p.get('source') == g.get('source'):
                source_correct += 1
        else:
            fp += 1; fn += 1; accepted_wrong += 1
    extra = [s for s in pred_hops if s not in gold_hops]
    fp += len(extra); accepted += len(extra); accepted_wrong += len(extra)
    return dict(relation_tp=tp, relation_fp=fp, relation_fn=fn, source_correct=source_correct, accepted=accepted, accepted_wrong=accepted_wrong,
                unresolved=unresolved, gold_hops=len(gold_hops), extra_hops=len(extra), controls_expected=len(expect_unresolved), controls_ok=controls_ok,
                status=pred.get('status'))


def risk_upper_95(accepted, wrong):
    """One-sided 95% upper bound on the accepted-wrong rate (exact zero-error formula, else Clopper-Pearson via beta quantile)."""
    if accepted == 0:
        return None
    if wrong == 0:
        return 1 - 0.05 ** (1 / accepted)
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, wrong + 1, accepted - wrong))
    except Exception:
        p = wrong / accepted
        return min(1., p + 1.645 * math.sqrt(p * (1 - p) / accepted))


def aggregate(scores):
    tp = sum(s['relation_tp'] for s in scores); fp = sum(s['relation_fp'] for s in scores); fn = sum(s['relation_fn'] for s in scores)
    accepted = sum(s['accepted'] for s in scores); wrong = sum(s['accepted_wrong'] for s in scores); gold = sum(s['gold_hops'] for s in scores)
    return dict(examples=len(scores), relation_tp=tp, relation_fp=fp, relation_fn=fn,
                relation_precision=tp / (tp + fp) if tp + fp else 0., relation_recall=tp / (tp + fn) if tp + fn else 0.,
                relation_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                source_binding_accuracy=sum(s['source_correct'] for s in scores) / tp if tp else None,
                accepted_hops=accepted, accepted_wrong=wrong, accepted_wrong_rate=wrong / accepted if accepted else None,
                risk_upper_95=risk_upper_95(accepted, wrong), coverage=accepted / gold if gold else 0.,
                unresolved_hops=sum(s['unresolved'] for s in scores), gold_hops=gold,
                controls_expected=sum(s['controls_expected'] for s in scores), controls_ok=sum(s['controls_ok'] for s in scores))
