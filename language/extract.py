"""D1 extraction arms: Phase 1 finite parser floor, LM zero-shot, LM with the heading-record schema.

Prompts carry only the question, the pid-labelled context paragraphs and the question id; never gold.
Output is parsed as JSON; anything unparsable or unsupported becomes UNRESOLVED (kept verbatim).
"""
import json
import re

SYSTEM = ('You extract the reasoning chain needed to answer a question from the given paragraphs. '
          'Use only the paragraphs. If a step cannot be supported by a paragraph, mark it "UNRESOLVED". '
          'Answer with JSON only.')

ZERO_SHOT = ('{system}\n\nParagraphs:\n{context}\n\nQuestion: {question}\nQUESTION_ID: {qid}\n\n'
             'Return JSON: {{"status": "OK" or "UNRESOLVED", "hops": [{{"step": 1, "subject": "...", "relation": "...", "answer": "...", '
             '"source": "<pid>", "depends_on": null or step number}}]}}. One hop per reasoning step; "answer" is the entity found for that step; '
             'set "answer": "UNRESOLVED" and "source": "UNRESOLVED" when no paragraph supports the step.')

SCHEMA = ('{system}\n\nParagraphs:\n{context}\n\nQuestion: {question}\nQUESTION_ID: {qid}\n\n'
          'Work in three parts. (1) Headings: list the entities the question is about, each bound to the pid where it appears. '
          '(2) Claims: for each reasoning step give a typed relation (e.g. performer, spouse, country, date of birth), the entity it applies to, and the answer entity. '
          '(3) Source binding: cite the pid that states the claim; if none states it, the step is "UNRESOLVED" — never guess. '
          'Later steps may refer to an earlier step with "depends_on". Return JSON only: {{"status": "OK" or "UNRESOLVED", '
          '"headings": [{{"name": "...", "pid": "..."}}], "hops": [{{"step": 1, "subject": "...", "relation": "...", "answer": "...", '
          '"source": "<pid>", "depends_on": null or step number}}]}}.')


def render_context(context):
    return '\n'.join(f"[{c['pid']}] {c['title']}: {c['text']}" for c in context)


def build_prompt(example, arm):
    template = SCHEMA if arm == 'lm_schema' else ZERO_SHOT
    return template.format(system=SYSTEM, context=render_context(example['context']), question=example['question'], qid=example['id'])


def parse_output(text, valid_pids):
    raw = text if isinstance(text, str) else ''
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return dict(status='UNRESOLVED', hops=[], raw=raw[:2000], parse='NO_JSON')
    try:
        obj = json.loads(raw[start:end + 1])
    except ValueError:
        return dict(status='UNRESOLVED', hops=[], raw=raw[:2000], parse='INVALID_JSON')
    hops = []
    for i, h in enumerate(obj.get('hops', []) if isinstance(obj, dict) else [], start=1):
        if not isinstance(h, dict):
            continue
        step = h.get('step') if isinstance(h.get('step'), int) else i
        source = h.get('source'); answer = h.get('answer')
        unresolved = answer in (None, '', 'UNRESOLVED') or source not in valid_pids
        hops.append(dict(step=step, subject=h.get('subject'), relation=h.get('relation'), answer=answer,
                         source=source if source in valid_pids else 'UNRESOLVED', depends_on=h.get('depends_on'), unresolved=bool(unresolved)))
    status = 'UNRESOLVED' if (isinstance(obj, dict) and obj.get('status') == 'UNRESOLVED') or not hops else 'OK'
    return dict(status=status, hops=hops, headings=obj.get('headings') if isinstance(obj, dict) else None, raw=raw[:2000], parse='OK')


def finite_parser(example):
    """Phase 1 affine-constraint parser as the regression floor: it does not model these questions."""
    try:
        from memory.api import parse_text
        events = parse_text(example['question'])
        return dict(status='UNRESOLVED', hops=[], raw=json.dumps(events, ensure_ascii=False)[:2000], parse='FINITE_PARSER_NO_CHAIN')
    except Exception as e:  # any failure is an abstention, recorded
        return dict(status='UNRESOLVED', hops=[], raw=repr(e)[:500], parse='FINITE_PARSER_ERROR')


def extract(examples, generator, arm='lm_schema', batch_size=4, max_new_tokens=384):
    if arm == 'finite_parser':
        return [finite_parser(e) for e in examples]
    if arm not in ('lm_zero', 'lm_schema'):
        raise ValueError('unknown arm')
    prompts = [build_prompt(e, arm) for e in examples]
    outputs = []
    for at in range(0, len(prompts), batch_size):
        outputs.extend(generator.generate(prompts[at:at + batch_size], max_new_tokens=max_new_tokens))
    return [dict(parse_output(o, {c['pid'] for c in e['context']}), prompt_chars=len(p)) for o, e, p in zip(outputs, examples, prompts)]
