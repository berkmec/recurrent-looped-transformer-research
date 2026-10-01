"""D1 examples from MuSiQue: question + supporting paragraphs + seeded distractors; gold = human-authored decomposition.

Gold never enters the context: paragraphs are stripped of is_supporting and identified only by a pid.
Negative controls are programmatic perturbations of human data and are labelled as such.
"""
import copy
import hashlib
import json
import re

import numpy as np


def _pid(row_id, idx):
    return 'p' + hashlib.sha256(f'{row_id}#{idx}'.encode()).hexdigest()[:10]


def _split_hop(sub_question):
    if '>>' in sub_question:
        subject, relation = sub_question.split('>>', 1)
        return subject.strip(), relation.strip()
    return None, sub_question.strip()


def build_examples(path, distractors=2, seed=0, limit=None, offset=0):
    rng = np.random.default_rng(seed)
    examples = []
    with open(path, encoding='utf8') as f:
        for n, line in enumerate(f):
            if n < offset:
                continue
            row = json.loads(line)
            supporting = [p for p in row['paragraphs'] if p.get('is_supporting')]
            others = [p for p in row['paragraphs'] if not p.get('is_supporting')]
            take = min(distractors, len(others))
            chosen = [others[i] for i in sorted(rng.choice(len(others), size=take, replace=False))] if take else []
            context = [dict(pid=_pid(row['id'], p['idx']), title=p.get('title', ''), text=p['paragraph_text']) for p in supporting + chosen]
            rng.shuffle(context)
            hops = []
            for step, hop in enumerate(row['question_decomposition'], start=1):
                subject, relation = _split_hop(hop['question'])
                ref = re.fullmatch(r'#(\d+)', subject or '')
                hops.append(dict(step=step, subject=None if ref else subject, depends_on=int(ref.group(1)) if ref else None,
                                 relation=relation, answer=hop['answer'],
                                 aliases=list(row.get('answer_aliases', [])) if step == len(row['question_decomposition']) else [],
                                 source=_pid(row['id'], hop['paragraph_support_idx'])))
            examples.append(dict(id=row['id'], group=row['id'], hops=len(hops), language='en', provenance='human_authored:MuSiQue',
                                 question=row['question'], context=context, gold=dict(answer=row['answer'], hops=hops)))
            if limit and len(examples) >= limit:
                break
    return examples


def _swap_word(text, a, b):
    pattern = re.compile(r'\b(' + re.escape(a) + '|' + re.escape(b) + r')\b', re.IGNORECASE)
    def sub(m):
        w = m.group(1)
        return b if w.lower() == a.lower() else a
    return pattern.sub(sub, text)


def negative_controls(examples, seed=0):
    rng = np.random.default_rng(seed)
    controls = []
    for e in examples:
        last = e['gold']['hops'][-1]
        # 1) missing source: drop the last hop's supporting paragraph
        c = copy.deepcopy(e); c['context'] = [x for x in c['context'] if x['pid'] != last['source']]
        c.update(id=e['id'] + '__missing_source', control='missing_source', provenance='programmatic_perturbation', base_id=e['id'],
                 expect_unresolved_hops=[last['step']])
        c['gold']['hops'][-1]['source'] = None
        controls.append(c)
        # 2) relation swap: ask for an unsupported relation in the question
        alt = 'child' if last['relation'].lower() != 'child' else 'spouse'
        if re.search(r'\b' + re.escape(last['relation']) + r'\b', e['question'], re.IGNORECASE):
            question = re.sub(r'\b' + re.escape(last['relation']) + r'\b', alt, e['question'], count=1, flags=re.IGNORECASE)
        else:
            question = e['question'].rstrip('?') + f' (specifically the {alt})?'
        c = copy.deepcopy(e); c.update(id=e['id'] + '__relation_swap', control='relation_swap', provenance='programmatic_perturbation', base_id=e['id'],
                                       question=question, expect_unresolved_hops=[last['step']])
        c['gold']['hops'][-1].update(relation=alt, answer=None, source=None)
        controls.append(c)
        # 3) subject/relation order swap: exchange the first and last relation words in the question
        first = e['gold']['hops'][0]
        if first['relation'].lower() != last['relation'].lower() and all(re.search(r'\b' + re.escape(r) + r'\b', e['question'], re.IGNORECASE) for r in (first['relation'], last['relation'])):
            question = _swap_word(e['question'], first['relation'], last['relation']); applicable = True
        else:
            question = e['question'].rstrip('?') + ' — asked in reverse order?'; applicable = False
        c = copy.deepcopy(e); c.update(id=e['id'] + '__subject_swap', control='subject_swap', provenance='programmatic_perturbation', base_id=e['id'],
                                       question=question, applicable=applicable, expect_unresolved_hops=[h['step'] for h in e['gold']['hops']])
        for h in c['gold']['hops']:
            h.update(answer=None, source=None)
        controls.append(c)
    return controls


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**22), b''):
            h.update(chunk)
    return h.hexdigest()
