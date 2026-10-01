"""B3: LM bridge extractor for multi-hop retrieval (B2 follow-up).

B2 measured that the cheap capitalised-term extractor produces generic bridges ("American", "United") and that the
resulting candidate union has a ceiling of 0.488 — so no reranker can reach the target. Here the bridge is named by the
LM instead: given the question and the first-hop passages, it must name the entities that have to be searched next.

Design notes:
- one LM call per UNIQUE question, batched and cached (`precompute`), so the tested `multihop()` path stays unchanged:
  the returned table is used as a plain `extractor(question, texts, n)`;
- parsing is deliberately strict (short entity names only, nothing already in the question, no prose, deduped);
- bridge quality is measured on its own (`bridge_hits`): does a bridge term actually occur in a gold passage that the
  first hop missed? That is the diagnostic B2 lacked, and it is reported whether or not retrieval improves.
"""
import re

PROMPT = """You are the bridge step of a multi-hop search system.
Given a question and the passages found for its FIRST step, name the entities that must be searched NEXT to answer it.

Rules:
- output 1 to {n} short entity names, one per line
- no numbering, no quotes, no explanation, no full sentences
- only names that appear in the passages, and not words already in the question
- if no further search is needed, output exactly NONE

Question: {question}

Passages:
{passages}

Next entities:"""

MAX_WORDS = 6
_DECOR = re.compile(r'^\s*(?:[-*•]|\(?\d+[.)])\s*')
_TRIM = re.compile(r'^["\'“‘\s]+|["\'”’\s.;,:]+$')


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', (text or '').lower())).strip()


def parse_bridges(raw, question, n=3):
    """Strict parse of the model's answer into at most n short entity names."""
    qwords = set(normalize(question).split())
    out = []
    for line in (raw or '').splitlines():
        term = _TRIM.sub('', _DECOR.sub('', line))
        if not term or term.strip().upper() == 'NONE':
            continue
        words = term.split()
        if len(words) > MAX_WORDS:
            continue
        key = normalize(term)
        if not key or key in {normalize(t) for t in out}:
            continue
        if set(key.split()) <= qwords:          # nothing that is entirely question words
            continue
        out.append(term)
        if len(out) >= n:
            break
    return out


class BridgeTable:
    """question -> bridge terms, usable directly as a multihop extractor."""

    def __init__(self, table, raw=None):
        self.table = dict(table); self.raw = dict(raw or {})

    def as_extractor(self):
        def extractor(question, texts, n=3):
            return list(self.table.get(question, []))[:n]
        return extractor

    def stats(self):
        sizes = [len(v) for v in self.table.values()]
        return dict(questions=len(self.table), mean_bridges=sum(sizes) / len(sizes) if sizes else 0.,
                    empty=sum(1 for s in sizes if s == 0))


def precompute(queries, passages_for, generator, batch_size=8, max_new_tokens=48, n=3):
    """One batched LM call per unique question; `passages_for(query)` supplies that query's first-hop passages."""
    unique = {}
    for q in queries:
        unique.setdefault(q['question'], q)
    items = list(unique.items())
    table, raw = {}, {}
    for at in range(0, len(items), batch_size):
        chunk = items[at:at + batch_size]
        prompts = [PROMPT.format(n=n, question=question, passages='\n'.join(f'- {t}' for t in passages_for(q)))
                   for question, q in chunk]
        outputs = generator.generate(prompts, max_new_tokens=max_new_tokens)
        for (question, _), text in zip(chunk, outputs):
            raw[question] = text
            table[question] = parse_bridges(text, question, n)
    return BridgeTable(table, raw)


def bridge_hits(bridges, missed_gold_texts):
    """Does a bridge term occur in a gold passage the first hop missed? Reported independently of retrieval gains."""
    if not missed_gold_texts:
        return dict(status='NO_MISSED_GOLD', any_hit=False, hits=0, bridges=len(bridges))
    joined = normalize(' \n '.join(missed_gold_texts))
    hits = sum(1 for b in bridges if normalize(b) and normalize(b) in joined)
    return dict(status='MEASURED', any_hit=hits > 0, hits=hits, bridges=len(bridges))
