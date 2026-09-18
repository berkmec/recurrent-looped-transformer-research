"""B1 audit: the patched reranker (base model + lm_head on the last hidden state) must reproduce the pre-patch
full-vocabulary-logits score per pair, and batching with left padding must not change scores.
Requires the pinned Qwen3-Reranker-0.6B on disk; otherwise the test is reported as MISSING_ASSET (skipped, visible)."""
import json
import os
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
QUERY = 'What is the capital of France?'
DOCS = ['Paris is the capital and most populous city of France.',
        'The Eiffel Tower was completed in 1889.',
        'Photosynthesis converts light energy into chemical energy in plants, algae and some bacteria; it takes place in chloroplasts.',
        'Berlin']


class RerankParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        pins = json.loads((ROOT / 'protocols' / 'G0-pins.json').read_text(encoding='utf8'))
        models = pins.get('models', {})
        cls.model_dir = next((v['local_dir'] for k, v in models.items() if 'Reranker' in k), 'E:/rlt-research/models/Qwen3-Reranker-0.6B')
        if not Path(cls.model_dir).exists():
            raise unittest.SkipTest('MISSING_ASSET: ' + cls.model_dir)
        if os.environ.get('RLT_SKIP_HEAVY', '') not in ('', '0'):
            raise unittest.SkipTest('HEAVY_DISABLED: RLT_SKIP_HEAVY is set (the 0.6B reranker would be loaded on CPU)')
        from semantic_memory.rerank import QwenReranker
        cls.rr = QwenReranker(cls.model_dir, device='cpu', max_length=256, dtype=torch.float32)

    def reference_scores(self, docs):
        """Pre-patch computation: full [batch, seq, vocab] logits, last position."""
        rr = self.rr
        pairs = [rr.format(QUERY, d) for d in docs]
        enc = rr.tokenizer(pairs, padding=False, truncation='longest_first', return_attention_mask=False, max_length=rr.max_length - len(rr.prefix) - len(rr.suffix))
        enc['input_ids'] = [rr.prefix + ids + rr.suffix for ids in enc['input_ids']]
        batch = rr.tokenizer.pad(enc, padding=True, return_tensors='pt')
        with torch.no_grad():
            logits = rr.model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, rr.no], logits[:, rr.yes]], dim=1).float()
        return torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()

    def test_patched_scores_match_full_logits_reference(self):
        patched = self.rr.score(QUERY, DOCS, batch_size=4)
        reference = self.reference_scores(DOCS)
        self.assertEqual(len(patched), len(DOCS))
        for a, b in zip(patched, reference):
            self.assertAlmostEqual(a, b, places=4)
        self.assertGreater(patched[0], patched[3])  # relevant document scores above an irrelevant one

    def test_batching_with_left_padding_does_not_change_scores(self):
        batched = self.rr.score(QUERY, DOCS, batch_size=4)
        single = [self.rr.score(QUERY, [d], batch_size=1)[0] for d in DOCS]
        for a, b in zip(batched, single):
            self.assertAlmostEqual(a, b, places=3)


if __name__ == '__main__':
    unittest.main()
