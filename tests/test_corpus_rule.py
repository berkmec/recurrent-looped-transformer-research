"""The user rule of 2026-10-24: the training entry points refuse the CNN corpus, accept a purpose-chosen one, and the default protocol
corpus (the CNN one) is refused."""
import json
import os
import tempfile
import unittest

from mline import corpus_rule
from mline import train as mt


class Rule(unittest.TestCase):
    def test_the_cnn_corpus_is_closed_and_another_is_open(self):
        with tempfile.TemporaryDirectory() as d:
            cnn, fw = os.path.join(d, 'cnn-bytes'), os.path.join(d, 'fineweb-edu-bytes')
            for path, source in ((cnn, 'cnn_dailymail'), (fw, 'HuggingFaceFW/fineweb-edu')):
                os.makedirs(path)
                with open(os.path.join(path, 'corpus.json'), 'w', encoding='utf8') as f:
                    json.dump(dict(source=source), f)
            with self.assertRaises(corpus_rule.ClosedCorpus):
                corpus_rule.check_trainable(cnn)
            self.assertEqual(corpus_rule.check_trainable(fw), 'HuggingFaceFW/fineweb-edu')
            renamed = os.path.join(d, 'news')
            os.makedirs(renamed)
            with open(os.path.join(renamed, 'corpus.json'), 'w', encoding='utf8') as f:
                json.dump(dict(source='cnn_dailymail'), f)
            with self.assertRaises(corpus_rule.ClosedCorpus):                     # the manifest decides, not only the name
                corpus_rule.check_trainable(renamed)

    def test_the_default_protocol_corpus_is_refused(self):
        if os.environ.get('MLINE_DATA'):
            self.skipTest('MLINE_DATA points elsewhere')
        with self.assertRaises(corpus_rule.ClosedCorpus):
            corpus_rule.check_trainable(mt.protocol()['data'])


if __name__ == '__main__':
    unittest.main()
