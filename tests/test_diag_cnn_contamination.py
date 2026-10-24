"""scripts/diag_cnn_contamination.py: documents are split on the corpus separator, and the traced row starts give the rows
mline.data.batch returns, so each validation row can be labelled by the documents it comes from."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from mline import data as md

spec = importlib.util.spec_from_file_location('diag', Path(__file__).resolve().parents[1] / 'scripts' / 'diag_cnn_contamination.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class Diag(unittest.TestCase):
    def test_documents_split_on_the_separator(self):
        raw = b'alpha' + diag.SEP + b'beta gamma' + diag.SEP + b'delta' + diag.SEP
        self.assertEqual([raw[s:e] for s, e in diag.documents(raw)], [b'alpha', b'beta gamma', b'delta'])
        self.assertEqual(diag.SEP, md.SEPARATOR)

    def test_traced_rows_match_the_batch_function(self):
        stream = np.random.default_rng(3).integers(0, 256, size=20_000).astype(np.uint8)
        x, y, _ = md.batch(stream, diag.SEED, 5, diag.BATCH, diag.LENGTH)
        tx, ty, _ = diag.rows(stream, [5])
        self.assertTrue(np.array_equal(x, tx) and np.array_equal(y, ty))


if __name__ == '__main__':
    unittest.main()
