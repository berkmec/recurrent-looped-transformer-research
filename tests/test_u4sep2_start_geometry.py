"""The corrected starting geometry, checked against the report that states it rather than against my memory of it.

The lesson an independent reviewer left this campaign was that an audit check whose evidence is a sentence rather
than a computation is not a check. Section 8 of `notes/U4sep2_RAPORU.md` is a table of numbers I typed. These
tests re-read every one of them out of `results/U4sep2/start-geometry.json` and refuse to pass on a mismatch, so
the report cannot drift from the evidence it cites the way `notes/U4sep_RAPORU.md` did for 28 minutes.

They also pin the defect itself. `ratio_u2c` lives in a file sealed by the U4-sep2 r2 manifest and is wrong by up
to 8.35x; it is not repaired in place, because repairing it would break a seal to correct a field that no verdict,
sign test, audit check or report sentence reads. If someone later does change it, the first test here fails and
asks them to say so out loud.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / 'results/U4sep2/start-geometry.json'
ANALYSIS = ROOT / 'results/U4sep2/analysis.json'
REPORT = ROOT / 'notes/U4sep2_RAPORU.md'
PAIRS = ('soft/s117', 'soft/s143', 'supervised/s117', 'supervised/s143', 'supervised/s191')


def _load(path):
    if not path.exists():
        raise unittest.SkipTest('MISSING_ASSET: ' + str(path))
    return json.loads(path.read_text(encoding='utf8'))


class StartGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = _load(START)['pairs']
        cls.analysis = _load(ANALYSIS)['pairs']
        cls.report = REPORT.read_text(encoding='utf8')

    def test_the_defect_is_still_exactly_what_the_report_describes(self):
        """ratio_u2c == the control arm's LAST fine-tuning ratio, in a sealed file, left in place deliberately."""
        for key in PAIRS:
            self.assertEqual(self.analysis[key]['ratio_u2c'], self.analysis[key]['ratio_none'],
                             f'{key}: ratio_u2c no longer equals ratio_none - the sealed file changed, say so')

    def test_the_measurement_used_the_frozen_checkpoint(self):
        """The whole point is that the start is read from the source model, not from a fine-tuned one."""
        for key in PAIRS:
            arm, seed = key.split('/s')
            train = json.loads((ROOT / 'results/U4sep2' / f'u4sep2-none-from_{arm}-s{seed}' / 'train.json')
                               .read_text(encoding='utf8'))
            self.assertEqual(self.start[key]['source_sha256'], train['source_sha256'],
                             f'{key}: the start geometry was measured on a different model than the pair loaded')

    def test_the_ratio_is_gap_over_drift(self):
        for key, row in self.start.items():
            self.assertAlmostEqual(row['start_ratio'], row['gap'] / row['drift'], places=6, msg=key)

    def test_the_reported_factors_match_the_evidence(self):
        for key, row in self.start.items():
            self.assertAlmostEqual(row['factor'], row['recorded_ratio_u2c'] / row['start_ratio'], places=6, msg=key)
        self.assertAlmostEqual(max(r['factor'] for r in self.start.values()), 8.35, places=1,
                               msg='the report headlines 8.35x as the worst case')

    def test_every_number_in_both_report_tables_comes_from_the_evidence(self):
        """Section 8 states two tables of numbers I typed. Both are recomputed here, cell by cell."""
        section = self.report[self.report.index('## 8. '):]
        checked = 0
        for key in PAIRS:
            rows = re.findall(r'^\| ' + re.escape(key) + r' \|(.+)$', section, re.M)
            self.assertEqual(len(rows), 2, f'{key} should appear once in each of the two section 8 tables')
            e, a = self.start[key], self.analysis[key]

            first = self.cells(rows[0])                      # start / recorded ratio_u2c / factor
            for got, expected in zip(first, [e['start_ratio'], e['recorded_ratio_u2c'], e['factor']]):
                self.assertAlmostEqual(float(got), expected, places=2,
                                       msg=f'{key} table 1: report {got}, evidence {expected:.3f}')
                checked += 1

            second = self.cells(rows[1])                     # start / control / hinge / two deltas / repair pair
            want = [e['start_ratio'], e['eval_ratio_none'], e['eval_ratio_hinge'],
                    e['eval_ratio_none'] - e['start_ratio'], e['eval_ratio_hinge'] - e['start_ratio']]
            for got, expected in zip(second[:5], want):
                self.assertAlmostEqual(float(got), expected, places=3,
                                       msg=f'{key} table 2: report {got}, evidence {expected:.3f}')
                checked += 1
            repair = [float(c) for c in second[5].split('→')]
            self.assertAlmostEqual(repair[0], a['none_repair'], places=1, msg=key)
            self.assertAlmostEqual(repair[1], a['hinge_repair'], places=1, msg=key)
            checked += 2

        self.assertEqual(checked, 50, 'five pairs: three numbers in the first table, seven in the second')

    @staticmethod
    def cells(row):
        keep = '0123456789.-→'                 # digits, decimal point, minus and the "->" that joins a repair pair
        out = []
        for c in row.split('|'):
            c = c.strip().replace(',', '.').replace('−', '-')
            if c:
                out.append(''.join(ch for ch in c if ch in keep))
        return out

    def test_the_report_does_not_turn_the_n_equals_two_observation_into_a_finding(self):
        section = self.report[self.report.index('## 8. '):]
        self.assertIn('n=2', section, 'the pattern must be labelled as an observation with its sample size')
        self.assertIn('gözlem', section)


if __name__ == '__main__':
    unittest.main()
