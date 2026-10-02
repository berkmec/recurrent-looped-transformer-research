"""Z1 permutation pilots: first errors per position window, pooled over the evaluation points of a result file.

Every evaluation point is an independent draw of n sequences and reports, per position window, how many of the sequences that were still
error-free at the window's start made their FIRST error inside it (`hazard` in the result file). The same window from different points is summed:
first errors, and the exposure (sequences at risk times the window width), so a window's rate uses every draw that reaches it.
A constant rate over the windows is a per-step error floor (precision, budget); no first error early and a rate that appears later is a break.

  python -B -m horizon.hazard_summary results/Z1-pilot/perm5-core-w128-s1-cur.json results/Z1-pilot/perm5-gru-w256-s1-cur.json
"""
import argparse
import json


def pooled_windows(result):
    """{(start, stop): [first errors, exposure]} summed over the evaluation points of one result file."""
    pooled = {}
    for point in result['curve'].values():
        for window in point.get('hazard') or []:
            cell = pooled.setdefault((window['start'], window['stop']), [0, 0])
            cell[0] += window['first_errors']
            cell[1] += window['at_risk'] * (window['stop'] - window['start'])
    return dict(sorted(pooled.items()))


def rate(cell):
    errors, exposure = cell
    return errors / exposure if exposure else None


def describe(result, name=''):
    pooled = pooled_windows(result)
    errors, exposure = sum(c[0] for c in pooled.values()), sum(c[1] for c in pooled.values())
    cells = '  '.join(f"[{a},{b}) {c[0]}/{c[1]}={rate(c) * 1e4:.1f}" if c[1] else f"[{a},{b}) -" for (a, b), c in pooled.items())
    tail = f"all {errors}/{exposure}={errors / exposure * 1e4:.1f}" if exposure else 'no hazard data'
    return f"{name or result.get('task', '?')} {result.get('arm', '?')} w{result.get('width', '?')} ({result['train']['parameters']} parameters): {cells}  | {tail}  (rates x1e-4 per step)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', nargs='+')
    a = p.parse_args()
    for path in a.results:
        with open(path, encoding='utf8') as f:
            print(describe(json.load(f), path.replace('\\', '/').rsplit('/', 1)[-1]))


if __name__ == '__main__':
    main()
