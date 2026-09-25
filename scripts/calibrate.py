"""Decide the M1 execution plan on the rented card by measuring it, not by reasoning about it.

Two questions the 8 GB machine at home settled by force and a 24 GB card re-opens:

  1. how large a batch is worth using;
  2. whether M1's three arms should run CONCURRENTLY on one GPU instead of one after another.

Neither has an answer that can be derived from what is already known. A compute-bound loop gains nothing from
sharing a card between three processes; a launch-bound one overlaps and turns M1's 22 hours into roughly 8. This
campaign has already been caught carrying a 5 % utilisation figure over from the w32 toy models to the 30M one
and concluding, wrongly, that a faster card buys little. So this measures, on the card that will do the work.

Bytes per second is the comparison, never seconds per step: a step at batch 64 moves twice the data of a step at
batch 32, so step time flatters the larger batch by construction.

Writes results/calibration.json and prints a plan. Costs a few minutes of a rented hour, which is the cheapest
part of any decision it informs.

Usage: python -B scripts/calibrate.py [--batches 32 48 64 96] [--workers 3]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROW = re.compile(r'^(\S+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+) GB')


def one_probe(arm, batch, device, steps):
    """Run the launch probe in its own process and return (seconds per step, bytes per second, peak GB)."""
    out = subprocess.run([sys.executable, '-B', '-m', 'mline.train', '--probe', '--device', device,
                          '--arms', arm, '--batches', str(batch)],
                         cwd=str(ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        return None, out.stderr.strip()[-400:]
    for line in out.stdout.splitlines():
        m = ROW.match(line.strip())
        if m and int(m.group(2)) == batch:
            return dict(arm=m.group(1), batch=batch, s_per_step=float(m.group(3)),
                        bytes_per_s=float(m.group(4)), peak_gb=float(m.group(5))), None
    return None, 'probe produced no parsable row:\n' + out.stdout[-400:]


def parallel_probe(arms, batch, device, steps):
    """The same probe in N processes at once. Aggregate throughput is what matters, not any one process's."""
    started = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, '-B', '-m', 'mline.train', '--probe', '--device', device,
                               '--arms', arm, '--batches', str(batch)],
                              cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for arm in arms]
    rows, errors = [], []
    for proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            errors.append(stderr.strip()[-200:]); continue
        for line in stdout.splitlines():
            m = ROW.match(line.strip())
            if m and int(m.group(2)) == batch:
                rows.append(dict(arm=m.group(1), s_per_step=float(m.group(3)), bytes_per_s=float(m.group(4)),
                                 peak_gb=float(m.group(5))))
    return dict(workers=len(arms), batch=batch, rows=rows, errors=errors,
                total_bytes_per_s=sum(r['bytes_per_s'] for r in rows),
                wall_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='*', default=[32, 48, 64, 96])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--out', default=str(ROOT / 'results/calibration.json'))
    args = parser.parse_args()

    from mline import train as mt
    p = mt.protocol()
    arms = list(p['arms'])[:args.workers]

    print('== single process, batch sweep ==')
    single = []
    for batch in args.batches:
        row, error = one_probe(arms[0], batch, args.device, args.steps)
        if row is None:
            print('batch %-4d FAILED (out of memory, most likely) - %s' % (batch, error.splitlines()[-1][:120]))
            break                                   # larger batches will fail too
        single.append(row)
        print('batch %-4d %7.3f s/step  %9.0f bytes/s  %5.2f GB peak' %
              (batch, row['s_per_step'], row['bytes_per_s'], row['peak_gb']))
    if not single:
        raise SystemExit('no batch size ran; the card or the setup is wrong, and nothing long should start')

    best = max(single, key=lambda r: r['bytes_per_s'])
    print('\nbest single-process batch: %d at %.0f bytes/s' % (best['batch'], best['bytes_per_s']))

    print('\n== %d processes at once, batch %d ==' % (len(arms), best['batch']))
    par = parallel_probe(arms, best['batch'], args.device, args.steps)
    for row in par['rows']:
        print('  %-9s %7.3f s/step  %9.0f bytes/s  %5.2f GB' %
              (row['arm'], row['s_per_step'], row['bytes_per_s'], row['peak_gb']))
    for e in par['errors']:
        print('  worker failed:', e.splitlines()[-1][:140] if e else '(no message)')

    speedup = par['total_bytes_per_s'] / best['bytes_per_s'] if best['bytes_per_s'] else 0.
    peak = max((r['peak_gb'] for r in par['rows']), default=0.) * max(len(par['rows']), 1)
    # A card shared three ways must beat one process by enough to be worth the extra failure modes. 1.5x is the
    # floor: below it the arms run one after another, which is also what the campaign has always done.
    plan = 'parallel' if (speedup >= 1.5 and not par['errors']) else 'serial'
    hours = p['steps'] * p['length'] * best['batch'] / max(par['total_bytes_per_s'] if plan == 'parallel'
                                                           else best['bytes_per_s'], 1) / 3600
    arms_total = hours if plan == 'parallel' else hours * len(p['arms'])

    print('\naggregate %.0f bytes/s across %d workers = %.2fx one process' %
          (par['total_bytes_per_s'], len(par['rows']), speedup))
    print('PLAN: run M1 %s at batch %d  ->  about %.1f hours for all %d arms'
          % (plan, best['batch'], arms_total, len(p['arms'])))
    if plan == 'serial':
        print('  (concurrency did not pay here; the arms go one after another, as they always have)')

    report = dict(kind='M1_CALIBRATION', utc=time.time(), single=single, best_single=best, parallel=par,
                  speedup=speedup, plan=plan, batch=best['batch'], estimated_hours_all_arms=arms_total,
                  combined_peak_gb_estimate=peak,
                  note='bytes per second, never seconds per step: a step at batch 64 moves twice the data of one '
                       'at batch 32. The 1.5x floor for choosing parallel is a judgement, recorded here so it can '
                       'be argued with rather than inferred from the outcome.')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
    print('written', args.out)


if __name__ == '__main__':
    main()
