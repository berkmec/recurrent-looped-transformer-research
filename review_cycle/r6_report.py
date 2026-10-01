"""R6 summary from frozen result.json files: means of three measured real updates per arm/length.

Reads only; never launches workers. Ratios are relative to the full arm at the same length.
CUDA peaks are reset phase peaks (max across phases of one update); RAM is the process working set.
"""
import argparse
import json
from pathlib import Path
import statistics

from .memory_bench import ARMS, sha, verify_frozen

PHASES = ('forward', 'backward', 'encoder', 'segment_forward_backward', 'global_backward', 'optimizer')


def condition(out, arm, length, diagnostic=False):
    name = f'{arm}-L{length}' + ('-hooks' if diagnostic else '')
    path = out / 'measurements' / name / 'result.json'
    if not path.exists():
        failure = out / 'measurements' / (name + '.failure.json')
        return dict(arm=arm, length=length, diagnostic=diagnostic, status='MISSING' if not failure.exists() else 'FAIL',
                    detail=json.loads(failure.read_text()) if failure.exists() else None)
    r = json.loads(path.read_text(encoding='utf8'))
    row = dict(arm=arm, length=length, diagnostic=diagnostic, status=r['status'], result_sha256=sha(path),
               optimizer_updates=r['optimizer_updates'], host_busy_fraction=r['host_cpu_observation'].get('busy_fraction'),
               initial_state_hash=r.get('initial_state_hash'), final_state_hash=r.get('final_state_hash'))
    if r['status'] == 'OOM':
        row.update(cuda_at_failure=r.get('cuda_at_failure'), ram_at_failure=r.get('ram_at_failure'),
                   interrupted_phases=[p['phase'] for p in r.get('interrupted_phases', [])], error=r.get('error', '')[:300])
        return row
    ms = r['measurements']
    if len(ms) != r['protocol']['repeats']:
        raise ValueError('unexpected measured update count: ' + name)
    row['measured_updates'] = len(ms)
    row['mean_update_seconds'] = statistics.mean(m['seconds'] for m in ms)
    row['update_seconds'] = [m['seconds'] for m in ms]
    row['losses'] = [m['loss'] for m in ms]
    row['phase_mean_seconds'] = {ph: statistics.mean(sum(p['seconds'] for p in m['phases'] if p['phase'] == ph) for m in ms)
                                 for ph in PHASES if any(p['phase'] == ph for p in ms[0]['phases'])}
    row['cuda_peak_allocated_bytes'] = max(m['cuda']['peak_allocated'] for m in ms)
    row['cuda_peak_reserved_bytes'] = max(m['cuda']['peak_reserved'] for m in ms)
    row['cuda_allocated_after_optimizer_bytes'] = max(p['cuda']['allocated'] for m in ms for p in m['phases'] if p['phase'] == 'optimizer')
    row['ram_lifetime_peak_working_set_bytes'] = max(p['ram']['lifetime_peak_working_set_bytes'] for m in ms for p in m['phases'])
    row['ram_working_set_after_optimizer_bytes'] = max(p['ram']['working_set_bytes'] for m in ms for p in m['phases'] if p['phase'] == 'optimizer')
    row['ram_private_commit_bytes'] = max(p['ram']['private_commit_bytes'] or 0 for m in ms for p in m['phases'])
    if diagnostic:
        row['saved_peak_unique_storage_bytes'] = max((p['saved'] or {}).get('peak_unique_storage_bytes', 0) for m in ms for p in m['phases'])
        row['saved_requests'] = max((p['saved'] or {}).get('save_requests', 0) for m in ms for p in m['phases'])
    return row


def summarize(out):
    out = Path(out)
    frozen = verify_frozen(out)
    p = frozen['protocol']
    rows = [condition(out, arm, length) for length in p['lengths'] for arm in ARMS]
    diagnostics = [r for r in (condition(out, arm, length, True) for length in p['lengths'] for arm in ARMS) if r['status'] != 'MISSING']
    for length in p['lengths']:
        base = next((r for r in rows if r['length'] == length and r['arm'] == 'full' and r['status'] == 'PASS'), None)
        for r in rows:
            if r['length'] == length and r['status'] == 'PASS' and base is not None:
                r['ratio_to_full'] = dict(update_seconds=r['mean_update_seconds'] / base['mean_update_seconds'],
                                          cuda_peak_allocated=r['cuda_peak_allocated_bytes'] / base['cuda_peak_allocated_bytes'],
                                          cuda_peak_reserved=r['cuda_peak_reserved_bytes'] / base['cuda_peak_reserved_bytes'])
    eq_path = out / 'validation' / 'equivalence.json'
    equivalence = json.loads(eq_path.read_text()) if eq_path.exists() else None
    value = dict(protocol_hash=frozen['protocol_hash'], sources=frozen['sources'],
                 equivalence=None if equivalence is None else dict(status=equivalence['status'], sha256=sha(eq_path),
                     comparisons=[dict(base=c['base'], candidate=c['candidate'], length=c['length'], status=c['status'],
                                       max_abs=max(v['max_abs'] for v in c['metrics'].values())) for c in equivalence['comparisons']]),
                 counts=dict(planned=len(p['lengths']) * len(ARMS), pass_=sum(r['status'] == 'PASS' for r in rows),
                             oom=sum(r['status'] == 'OOM' for r in rows), fail=sum(r['status'] == 'FAIL' for r in rows),
                             missing=sum(r['status'] == 'MISSING' for r in rows)),
                 scope='Three measured real AdamW updates per fresh process after two warmups; seconds include explicit synchronization at phase marks; CUDA peaks are reset phase peaks, not resident VRAM of the whole process; RAM is Windows working set including the Python/PyTorch runtime; host busy fraction is system-wide over the worker lifetime.',
                 rows=rows, diagnostics=diagnostics)
    return value


def markdown(value):
    def mb(b):
        return f'{b / 2**20:.1f}'
    lines = ['| Uzunluk | Kol | Durum | Güncelleme s (3 ort.) | Full oranı | CUDA tepe alloc MiB | CUDA tepe reserved MiB | Süreç WS tepe MiB | Host meşgul |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in value['rows']:
        if r['status'] == 'PASS':
            ratio = r.get('ratio_to_full', {}).get('update_seconds', float('nan'))
            lines.append(f"| {r['length']} | {r['arm']} | PASS | {r['mean_update_seconds']:.4f} | {ratio:.3f} | "
                         f"{mb(r['cuda_peak_allocated_bytes'])} | {mb(r['cuda_peak_reserved_bytes'])} | {mb(r['ram_lifetime_peak_working_set_bytes'])} | {r['host_busy_fraction']:.2f} |")
        elif r['status'] == 'OOM':
            c = r.get('cuda_at_failure') or {}
            lines.append(f"| {r['length']} | {r['arm']} | OOM | - | - | {mb(c.get('peak_allocated', 0))} (hata ani) | {mb(c.get('peak_reserved', 0))} (hata ani) | - | {r['host_busy_fraction']:.2f} |")
        else:
            lines.append(f"| {r['length']} | {r['arm']} | {r['status']} | - | - | - | - | - | - |")
    for d in value['diagnostics']:
        if d['status'] == 'PASS':
            lines.append(f"| {d['length']} | {d['arm']} (hooks tani) | PASS | {d['mean_update_seconds']:.4f} | - | {mb(d['cuda_peak_allocated_bytes'])} | {mb(d['cuda_peak_reserved_bytes'])} | {mb(d['ram_lifetime_peak_working_set_bytes'])} | saklanan depolama tepe {mb(d['saved_peak_unique_storage_bytes'])} MiB |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--write', action='store_true', help='write summary.json next to frozen.json')
    args = parser.parse_args()
    value = summarize(args.out)
    if args.write:
        path = Path(args.out) / 'summary.json'
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf8')
        value['summary_sha256'] = sha(path)
    print(markdown(value))
    print(json.dumps(dict(counts=value['counts'], equivalence=None if value['equivalence'] is None else value['equivalence']['status'],
                          summary_sha256=value.get('summary_sha256')), ensure_ascii=False))


if __name__ == '__main__':
    main()
