"""R6: real-update memory measurement with explicit gradient routes.

For chunk j, L_j=sum(CE_j)/(B*T). Let u=(e,mk,mv)=g(theta).
Detach u into leaves, backward L_j for each chunk with detached s/dk/dv,
then apply J_g(theta)^T sum_j dL_j/du once. Decoder parameter gradients
accumulate directly. This is the original detach-only TBPTT gradient route;
it is not the full recurrence route. Encoder checkpoint recomputes g only.
"""
import argparse
from contextlib import nullcontext
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from core.model import Core, generate

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('full', 'detach', 'checkpoint', 'segment', 'segment_encoder_checkpoint')
SOURCES = ('core/model.py', 'review_cycle/memory_bench.py', 'review_cycle/test_memory_bench.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def protocol(device='cuda'):
    return dict(version=1, purpose='R6 real optimizer update memory diagnostics',
        device=device, width=32, heads=2, batch=8, lengths=[32, 128, 512, 1024],
        chunk=8, precision='float32', seed=701, data_seed=970100,
        warmup=2, repeats=3, threads=1, arms=list(ARMS), lr=.003, weight_decay=.01,
        clip=None, window=4, alpha=.1, optimizer='AdamW foreach=False',
        route_full='all recurrence and global encoder/projection paths attached',
        route_truncated='detach s/dk/dv each8; global e/mk/mv gradients retained',
        loss='parity dense sum cross entropy divided by batch*length for every chunk',
        process='one fresh subprocess per arm and length; no allocator sharing',
        data='independent deterministic batch per optimizer update; paired across arms',
        memory='CUDA phase live allocated/reserved and reset phase peak allocated/reserved; whole-update peaks reconstructed as max of phase peaks; separate process working set',
        hooks='separate diagnostic run; deduplicated backing-storage save lifetime, not resident VRAM; inner nonreentrant checkpoint hooks shadow outer hooks so checkpoint-interior saves are not fully observed',
        host_load='external CPU training may run concurrently; CPU/GPU latency is scoped to recorded host load, not isolated throughput',
        equivalence=dict(rtol=3e-4, atol=3e-6, lengths=[32, 128], repeats=3),
        oom='record OOM with unchanged batch/precision/length; no automatic retry')


def validate_protocol(p):
    for key in ('width', 'heads', 'batch', 'chunk', 'warmup', 'repeats', 'threads'):
        if not isinstance(p[key], int) or p[key] < 1:
            raise ValueError('positive integer required: '+key)
    if p['warmup'] < 2 or p['threads'] != 1 or p['precision'] != 'float32':
        raise ValueError('two optimizer warmups, one thread and float32 required')
    if p['device'] not in ('cpu', 'cuda') or p['arms'] != list(ARMS):
        raise ValueError('invalid device/arms')
    if not p['lengths'] or any(not isinstance(n, int) or n < 1 for n in p['lengths']):
        raise ValueError('invalid length ladder')


def freeze(out, p):
    validate_protocol(p)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('freeze requires empty output directory')
    value = dict(protocol=p, protocol_hash=digest(p), sources=sources(), frozen_utc=time.time())
    write_json(out/'frozen.json', value)
    return value


def verify_frozen(out):
    value = json.loads((Path(out)/'frozen.json').read_text())
    if digest(value['protocol']) != value['protocol_hash']:
        raise ValueError('frozen protocol hash mismatch')
    validate_protocol(value['protocol'])
    if value['sources'] != sources():
        raise ValueError('frozen source drift')
    return value


def global_inputs(model, x):
    e, _, _ = model.encode(x)
    n = model.mn(e)
    return e, model.rope(model.split(model.mk(n))), model.split(model.mv(n))


def backward_loss(model, x, y, arm, chunk=8, bridge_global=True, mark=None):
    if arm not in ARMS or chunk < 1:
        raise ValueError('invalid arm/chunk')
    mark = mark or (lambda _: None)
    if arm not in ('segment', 'segment_encoder_checkpoint'):
        z = checkpoint(model, x, use_reentrant=False) if arm == 'checkpoint' else model(x, tbptt=chunk if arm == 'detach' else 0)
        value = F.cross_entropy(z.flatten(0, 1), y.flatten())
        mark('forward')
        value.backward()
        mark('backward')
        return float(value.detach())
    original = (checkpoint(lambda token: global_inputs(model, token), x, use_reentrant=False)
                if arm == 'segment_encoder_checkpoint' else global_inputs(model, x))
    leaves = tuple(v.detach().requires_grad_(True) for v in original)
    e, mk, mv = leaves
    mark('encoder')
    state = None; total = 0.
    for start in range(0, x.shape[1], chunk):
        if state is not None:
            state = {k: state[k].detach() for k in ('s', 'dk', 'dv')}
        outputs = []
        for t in range(start, min(start+chunk, x.shape[1])):
            z, state = model.decode(e[:, t], state, mk[:, :, :t+1], mv[:, :, :t+1], t)
            outputs.append(z)
        logits = torch.stack(outputs, 1)
        value = F.cross_entropy(logits.flatten(0, 1), y[:, start:start+len(outputs)].flatten(), reduction='sum') / y.numel()
        value.backward()
        total += float(value.detach())
        del outputs, logits, value, z
    # The final decoder state may retain references, but backward freed saves.
    del state
    mark('segment_forward_backward')
    if bridge_global:
        torch.autograd.backward(original, tuple(v.grad for v in leaves))
    mark('global_backward')
    return total


class _Saved:
    def __init__(self, tensor, owner, key):
        self.tensor, self.owner, self.key = tensor, owner, key

    def __del__(self):
        self.owner.release(self.key)


class SavedStorageTracker:
    """Tracks only backing storage referenced by saved autograd tensors.

    Parameter/input storage is included when saved; outputs not saved by
    autograd are excluded. This metric is neither total allocation nor VRAM.
    """
    def __init__(self):
        self.live = {}; self.live_bytes = 0; self.peak = 0; self.requests = 0

    def pack(self, tensor):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        n = storage.nbytes(); self.requests += 1
        if key not in self.live:
            self.live[key] = [0, n]; self.live_bytes += n
        self.live[key][0] += 1
        self.peak = max(self.peak, self.live_bytes)
        # Returning the original Tensor inside a container could retain its
        # grad_fn and form reference cycles; detach preserves backing storage.
        return _Saved(tensor.detach(), self, key)

    def release(self, key):
        self.live[key][0] -= 1
        if self.live[key][0] == 0:
            self.live_bytes -= self.live.pop(key)[1]

    @staticmethod
    def unpack(saved):
        return saved.tensor

    def snapshot(self):
        return dict(live_unique_storage_bytes=self.live_bytes,
                    peak_unique_storage_bytes=self.peak, save_requests=self.requests,
                    live_unique_storages=len(self.live), scope='autograd saved backing storage only; not resident VRAM')


def process_ram():
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        c = Counters(); c.cb = ctypes.sizeof(c)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        call = ctypes.windll.psapi.GetProcessMemoryInfo
        call.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not call(handle(), ctypes.byref(c), c.cb):
            raise ctypes.WinError()
        return dict(working_set_bytes=c.WorkingSetSize, lifetime_peak_working_set_bytes=c.PeakWorkingSetSize, private_commit_bytes=c.PagefileUsage)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    current = None
    if Path('/proc/self/statm').exists():
        current = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    return dict(working_set_bytes=current, lifetime_peak_working_set_bytes=peak, private_commit_bytes=None)


def host_cpu_times():
    """System-wide counters; delta is observed host load, not this process CPU."""
    if os.name == 'nt':
        idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError()
        return dict(idle=idle.value, total=kernel.value+user.value, unit='100ns')
    if Path('/proc/stat').exists():
        values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        return dict(idle=values[3]+values[4], total=sum(values[:8]), unit='ticks')
    return None


class Phases:
    def __init__(self, device, tracker=None):
        self.device, self.tracker, self.rows = device, tracker, []
        self.sync(); self.started = self.previous = time.perf_counter()
        self.reset()

    def sync(self):
        if self.device == 'cuda': torch.cuda.synchronize()

    def reset(self):
        if self.device == 'cuda': torch.cuda.reset_peak_memory_stats()

    def mark(self, name):
        self.sync(); now = time.perf_counter()
        cuda = None
        if self.device == 'cuda':
            cuda = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
        self.rows.append(dict(phase=name, seconds=now-self.previous, cuda=cuda,
                             ram=process_ram(), saved=None if self.tracker is None else self.tracker.snapshot()))
        self.previous = time.perf_counter(); self.reset()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return value


def worker(out, p, arm, length, diagnostic=False):
    validate_protocol(p)
    if arm not in p['arms'] or length not in p['lengths']:
        raise ValueError('condition outside protocol')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('worker output must be empty')
    torch.set_num_threads(1)
    torch.manual_seed(p['seed'])
    if p['device'] == 'cuda':
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    result = dict(status='RUNNING', pid=os.getpid(), protocol=p, protocol_hash=digest(p), sources=sources(),
        arm=arm, length=length, diagnostic=diagnostic, optimizer_updates=0, measurements=[],
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if p['device']=='cuda' else None,
            logical_cpu_count=os.cpu_count(), host_load_note=p['host_load']),
        optimizer_initialized_before_measurement=False, started=time.time())
    write_json(out/'started.json', result)
    host_start = host_cpu_times()
    try:
        model = Core(width=p['width'], heads=p['heads'], window=p['window'], alpha=p['alpha']).to(p['device'])
        opt = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'], foreach=False)
        result['initial_state_hash'] = state_hash(model)
        torch.save(cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict())), out/'initial.pt')
        result['initial_checkpoint_sha256'] = sha(out/'initial.pt')
        for step in range(p['warmup']+p['repeats']):
            x, y = generate('parity', p['batch'], length, p['data_seed']+step)
            x, y = x.to(p['device']), y.to(p['device'])
            opt.zero_grad(set_to_none=True)
            if step == p['warmup']:
                result['optimizer_initialized_before_measurement'] = bool(opt.state) and all(float(v['step']) == p['warmup'] for v in opt.state.values())
                if not result['optimizer_initialized_before_measurement']: raise ValueError('optimizer state not initialized')
            tracker = SavedStorageTracker() if diagnostic and step >= p['warmup'] else None
            phases = Phases(p['device'], tracker)
            phases.mark('ready')
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack) if tracker else nullcontext():
                loss = backward_loss(model, x, y, arm, p['chunk'], mark=phases.mark)
            opt.step(); phases.mark('optimizer')
            result['optimizer_updates'] += 1
            if step >= p['warmup']:
                cuda_rows = [r['cuda'] for r in phases.rows if r['cuda'] is not None]
                result['measurements'].append(dict(update=step+1, loss=loss,
                    seconds=sum(r['seconds'] for r in phases.rows), phases=phases.rows,
                    cuda=None if not cuda_rows else dict(peak_allocated=max(r['peak_allocated'] for r in cuda_rows),
                                                       peak_reserved=max(r['peak_reserved'] for r in cuda_rows)),
                    saved=None if tracker is None else tracker.snapshot()))
            with (out/'updates.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(dict(step=step+1, loss=loss, measured=step>=p['warmup'], pid=os.getpid()))+'\n')
                f.flush(); os.fsync(f.fileno())
        state = cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                             gradients={name: param.grad for name, param in model.named_parameters()}))
        torch.save(state, out/'final.pt')
        result.update(status='PASS', final_state_hash=state_hash(model), final_checkpoint_sha256=sha(out/'final.pt'))
        if result['initial_state_hash'] == result['final_state_hash']: raise ValueError('no parameter update')
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='OOM', error=str(exc), ram_at_failure=process_ram(),
                      interrupted_phases=phases.rows if 'phases' in locals() else [])
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc)); write_json(out/'result.json', result); raise
    result['finished'] = time.time()
    host_end = host_cpu_times()
    result['host_cpu_observation'] = dict(start=host_start, end=host_end, busy_fraction=None)
    if host_start and host_end and host_end['total'] > host_start['total']:
        result['host_cpu_observation']['busy_fraction'] = 1-(host_end['idle']-host_start['idle'])/(host_end['total']-host_start['total'])
    write_json(out/'result.json', result)
    return result


def launch(out, p, arm, length, diagnostic=False):
    out = Path(out)
    if out.exists(): raise ValueError('refusing existing condition directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    request = out.with_name(out.name+'.request.json')
    write_json(request, dict(protocol=p, arm=arm, length=length, diagnostic=diagnostic, sources=sources()))
    for name, expected in sources().items():
        snapshot = out.parent/'source-snapshots'/expected/Path(name).name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists(): snapshot.write_bytes((ROOT/name).read_bytes())
        if sha(snapshot) != expected: raise ValueError('source snapshot mismatch')
    command = [sys.executable, '-m', 'review_cycle.memory_bench', '_worker', '--out', str(out), '--request', str(request)]
    with out.with_name(out.name+'.stdout.log').open('w') as stdout, out.with_name(out.name+'.stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        write_json(out.with_name(out.name+'.launch.json'), dict(pid=process.pid, command=command, started=time.time()))
        code = process.wait()
    if not (out/'result.json').exists():
        write_json(out.with_name(out.name+'.failure.json'), dict(status='FAIL', exit_code=code, pid=process.pid))
        raise RuntimeError('worker failed without result: '+str(out))
    value = json.loads((out/'result.json').read_text())
    if code != 0 or value['status'] not in ('PASS', 'OOM'): raise RuntimeError('worker failed: '+str(out))
    return value


def compare_states(a, b, rtol, atol):
    metrics = {}
    def compare(x, y, path):
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor): raise ValueError('tensor mismatch '+path)
            torch.testing.assert_close(x, y, rtol=rtol, atol=atol, msg=path)
            metrics[path] = dict(max_abs=float((x-y).abs().max()), reference_norm=float(x.float().norm()))
        elif isinstance(x, dict):
            if x.keys() != y.keys(): raise ValueError('keys mismatch '+path)
            for k in x: compare(x[k], y[k], path+'/'+str(k))
        elif isinstance(x, list):
            if len(x) != len(y): raise ValueError('length mismatch '+path)
            for i, (u, v) in enumerate(zip(x, y)): compare(u, v, path+'/'+str(i))
        elif x != y: raise ValueError('state mismatch '+path)
    compare(a, b, '')
    return metrics


def validate_routes(out, p):
    out = Path(out)
    if out.exists(): raise ValueError('validation output already exists')
    out.mkdir(parents=True)
    validation_p = dict(p, lengths=p['equivalence']['lengths'], repeats=p['equivalence']['repeats'])
    comparisons = []
    for length in validation_p['lengths']:
        for arm in ARMS:
            r = launch(out/f'{arm}-L{length}', validation_p, arm, length)
            if r['status'] != 'PASS': raise ValueError('validation condition did not PASS')
        for base, candidate in [('full', 'checkpoint'), ('detach', 'segment'), ('detach', 'segment_encoder_checkpoint')]:
            left = torch.load(out/f'{base}-L{length}'/'final.pt', weights_only=True)
            right = torch.load(out/f'{candidate}-L{length}'/'final.pt', weights_only=True)
            metrics = compare_states(left, right, **{k:p['equivalence'][k] for k in ('rtol', 'atol')})
            lr = json.loads((out/f'{base}-L{length}'/'result.json').read_text())
            rr = json.loads((out/f'{candidate}-L{length}'/'result.json').read_text())
            torch.testing.assert_close(torch.tensor([v['loss'] for v in lr['measurements']]),
                                       torch.tensor([v['loss'] for v in rr['measurements']]),
                                       rtol=p['equivalence']['rtol'], atol=p['equivalence']['atol'])
            comparisons.append(dict(base=base, candidate=candidate, length=length, status='PASS', metrics=metrics,
                left_checkpoint_sha256=sha(out/f'{base}-L{length}'/'final.pt'), right_checkpoint_sha256=sha(out/f'{candidate}-L{length}'/'final.pt')))
    value = dict(status='PASS', protocol_hash=digest(p), sources=sources(), comparisons=comparisons)
    write_json(out/'equivalence.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pilot', 'freeze', 'validate', 'run', '_worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--arm', choices=ARMS, default='segment')
    parser.add_argument('--length', type=int, default=32)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        print(json.dumps(freeze(args.out, protocol(args.device)))); return
    if args.action == 'pilot':
        p = protocol(args.device)
        print(json.dumps(launch(args.out, p, args.arm, args.length, args.diagnostic))); return
    if args.action == '_worker':
        request = json.loads(args.request.read_text())
        if request['sources'] != sources(): raise ValueError('source changed since launch')
        print(json.dumps(worker(args.out, request['protocol'], request['arm'], request['length'], request['diagnostic']))); return
    frozen = verify_frozen(args.out); p = frozen['protocol']
    if args.action == 'validate':
        print(json.dumps(validate_routes(args.out/'validation', p))); return
    evidence = json.loads((args.out/'validation'/'equivalence.json').read_text())
    if evidence['status'] != 'PASS' or evidence['protocol_hash'] != frozen['protocol_hash'] or evidence['sources'] != sources():
        raise ValueError('matching route equivalence required before main measurement')
    suffix = '-hooks' if args.diagnostic else ''
    print(json.dumps(launch(args.out/'measurements'/f'{args.arm}-L{args.length}{suffix}', p, args.arm, args.length, args.diagnostic)))


if __name__ == '__main__':
    main()
