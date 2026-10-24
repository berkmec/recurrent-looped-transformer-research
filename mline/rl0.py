"""E-RL0 (program note 5.68): the first supervised fine-tuning (SFT) and reinforcement learning (RL) of the RLT language model, on the RLT paper's
own addition task, against the same steps on the twin without feedback. Evaluation and training on the CPU.

The paper (Zhang, Feng, Qin 2026) trains addition on 1-8-digit operands with the digits serialized least significant first, scores teacher-forced
answer-token accuracy and finds it falls beyond the trained widths (9 digits: RLT-1 7+1 68.05 +- 3.64 %); it gives the replay rule for policy
gradients (a sampled response is replayed under the current parameters; prompt tokens update the state and carry gradients, only sampled tokens get
the action term) and says RL performance is not measured. Here, for L1 (feedback on, 30,518 updates) and A0F (the twin without feedback):

  format  separator + rev(a) '+' rev(b) '=' -> rev(a + b) '\\n'   (bytes; rev: least significant digit first; no leading zeros)
  SFT     2,000 updates of 24 problems (each operand 1-8 digits) + 8 snippets of 64 training-corpus bytes; loss 0.75 x mean answer-byte
          cross-entropy + 0.25 x mean snippet cross-entropy; AdamW lr 1e-4 (100-update warm-up), weight decay 0.01, clip 1
  RL      500 steps: 8 problems (operands 1-10 digits) x 8 samples at temperature 1; reward 1 for the exact answer (newline included), else 0;
          advantage (r - group mean) / (group std + 1e-6); one update per batch (on-policy, ratio 1): loss = -mean_i A_i mean_t log p(sampled
          byte) + 0.02 x mean k3 KL to the frozen SFT model; every sample replayed under the current parameters; AdamW lr 3e-5, clip 1
  eval    200 fixed problems per width w in {1, 4, 8, 9, 10, 12, 16} (both operands w digits): greedy exact match and the paper's teacher-forced
          answer-digit accuracy; validation bits per byte on 64 x 512 bytes (forgetting); before SFT, after SFT, after RL

Generation runs one full forward per new byte, the computation the update replays, so sampling and replay cannot disagree.

Reading rule (written before the first run, program note 5.68); E = mean greedy exact match over a set of widths, intervals from a paired
bootstrap over the problems within every width (2,000 draws, seed 0):
  SFT LEARNED (per arm)   E_after_SFT over {1, 4, 8} >= 0.5; otherwise that arm's RL questions are VACUOUS
  Q1 RL (L1 registered, A0F descriptive)  delta = E_RL - E_SFT over the RL range {1, 4, 8, 9, 10}:
      RL HELPS   delta >= +0.05 and interval > 0;   RL HURTS   delta <= -0.05 and interval < 0;
      NO EFFECT  interval inside (-0.05, +0.05);    INCONCLUSIVE otherwise
  Q2 feedback and length  D = E_L1 - E_A0F after RL over {9, 10, 12} (beyond the SFT range; 12 also beyond the RL range):
      FEEDBACK HELPS LENGTH  D >= +0.10 and interval > 0;   FEEDBACK HURTS LENGTH  D <= -0.10 and interval < 0;
      NO DIFFERENCE  interval inside (-0.10, +0.10);       INCONCLUSIVE otherwise;   VACUOUS unless both arms SFT LEARNED
Descriptive: token accuracy against the paper's, width 16, D after SFT, the forgetting (bits per byte), the reward curve.

  python -B -m mline.rl0 run results/L1-scratch-w928-s401/latest.pt --out results/RL0/L1
  python -B -m mline.rl0 reading results/RL0/L1/final.json results/RL0/A0F/final.json
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from horizon import guards as gu
from horizon import sweep as sw
from mline import blimp as bl
from mline import checkpoint as ck
from mline import data as md
from mline import lm as ml

LN2 = math.log(2)
SEP = md.SEPARATOR
NEWLINE = 10
SFT_WIDTHS, RL_WIDTHS = (1, 8), (1, 10)
EVAL_WIDTHS, EVAL_N = (1, 4, 8, 9, 10, 12, 16), 200
IN_RANGE, RL_RANGE, OUT_RANGE = (1, 4, 8), (1, 4, 8, 9, 10), (9, 10, 12)
SFT = dict(steps=2000, problems=24, snippets=8, snippet_bytes=64, lr=1e-4, warmup=100, answer_weight=0.75, ckpt_every=250,
           schedule='constant', lr_min=0.0, betas=(0.9, 0.999), weight_decay=0.01, micro=None)
RL = dict(steps=500, prompts=8, group=8, temperature=1.0, beta=0.02, lr=3e-5, ckpt_every=50)
# E-RL1 (note 5.69): E-RL0 with the paper's addition budget on the GPU (global batch 512, AdamW (0.9, 0.95), weight decay 0.1, warm-up 200,
# cosine decay to 5e-6), the snippet share kept at a fifth of the rows, micro-batches of 128 problems + 32 snippets; RL with 32 prompts x 8 samples
SFT_RL1 = dict(SFT, problems=512, snippets=128, warmup=200, schedule='cosine', lr_min=5e-6, betas=(0.9, 0.95), weight_decay=0.1, micro=4)
RL_RL1 = dict(RL, prompts=32)
LEARNED, Q1_EFFECT, Q2_EFFECT, DRAWS = 0.5, 0.05, 0.10, 2000


def number(rng, width):
    """A uniform `width`-digit non-negative integer without leading zeros (0-9 for one digit)."""
    return int(rng.integers(0 if width == 1 else 10 ** (width - 1), 10 ** width))


def problem(a, b):
    """(prompt, answer) bytes in the paper's format: digits least significant first; the answer ends with a newline."""
    return f'{str(a)[::-1]}+{str(b)[::-1]}='.encode(), f'{str(a + b)[::-1]}\n'.encode()


def draw(rng, widths):
    wa, wb = (int(rng.integers(widths[0], widths[1] + 1)) for _ in range(2))
    return problem(number(rng, wa), number(rng, wb))


def eval_set(width, n=EVAL_N):
    """n fixed problems with both operands `width` digits, from their own random stream (training draws use other seeds)."""
    rng = np.random.default_rng([7919, width])
    return [problem(number(rng, width), number(rng, width)) for _ in range(n)]


def device_of(model):
    return next(model.parameters()).device


def grid(rows, device=None):
    """(B, T) int64 tensor of right-padded byte rows (padding 0) and the row lengths; causal models ignore the padding."""
    g = np.zeros((len(rows), max(len(r) for r in rows)), dtype=np.int64)
    for i, r in enumerate(rows):
        g[i, :len(r)] = np.frombuffer(bytes(r), dtype=np.uint8)
    return torch.from_numpy(g).to(device or 'cpu'), [len(r) for r in rows]


def answer_logprobs(model, prompts, answers):
    """[(len(answer),) log-probabilities in nats of every answer byte given the separator and its prompt]: one teacher-forced forward over
    the whole history, which is the paper's replay of a sampled response under the current parameters."""
    x, _ = grid([SEP + p + a for p, a in zip(prompts, answers)], device_of(model))
    logp = F.log_softmax(ml.forward_logits(model, x[:, :-1]).float(), -1)
    out = []
    for i, (p, a) in enumerate(zip(prompts, answers)):
        idx = torch.arange(len(SEP) + len(p) - 1, len(SEP) + len(p) - 1 + len(a), device=x.device)
        out.append(logp[i, idx, x[i, idx + 1]])
    return out


def generate(model, prompts, max_new, greedy=False, generator=None, temperature=1.0):
    """Continuations (bytes up to and including the first newline, or max_new bytes) of the separator plus each prompt."""
    rows, out, live = [bytearray(SEP + p) for p in prompts], [bytearray() for _ in prompts], list(range(len(prompts)))
    with torch.no_grad():
        for _ in range(max_new):
            if not live:
                break
            x, lengths = grid([rows[i] for i in live], device_of(model))
            last = ml.forward_logits(model, x)[torch.arange(len(live), device=x.device), torch.tensor(lengths, device=x.device) - 1].float()
            # sampling on the CPU: one seeded generator gives the same samples on either device
            nxt = last.argmax(-1) if greedy else torch.multinomial(F.softmax(last / temperature, -1).cpu(), 1, generator=generator).squeeze(1)
            for k, i in enumerate(live):
                rows[i].append(int(nxt[k]))
                out[i].append(int(nxt[k]))
            live = [i for i in live if out[i][-1] != NEWLINE]
    return [bytes(o) for o in out]


def advantages(rewards, group):
    """Group-relative advantages: (r - group mean) / (group std + 1e-6); a group with one reward value gets 0."""
    r = rewards.view(-1, group)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True, unbiased=False) + 1e-6)).view(-1)


def sft_terms(model, problems, snippets):
    """Summed cross-entropy (nats) of the answer bytes and of the snippet bytes of one batch, and their counts."""
    rows = [SEP + p + a for p, a in problems] + list(snippets)
    x, _ = grid(rows, device_of(model))
    ce = F.cross_entropy(ml.forward_logits(model, x[:, :-1]).float().transpose(1, 2), x[:, 1:], reduction='none')
    answer, text = torch.zeros_like(ce, dtype=torch.bool), torch.zeros_like(ce, dtype=torch.bool)
    for i, (p, a) in enumerate(problems):
        answer[i, len(SEP) + len(p) - 1:len(SEP) + len(p) - 1 + len(a)] = True
    for j, s in enumerate(snippets):
        text[len(problems) + j, :len(s) - 1] = True
    return ce[answer].sum(), int(answer.sum()), ce[text].sum(), int(text.sum())


def sft_loss(model, problems, snippets, answer_weight=SFT['answer_weight']):
    a_sum, a_n, t_sum, t_n = sft_terms(model, problems, snippets)
    a, t = a_sum / a_n, t_sum / t_n if t_n else a_sum.new_zeros(())
    return answer_weight * a + (1 - answer_weight) * t, float(a), float(t)


def sft_step(model, opt, problems, snips, cfg):
    """One guarded SFT update -> (report, mean answer CE, mean snippet CE). With cfg['micro'] = m > 1 the batch is cut into m micro-batches
    (every m-th problem and snippet), each backpropagated at once, the loss normalised by the full batch's byte counts so the gradients add up
    to the full batch's; the guards are horizon.guards.guarded_step's (finite loss and gradients, raw norm, clip, then the step)."""
    w, m = cfg['answer_weight'], cfg.get('micro') or 1
    if m == 1:
        loss, a, t = sft_loss(model, problems, snips, w)
        return gu.guarded_step(opt, model.parameters(), loss, clip_norm=1.0), a, t
    params = list(model.parameters())
    opt.zero_grad(set_to_none=True)
    a_total, t_total = sum(len(a) for _, a in problems), sum(len(s) - 1 for s in snips)
    a_acc = t_acc = 0.0
    report = dict(applied=False, finite=False, loss=None, grad_norm_before_clip=None, clipped=False)
    for k in range(m):
        a_sum, _, t_sum, _ = sft_terms(model, problems[k::m], snips[k::m])
        loss = w * a_sum / a_total + ((1 - w) * t_sum / t_total if t_total else 0.0)
        if not bool(torch.isfinite(loss.detach())):
            for p in params:
                p.grad = None
            report['reason'] = 'nonfinite_loss'
            return report, float('nan'), float('nan')
        loss.backward()
        a_acc, t_acc = a_acc + float(a_sum), t_acc + float(t_sum)
    a, t = a_acc / a_total, (t_acc / t_total if t_total else 0.0)
    report['loss'] = w * a + (1 - w) * t
    grads = [p.grad for p in params if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        for p in params:
            p.grad = None
        report['reason'] = 'nonfinite_gradient'
        return report, a, t
    norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    report.update(finite=True, grad_norm_before_clip=norm)
    if norm > 1.0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        report['clipped'] = True
    opt.step()
    report['applied'] = True
    return report, a, t


def rl_step(model, ref, opt, rng, generator, cfg=RL):
    problems = [draw(rng, RL_WIDTHS) for _ in range(cfg['prompts'])]
    prompts = [p for p, _ in problems for _ in range(cfg['group'])]
    answers = [a for _, a in problems for _ in range(cfg['group'])]
    model.eval()
    samples = generate(model, prompts, max(len(a) for a in answers) + 1, generator=generator, temperature=cfg['temperature'])
    rewards = torch.tensor([float(s == a) for s, a in zip(samples, answers)])
    adv = advantages(rewards, cfg['group'])
    model.train()
    logps = answer_logprobs(model, prompts, samples)
    with torch.no_grad():
        refs = answer_logprobs(ref, prompts, samples)
    pg = torch.stack([-float(adv[i]) * lp.mean() for i, lp in enumerate(logps)]).mean()
    kl = torch.stack([(torch.exp(r - lp) - (r - lp) - 1).mean() for r, lp in zip(refs, logps)]).mean()
    report = gu.guarded_step(opt, model.parameters(), pg + cfg['beta'] * kl, clip_norm=1.0)
    widths = [max(len(p.split(b'+')[0]), len(p.split(b'+')[1]) - 1) for p in prompts]
    return dict(reward=float(rewards.mean()), pg=float(pg), kl=float(kl), applied=bool(report['applied']),
                grad_norm=report['grad_norm_before_clip'], reward_wide=float(rewards[torch.tensor(widths) >= 9].mean()) if max(widths) >= 9 else None)


def bits(model, rows, batch=16):
    """Validation bits per byte of (n, 513) byte rows (512 predictions each)."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for at in range(0, len(rows), batch):
            x = torch.from_numpy(rows[at:at + batch]).to(device_of(model))
            total += float(F.cross_entropy(ml.forward_logits(model, x[:, :-1]).float().transpose(1, 2), x[:, 1:], reduction='sum'))
            count += x[:, 1:].numel()
    return total / count / LN2


def evaluate(model, val_rows=None, widths=None, n=EVAL_N, batch=100):
    model.eval()
    out = {}
    for w in (EVAL_WIDTHS if widths is None else widths):
        probs, exact, token = eval_set(w, n), [], []
        for at in range(0, n, batch):
            chunk = probs[at:at + batch]
            gen = generate(model, [p for p, _ in chunk], max(len(a) for _, a in chunk) + 1, greedy=True)
            exact += [int(g == a) for g, (_, a) in zip(gen, chunk)]
            x, _ = grid([SEP + p + a for p, a in chunk], device_of(model))
            with torch.no_grad():
                pred = ml.forward_logits(model, x[:, :-1]).argmax(-1)
            for i, (p, a) in enumerate(chunk):
                s = len(SEP) + len(p) - 1
                token.append(float((pred[i, s:s + len(a) - 1] == x[i, s + 1:s + len(a)]).float().mean()))
        out[str(w)] = dict(exact=float(np.mean(exact)), token=float(np.mean(token)), items=exact)
    if val_rows is not None:
        out['bits_per_byte'] = bits(model, val_rows)
    return out


def validation_rows(val, n=64, length=512):
    return np.asarray(val[:n * (length + 1)]).astype(np.int64).reshape(n, length + 1)


def snippets(corpus, rng, n, length):
    starts = rng.integers(0, len(corpus) - length - 1, size=n)
    return [bytes(np.asarray(corpus[s:s + length + 1])) for s in starts]


def sft_lr(cfg, step):
    """Linear warm-up, then constant (E-RL0) or cosine decay to lr_min over the remaining updates (E-RL1, the paper's schedule)."""
    if step <= cfg['warmup'] or cfg.get('schedule', 'constant') == 'constant':
        return cfg['lr'] * min(1.0, step / cfg['warmup'])
    done = (step - cfg['warmup']) / max(1, cfg['steps'] - cfg['warmup'])
    return cfg['lr_min'] + 0.5 * (cfg['lr'] - cfg['lr_min']) * (1 + math.cos(math.pi * min(1.0, done)))


def run(weights, out, seed=1, corpus=None, val_rows=None, model=None, arm=None, sft=SFT, rl=RL, log=print, eval_n=EVAL_N, widths=None,
        device='cpu'):
    """Base evaluation, SFT, its evaluation, RL, its evaluation; every stage resumes from its own checkpoint or result file."""
    widths = EVAL_WIDTHS if widths is None else widths
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.json').exists():
        return json.loads((out / 'final.json').read_bytes().decode('utf8'))
    if model is None:
        model, _, arm = bl.load(weights)
    model = model.to(device)
    started = time.perf_counter()

    def stage_eval(name):
        path = out / f'eval-{name}.json'
        if not path.exists():
            sw.write_json(path, evaluate(model, val_rows, widths=widths, n=eval_n))
            r = json.loads(path.read_bytes().decode('utf8'))
            log(f'  eval {name}: ' + ' '.join(f'{w}:{r[w]["exact"]:.2f}' for w in map(str, widths)) + f'  bpb {r.get("bits_per_byte")}')
        return json.loads(path.read_bytes().decode('utf8'))

    base = stage_eval('base')
    opt = torch.optim.AdamW(model.parameters(), lr=sft['lr'], betas=tuple(sft.get('betas', (0.9, 0.999))), weight_decay=sft.get('weight_decay', 0.01))
    start, resume = 1, ck.load_latest(out / 'sft')
    if resume is not None and resume['step']:
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
    for step in range(start, sft['steps'] + 1):
        for g in opt.param_groups:
            g['lr'] = sft_lr(sft, step)
        rng = np.random.default_rng([seed, 1, step])
        problems = [draw(rng, SFT_WIDTHS) for _ in range(sft['problems'])]
        model.train()
        snips = snippets(corpus, rng, sft['snippets'], sft['snippet_bytes']) if sft['snippets'] else []
        report, a, t = sft_step(model, opt, problems, snips, sft)
        sw.append_line(out / 'sft.jsonl', dict(step=step, answer_ce=a, text_ce=t, applied=bool(report['applied']),
                                              seconds=round(time.perf_counter() - started, 2)))
        if step % sft['ckpt_every'] == 0 or step == sft['steps']:
            ck.save(out / 'sft' / 'latest.pt', model, opt, step)
        if step % 100 == 0:
            log(f'  sft {step}: answer {a:.4f} text {t:.4f} ({time.perf_counter() - started:.0f} s)')
    after_sft = stage_eval('sft')

    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=rl['lr'], weight_decay=0.0)
    start, resume = 1, ck.load_latest(out / 'rl')
    if resume is not None and resume['step']:
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        start = int(resume['step']) + 1
    for step in range(start, rl['steps'] + 1):
        rng = np.random.default_rng([seed, 2, step])
        generator = torch.Generator().manual_seed(seed * 1_000_003 + step)
        row = rl_step(model, ref, opt, rng, generator, rl)
        sw.append_line(out / 'rl.jsonl', dict(step=step, seconds=round(time.perf_counter() - started, 2), **row))
        if step % rl['ckpt_every'] == 0 or step == rl['steps']:
            ck.save(out / 'rl' / 'latest.pt', model, opt, step)
        if step % 10 == 0:
            log(f'  rl {step}: reward {row["reward"]:.3f} kl {row["kl"]:.4f} ({time.perf_counter() - started:.0f} s)')
    after_rl = stage_eval('rl')
    final = dict(weights=str(weights), arm=arm, seed=seed, base=base, sft=after_sft, rl=after_rl, sft_config=sft, rl_config=rl)
    sw.write_json(out / 'final.json', final)
    return final


def mean_exact(stage, widths):
    return float(np.mean([stage[str(w)]['exact'] for w in widths]))


def paired_interval(a, b, widths, draws=DRAWS, seed=0):
    """Mean over the widths of (a - b) exact match, and its 95% interval: bootstrap over the problems within every width, same indices for both."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([np.mean(a[str(w)]['items']) - np.mean(b[str(w)]['items']) for w in widths]))
    boot = np.zeros(draws)
    for w in widths:
        x, y = np.array(a[str(w)]['items'], dtype=float), np.array(b[str(w)]['items'], dtype=float)
        idx = rng.integers(0, len(x), size=(draws, len(x)))
        boot += x[idx].mean(1) - y[idx].mean(1)
    boot /= len(widths)
    return point, [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def verdict(delta, ci, effect, up, down, none):
    if delta >= effect - 1e-12 and ci[0] > 0:
        return up
    if delta <= -effect + 1e-12 and ci[1] < 0:
        return down
    if ci[0] > -effect and ci[1] < effect:
        return none
    return 'INCONCLUSIVE'


def reading(l1, a0):
    learned = {name: mean_exact(r['sft'], IN_RANGE) >= LEARNED for name, r in (('L1', l1), ('A0F', a0))}
    q1 = {}
    for name, r in (('L1', l1), ('A0F', a0)):
        d, ci = paired_interval(r['rl'], r['sft'], RL_RANGE)
        q1[name] = dict(delta=d, ci=ci, verdict=verdict(d, ci, Q1_EFFECT, 'RL HELPS', 'RL HURTS', 'NO EFFECT') if learned[name] else 'VACUOUS')
    d2, ci2 = paired_interval(l1['rl'], a0['rl'], OUT_RANGE)
    d2_sft, ci2_sft = paired_interval(l1['sft'], a0['sft'], OUT_RANGE)
    q2 = dict(delta=d2, ci=ci2, after_sft=dict(delta=d2_sft, ci=ci2_sft),
              verdict=verdict(d2, ci2, Q2_EFFECT, 'FEEDBACK HELPS LENGTH', 'FEEDBACK HURTS LENGTH', 'NO DIFFERENCE') if all(learned.values()) else 'VACUOUS')
    table = {name: {stage: {w: r[stage][str(w)]['exact'] for w in EVAL_WIDTHS} for stage in ('base', 'sft', 'rl')} for name, r in (('L1', l1), ('A0F', a0))}
    token = {name: {stage: {w: r[stage][str(w)]['token'] for w in EVAL_WIDTHS} for stage in ('sft', 'rl')} for name, r in (('L1', l1), ('A0F', a0))}
    forgetting = {name: {stage: r[stage].get('bits_per_byte') for stage in ('base', 'sft', 'rl')} for name, r in (('L1', l1), ('A0F', a0))}
    return dict(sft_learned=learned, q1=q1, q2=q2, exact=table, token=token, bits_per_byte=forgetting)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('weights')
    r.add_argument('--out', required=True)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--threads', type=int, default=2)
    r.add_argument('--smoke', action='store_true', help='a few steps of every stage, small evaluation, for timing and memory only')
    r.add_argument('--config', choices=('rl0', 'rl1'), default='rl0', help='rl0: note 5.68 (CPU budget); rl1: note 5.69 (the paper\'s budget)')
    r.add_argument('--device', default='cpu')
    g = sub.add_parser('reading')
    g.add_argument('l1')
    g.add_argument('a0')
    a = p.parse_args()
    if a.cmd == 'reading':
        load = lambda f: json.loads(Path(f).read_bytes().decode('utf8'))
        print(json.dumps(reading(load(a.l1), load(a.a0)), indent=1))
        return
    torch.set_num_threads(a.threads)
    from mline import train as mt
    protocol = mt.protocol()
    from mline import corpus_rule
    corpus_rule.check_trainable(protocol['data'])          # user rule 2026-10-24: the SFT snippets are training data too (note 5.71)
    corpus, val = md.load(protocol['data'], 'train'), md.load(protocol['data'], 'validation')
    sft, rl = (dict(SFT), dict(RL)) if a.config == 'rl0' else (dict(SFT_RL1), dict(RL_RL1))
    if a.smoke:
        sft.update(steps=6, ckpt_every=3)
        rl.update(steps=4, ckpt_every=2)
    # the smoke test times the stages on unregistered cells (widths 1 and 2, 5 problems) so no registered number is seen before the run
    run(a.weights, a.out, seed=a.seed, corpus=corpus, val_rows=validation_rows(val, n=8 if a.smoke else 64), sft=sft, rl=rl,
        log=lambda m: print(m, flush=True), eval_n=5 if a.smoke else EVAL_N, widths=(1, 2) if a.smoke else None, device=a.device)


if __name__ == '__main__':
    main()
