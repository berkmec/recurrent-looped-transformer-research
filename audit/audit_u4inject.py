"""Inline audit of the U4-inject cycle: recompute every claim in notes/U4inject_RAPORU.md from raw evidence.

Includes the two claims the report makes that are not simple table lookups: that the baseline shows no residual
explosion, and that the gradient comparison between arms is confounded because one arm learns and the other does not.
Also checks the source-level claim that both injection arms route the state through the query alone.
"""
import glob, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from horizon import sweep as sw

OUT = ROOT / 'results/U4inj'
ARMS, SEEDS = ('additive', 'inject', 'inject_mix'), (117, 143, 191)
checks, disc = [], []

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add(cid, ok, ev):
    checks.append(dict(id=cid, status='PASS' if ok else 'FAIL', evidence=ev))
    if not ok: disc.append(cid + ': ' + ev)

man = json.loads((OUT / 'frozen.json').read_text(encoding='utf8'))
problems = []
if sha(OUT / 'frozen.json') != (OUT / 'frozen.sha256').read_text(encoding='ascii').strip(): problems.append('sidecar')
if man['protocol_hash'] != sw.digest(man['protocol']): problems.append('protocol_hash')
for n, s in man['sources'].items():
    if sha(ROOT / n) != s: problems.append('drift ' + n)
add('1-freeze', not problems, f"{len(man['sources'])} sources, hash {man['protocol_hash'][:16]}; problems=" + (', '.join(problems) or 'none'))

done = sorted(Path(p).parent.name for p in glob.glob(str(OUT / 'u4inj-*/evaluation.json')))
add('2-completeness', len(done) == 9, f'{len(done)}/9 conditions evaluated')

problems = []
rows = {}
for arm in ARMS:
    for s in SEEDS:
        d = OUT / f'u4inj-{arm}-five-w32-s{s}'
        t = json.loads((d / 'train.json').read_text(encoding='utf8')); e = json.loads((d / 'evaluation.json').read_text(encoding='utf8'))
        rows[(arm, s)] = (t, e)
        if sha(d / 'selected.pt') != t['selected_sha256'] or sha(d / 'updates.jsonl') != t['updates_sha256']: problems.append(f'{arm}/{s}: hash')
        n = len((d / 'updates.jsonl').read_text(encoding='utf8').strip().splitlines())
        if n != man['protocol']['steps']: problems.append(f'{arm}/{s}: {n} rows')
add('3-integrity', not problems, f'9 conditions: checkpoint/updates hashes and {man["protocol"]["steps"]} rows each; problems=' + (', '.join(problems) or 'none'))

learned = {a: sum(1 for s in SEEDS if rows[(a, s)][0]['selected_validation']['final_accuracy'] > .9) for a in ARMS}
add('4-headline', learned['additive'] == 3 and learned['inject'] == 0 and learned['inject_mix'] == 0,
    f"arms that learned (validation > 0.9): additive {learned['additive']}/3, inject {learned['inject']}/3, inject_mix {learned['inject_mix']}/3")

chance = [rows[(a, s)][0]['selected_validation']['final_accuracy'] for a in ('inject', 'inject_mix') for s in SEEDS]
add('5-chance', all(abs(v - .2) < .12 for v in chance), f'all 6 injection validations within 0.12 of the 0.200 chance level: {[round(v,3) for v in chance]}')

growth = [rows[('additive', s)][1]['norm_growth'] for s in SEEDS]
add('6-no-residual-explosion', all(g < 1.25 for g in growth), f'control state RMS growth across a 512-step neutral run: {[round(g,2) for g in growth]} - no growth in 3/3, so the premise the cycle was built on is absent from the baseline')

inj_growth = [rows[(a, s)][1]['norm_growth'] for a in ('inject', 'inject_mix') for s in SEEDS]
add('7-injection-shrinks', all(g < .6 for g in inj_growth), f'injection arms shrink instead: {[round(g,2) for g in inj_growth]}')

a_rows = [json.loads(l) for l in (OUT / 'u4inj-additive-five-w32-s117' / 'updates.jsonl').read_text(encoding='utf8').splitlines()]
i_rows = [json.loads(l) for l in (OUT / 'u4inj-inject-five-w32-s117' / 'updates.jsonl').read_text(encoding='utf8').splitlines()]
import statistics as st
ga = st.median([r['grad_norm'] for r in a_rows[:64]]); gi = st.median([r['grad_norm'] for r in i_rows[:64]])
la = st.mean([r['loss'] for r in a_rows[:64]]); li = st.mean([r['loss'] for r in i_rows[:64]])
add('8-gradient-confound', abs(ga - gi) < .02 and abs(la - li) < .02,
    f'over the first 64 steps, before either arm learned: loss {la:.4f} vs {li:.4f}, median gradient {ga:.3f} vs {gi:.3f} - indistinguishable, so the later gradient difference cannot be attributed to the mechanism')

late = st.median([r['grad_norm'] for r in a_rows[1024:]])
add('9-peak-vs-median', late < st.median([r['grad_norm'] for r in i_rows[1024:]]),
    f"the control's LATE median gradient {late:.4f} is below the inject arm's {st.median([r['grad_norm'] for r in i_rows[1024:]]):.4f}; its 1.78e4 figure is a rare spike, not its typical value")

src = (ROOT / 'horizon/inject.py').read_text(encoding='utf8')
query_only = 'a = self.io(core.attend(core.rope(core.split(self.iq(r)), t), k, v)[:, 0])' in src
mix_on_input = 'z = w[:, :1] * e + w[:, 1:] * a' in src
add('10-state-is-query-only', query_only and mix_on_input,
    'source confirms both injection arms build `a` from a query on the previous state over the INPUT-derived k/v, and that inject_mix mixes e with a - so neither arm passes the state\'s content, and neither tests channel width')

# 11 - the arms differ in TWO terms, measured by running them rather than by matching source text (check 10)
import torch as _t
from core.model import Core as _Core
from horizon import inject as _ij

def _cache_depends_on_state(model):
    """Feed one input with two different previous states; does the window key cache change?"""
    core = getattr(model, 'core', model)
    _t.manual_seed(2)
    es = _t.randn(4, 3, 16)
    n = core.mn(es[0].unsqueeze(1))
    mk, mv = core.rope(core.split(core.mk(n))), core.split(core.mv(n))
    state = None
    for t in range(3):
        _, state = model.decode(es[t], state, mk, mv, t)
    a, b = dict(state), dict(state)
    _t.manual_seed(7)
    a['s'] = _t.randn_like(state['s']); b['s'] = _t.randn_like(state['s'])
    ka = model.decode(es[3], a, mk, mv, 3)[1]['dk']
    kb = model.decode(es[3], b, mk, mv, 3)[1]['dk']
    za = model.decode(es[3], a, mk, mv, 3)[1]['s']
    zb = model.decode(es[3], b, mk, mv, 3)[1]['s']
    return (not _t.allclose(ka, kb, atol=1e-6)), (not _t.allclose(za, zb, atol=1e-6))

_t.manual_seed(0)
_sealed = _Core(width=16, heads=2, window=4, classes=5, feedback=True, alpha=.1)
sealed_cache, _ = _cache_depends_on_state(_sealed)
arm_cache, arm_out = {}, {}
for _mode in ('inject', 'inject_mix'):
    _t.manual_seed(0)
    _m = _ij.InjectCore(_Core(width=16, heads=2, window=4, classes=5, feedback=True, alpha=.1), mode=_mode)
    arm_cache[_mode], arm_out[_mode] = _cache_depends_on_state(_m)
add('11-two-terms-not-one', sealed_cache and not any(arm_cache.values()) and all(arm_out.values()),
    'measured by running the decoders, not by matching source text: with one input and two different previous '
    f'states, the sealed Core window cache CHANGES ({sealed_cache}) because its k/v come from the post-feedback z, '
    f'while inject ({arm_cache["inject"]}) and inject_mix ({arm_cache["inject_mix"]}) leave it bitwise identical '
    'because theirs come from e. So the arms differ in two terms - how the state enters AND whether it reaches the '
    'sliding-window cache - and the 6/6 failure cannot be attributed to the injection mechanism alone. The arms '
    f'do remain recurrent through the query (output changes: {arm_out}), so this is a confound, not an inert arm. '
    'Report section 7 states the bound; the separating arm (injection plus k/v from the post-injection z) is not built.')

summary = dict(**{k: sum(1 for c in checks if c['status'] == k.upper()) for k in ('pass', 'fail')}, unverified=0)
report = dict(cycle='U4-inject', auditor='same session, inline script', utc=time.time(), checks=checks, summary=summary, discrepancies=disc,
              table={f'{a}/s{s}': dict(validation=rows[(a,s)][0]['selected_validation']['final_accuracy'],
                                       norm_growth=rows[(a,s)][1]['norm_growth'], separation=rows[(a,s)][1]['separation_ratio'],
                                       grad_max=rows[(a,s)][0]['guard']['max_grad_norm_before_clip']) for a in ARMS for s in SEEDS})
o = ROOT / '.superpowers/sdd/phase2'
(o / 'audit-u4inject.json').write_text(json.dumps(report, indent=1, default=float) + '\n', encoding='utf8')
(o / 'audit-u4inject.md').write_text('\n'.join([f'# U4-inject denetimi — {summary["pass"]} PASS / {summary["fail"]} FAIL', ''] +
    [f'- **{c["id"]}** — {c["status"]}: {c["evidence"]}' for c in checks]) + '\n', encoding='utf8')
print(json.dumps(dict(summary=summary, discrepancies=disc), indent=1))
for c in checks: print(c['status'], c['id'], '-', c['evidence'][:145])
