"""Checkpoints that can actually resume a run, written so an interruption cannot cost more than one interval.

What this replaces: a single `latest.pt` holding model weights only, rewritten in place every 250 steps, with no
optimizer moments, no step counter and no load path anywhere in the M-line. An interruption cost the whole run;
an interruption DURING the write cost the only copy. That is survivable for a 9-hour probe, reckless for the
22-hour M1, and disqualifying for the multi-day runs a larger model would need - and this campaign has already
lost a run to a power cut.

Three properties, each pinned by tests/test_checkpoint_resume.py:

  ROUND TRIP  weights, optimizer moments and the step number come back exactly as they went in. Dropping the
              optimizer state is not a small loss: AdamW's moments are the run's memory of its own trajectory,
              and restarting without them is a different run that happens to share weights.
  ATOMIC      the new checkpoint is written to a temporary file and renamed. A crash before the rename leaves
              the live file untouched; os.replace is atomic on Windows and POSIX alike.
  ROTATION    the previous checkpoint is kept, so one corrupt or truncated file is never fatal.

Old weights-only files stay readable, because one is being written by a probe that is running right now. Their
`step` comes back as None rather than as a guess - a resume that invents its own position is worse than one that
refuses.
"""
import json
import os
import shutil
from pathlib import Path

import torch

LATEST = 'latest.pt'
PREVIOUS = 'previous.pt'


def save(path, model, optimizer=None, step=None, meta=None):
    """Write atomically, keeping the file it replaces as `previous.pt` in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `meta` goes through JSON first. Two reasons, one of which cost an hour: `validate` returns bits per byte
    # as a numpy float64 (it divides by np.log(2)), and torch.load(weights_only=True) refuses numpy scalars, so
    # the checkpoint wrote fine and then could not be read - the worst possible failure for a resume. The second
    # reason is that a checkpoint whose metadata cannot be serialised is one whose metadata cannot be inspected.
    payload = dict(format=1, step=step, meta=json.loads(json.dumps(dict(meta or {}), default=float)),
                   model={k: v.detach().cpu() for k, v in model.state_dict().items()})
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    tmp = Path(str(path) + '.tmp')
    torch.save(payload, tmp)
    if path.exists():
        shutil.copy2(path, path.parent / PREVIOUS)      # copy, not move: the live file stays readable throughout
    os.replace(tmp, path)                               # atomic; a crash before this leaves `path` as it was
    return path


def load(path, map_location='cpu'):
    """Read a checkpoint. Old weights-only files are accepted and reported as having no step."""
    state = torch.load(Path(path), weights_only=True, map_location=map_location)
    if isinstance(state, dict) and state.get('format') == 1:
        return dict(model=state['model'], optimizer=state.get('optimizer'), step=state.get('step'),
                    meta=state.get('meta') or {}, legacy=False)
    return dict(model=state, optimizer=None, step=None, meta={}, legacy=True)


def load_latest(directory, map_location='cpu', report=None):
    """The newest loadable checkpoint in `directory`, or None if there is none.

    Falls back past a corrupt file, which is the entire reason the previous one is kept - but never silently. A
    swallowed load error looks exactly like "there was nothing to resume", and the run would then be discarded
    and restarted with no one the wiser. Every failure is reported to `report` (default: print), and if files
    exist but NONE of them load, that is an error rather than an absence.
    """
    directory = Path(directory)
    report = report if report is not None else (lambda m: print('  checkpoint: ' + m, flush=True))
    present, failures = [], []
    for name in (LATEST, PREVIOUS):
        candidate = directory / name
        if not candidate.exists():
            continue
        present.append(name)
        try:
            state = load(candidate, map_location)
        except Exception as exc:                        # truncated, half-written, or not a torch file at all
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
            report(f'{name} is unreadable ({type(exc).__name__}), falling back')
            continue
        if failures:
            report(f'resuming from {name} instead')
        return state
    if present:
        joined = ('\\n  ').join(failures)
        raise ValueError('checkpoints exist but none could be read, so nothing may be assumed about this run:' + '\\n  ' + joined)
    return None


def resume_step(directory):
    """Where a run should continue from: the next step after the newest checkpoint, or 1 if there is none."""
    state = load_latest(directory)
    return 1 if state is None or state['step'] is None else int(state['step']) + 1
