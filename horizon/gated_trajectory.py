"""U2d: what distinguishes a robust gated solution from a fragile one?

U2c left a concrete puzzle: soft at lr 0.003 (hold 0.389) survives a 512-step neutral run with accuracy 1.000, while
soft at lr 0.0003 (hold 0.394) collapses to 0.182. Nearly identical gate behaviour, opposite robustness — so the
answer is in the states, not in how often the gate holds.

This module measures, for a GatedCore, the same quantities the U2 layer measured for the plain Core, plus the norm
growth that the Fully Looped Transformer paper calls residual explosion:
  - committed-state trajectories and the gate decisions that produced them,
  - per-class anchors on the training distribution and their separation,
  - displacement during a neutral run, measured from the state just before the run,
  - the RMS norm curve of the committed state.
No training, no protocol: this is a diagnostic over already-frozen checkpoints.
"""
import numpy as np
import torch


@torch.no_grad()
def run_states(model, x, batch=32):
    """Committed states, gate values and logits for a GatedCore, batched."""
    model.eval()
    states, gates, logits = [], [], []
    for at in range(0, len(x), batch):
        z, g, _ = model(x[at:at + batch])
        logits.append(z); gates.append(g); states.append(model.last_states)
    return dict(states=torch.cat(states), gates=torch.cat(gates), logits=torch.cat(logits))


def norm_curve(states):
    """RMS norm of the committed state at each position, averaged over the batch."""
    return [float(s.square().mean().sqrt()) for s in states.transpose(0, 1)]


def displacement_curve(states, start):
    """Distance from the state just before position `start`, for every position from `start` on."""
    ref = states[:, start - 1] if start > 0 else torch.zeros_like(states[:, 0])
    return [float(torch.linalg.vector_norm(states[:, t] - ref, dim=-1).mean()) for t in range(start, states.shape[1])]


def anchor_summary(z, labels, classes):
    """Class centroids of the hidden states, their minimum separation and within-class spread."""
    z = np.asarray(z, dtype=np.float64); labels = np.asarray(labels)
    centers, spread, counts = [], [], []
    for c in range(classes):
        m = labels == c
        counts.append(int(m.sum()))
        if not m.any():
            centers.append(np.full(z.shape[1], np.nan)); spread.append(float('nan')); continue
        mu = z[m].mean(0); centers.append(mu)
        spread.append(float(np.linalg.norm(z[m] - mu, axis=1).mean()))
    centers = np.stack(centers)
    dists = [float(np.linalg.norm(centers[i] - centers[j])) for i in range(classes) for j in range(i + 1, classes)
             if np.isfinite(centers[i]).all() and np.isfinite(centers[j]).all()]
    finite_spread = [s for s in spread if np.isfinite(s)]
    mean_spread = float(np.mean(finite_spread)) if finite_spread else float('nan')
    min_d = min(dists) if dists else float('nan')
    return dict(centers=centers, counts=counts, spread=spread, min_center_distance=min_d, mean_spread=mean_spread,
                separation_ratio=(min_d / mean_spread) if mean_spread and mean_spread > 0 else float('inf'))


def nearest_class(states, centers):
    """Index of the nearest class centre for each state."""
    c = torch.as_tensor(np.asarray(centers), dtype=states.dtype)
    flat = states.reshape(-1, states.shape[-1])
    return torch.cdist(flat, c).argmin(1).reshape(states.shape[:-1])
