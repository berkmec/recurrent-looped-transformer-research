"""U4-sep: an auxiliary loss on the geometry of the state code.

U2d measured that robust and fragile gated models are separated with no overlap by one quantity — the minimum
distance between class anchors relative to the drift seen during a neutral run — while absolute drift is identical
in both groups. That finding is correlational. This loss makes it testable causally: it directly maximises the
separation ratio (minimum centre distance / mean within-class spread), which is exactly what the diagnostic reports.

Deliberately scale invariant: multiplying every state by a constant leaves the ratio unchanged, so the loss cannot
be satisfied by merely inflating the state norm — it has to reorganise the code.
"""
import torch

EPS = 1e-6


def _centres_and_spread(z, y, classes):
    present, centres, spreads = [], [], []
    for c in range(classes):
        m = y == c
        if not bool(m.any()):
            continue
        zc = z[m]
        mu = zc.mean(0)
        present.append(c); centres.append(mu)
        spreads.append(torch.linalg.vector_norm(zc - mu, dim=-1).mean())
    if len(present) < 2:
        return None, None
    return torch.stack(centres), torch.stack(spreads)


def separation_ratio(z, y, classes):
    """Minimum pairwise centre distance divided by the mean within-class spread (the U2d diagnostic quantity)."""
    centres, spreads = _centres_and_spread(z, y, classes)
    if centres is None:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    d = torch.cdist(centres, centres)
    n = centres.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return d[off].min() / (spreads.mean() + EPS)


def separation_loss(z, y, classes):
    """-log(ratio): decreasing it widens the anchor separation relative to the within-class spread.

    Returns exactly 0.0 when fewer than two classes are present in the batch, so degenerate batches contribute
    nothing rather than a spurious gradient.
    """
    centres, _ = _centres_and_spread(z, y, classes)
    if centres is None:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return -torch.log(separation_ratio(z, y, classes) + EPS)
