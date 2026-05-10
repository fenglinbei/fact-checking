from __future__ import annotations

import torch
import torch.nn.functional as F



def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class labels [0, C-1] to CORAL binary targets [B, C-1]."""
    levels = []
    for k in range(num_classes - 1):
        levels.append((labels > k).float())
    return torch.stack(levels, dim=1)



def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = coral_targets(labels, num_classes=num_classes)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_decode(logits: "torch.Tensor | np.ndarray") -> "torch.Tensor | np.ndarray":
    """Decode CORAL logits [*, C-1] → class IDs [*] (0..C-1).

    Thresholds the cumulative probabilities P(y > k) at 0.5 and counts
    how many thresholds are passed, yielding the predicted class index.
    """
    import numpy as np

    if isinstance(logits, torch.Tensor):
        cum_p = torch.sigmoid(logits.float())
        return (cum_p > 0.5).sum(dim=-1)
    x = np.asarray(logits, dtype=np.float32)
    # Stable sigmoid: overflow-safe for large |x|
    pos = x >= 0
    z = np.empty_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    z[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return (z > 0.5).sum(axis=-1)
