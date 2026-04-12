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
