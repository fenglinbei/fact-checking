from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_class_balanced_weights(
    label_ids: Iterable[int],
    num_classes: int = 6,
    beta: float = 0.999,
) -> torch.Tensor:
    counts = Counter(int(x) for x in label_ids)
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 0)
        if n <= 0:
            weights.append(0.0)
        else:
            effective_num = 1.0 - beta ** n
            weight = (1.0 - beta) / effective_num
            weights.append(weight)
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum().clamp_min(1e-12) * num_classes
    return weights


def cumulative_ordinal_targets(labels: torch.Tensor, num_classes: int = 6) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


class GraphVerifierCriterion(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_ordinal: float = 0.4,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.lambda_ordinal = lambda_ordinal
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(outputs["class_logits"], labels, weight=self.class_weights)
        total = ce
        out = {"ce": ce}

        if outputs.get("ordinal_logits") is not None:
            ord_targets = cumulative_ordinal_targets(labels, num_classes=outputs["class_logits"].size(-1))
            ord_loss = self.bce(outputs["ordinal_logits"], ord_targets)
            total = total + self.lambda_ordinal * ord_loss
            out["ordinal"] = ord_loss
        else:
            out["ordinal"] = torch.tensor(0.0, device=labels.device)

        out["loss"] = total
        return out
