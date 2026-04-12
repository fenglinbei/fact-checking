from __future__ import annotations

import torch



def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax over valid entries only.

    Args:
        logits: Tensor of arbitrary shape.
        mask: Boolean tensor broadcastable to logits. True means valid.
        dim: dimension over which to apply sparsemax.
    """
    logits = logits.masked_fill(~mask, float("-inf"))

    # When all positions are invalid, return zeros.
    all_invalid = (~mask).all(dim=dim, keepdim=True)
    safe_logits = logits.masked_fill(all_invalid, 0.0)

    shifted = safe_logits - safe_logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(shifted, dim=dim, descending=True).values
    z_cumsum = zs.cumsum(dim) - 1
    range_values = torch.arange(1, zs.size(dim) + 1, device=zs.device, dtype=zs.dtype)
    view_shape = [1] * zs.dim()
    view_shape[dim] = -1
    range_values = range_values.view(*view_shape)
    support = range_values * zs > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = z_cumsum.gather(dim, k - 1) / k.to(zs.dtype)
    output = torch.clamp(shifted - tau, min=0.0)
    output = output.masked_fill(~mask, 0.0)
    output = output.masked_fill(all_invalid, 0.0)
    return output
