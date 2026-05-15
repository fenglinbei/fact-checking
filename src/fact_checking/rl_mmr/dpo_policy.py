"""DPO policy model and loss for step-wise λ selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LAMBDA_GRID_DEFAULT: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]


class StepLambdaPolicy(nn.Module):
    """MLP policy: state features → logits over discrete λ actions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        n_actions: int = 5,
    ):
        super().__init__()
        dims = hidden_dims or [64, 32]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.n_actions = n_actions

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """state_features [B, D] → logits [B, n_actions]."""
        return self.net(state_features)

    def log_prob(self, state_features: torch.Tensor, lambda_idx: torch.Tensor) -> torch.Tensor:
        """Compute log π(λ_idx | state_features) → [B]."""
        logits = self.forward(state_features)
        return F.log_softmax(logits, dim=-1).gather(1, lambda_idx.unsqueeze(-1)).squeeze(-1)


class FixedReferencePolicy(nn.Module):
    """Reference policy centered at a specific λ value.

    π_ref(λ) ∝ exp(-|λ - center| / temperature), applied as fixed logits.
    """

    def __init__(self, lambda_grid: list[float] | None = None, center: float = 0.7, temperature: float = 0.3):
        super().__init__()
        grid = lambda_grid or LAMBDA_GRID_DEFAULT
        weights = np.array([-abs(lam - center) / temperature for lam in grid], dtype=np.float32)
        self.register_buffer("fixed_logits", torch.from_numpy(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fixed logits for any batch of states."""
        return self.fixed_logits.unsqueeze(0).expand(x.shape[0], -1)


def dpo_loss(
    policy: StepLambdaPolicy,
    ref_policy: FixedReferencePolicy,
    win_features: torch.Tensor,     # [B, K, D]
    win_lambdas: torch.Tensor,      # [B, K]  λ indices
    lose_features: torch.Tensor,    # [B, K, D]
    lose_lambdas: torch.Tensor,     # [B, K]  λ indices
    beta: float = 1.0,
) -> torch.Tensor:
    """DPO loss for step-wise trajectory preference.

    L = -log σ(β * [(log π_θ(τ⁺) - log π_θ(τ⁻)) - (log π_ref(τ⁺) - log π_ref(τ⁻))])

    where log π(τ) = Σ_{t=1}^{K} log π(λ_t | s_t)
    """
    B, K, D = win_features.shape

    # Flatten steps
    wf = win_features.reshape(B * K, D)
    lf = lose_features.reshape(B * K, D)
    wl = win_lambdas.reshape(B * K)
    ll = lose_lambdas.reshape(B * K)

    # Per-step log probs → [B * K]
    logp_win = policy.log_prob(wf, wl)
    logp_lose = policy.log_prob(lf, ll)

    with torch.no_grad():
        logp_win_ref = ref_policy.log_prob(wf, wl) if hasattr(ref_policy, 'log_prob') else _ref_log_prob(ref_policy, wf, wl)
        logp_lose_ref = ref_policy.log_prob(lf, ll) if hasattr(ref_policy, 'log_prob') else _ref_log_prob(ref_policy, lf, ll)

    # Sum over steps → [B]
    logp_win_sum = logp_win.reshape(B, K).sum(dim=-1)
    logp_lose_sum = logp_lose.reshape(B, K).sum(dim=-1)
    logp_win_ref_sum = logp_win_ref.reshape(B, K).sum(dim=-1)
    logp_lose_ref_sum = logp_lose_ref.reshape(B, K).sum(dim=-1)

    log_ratio = (logp_win_sum - logp_lose_sum) - (logp_win_ref_sum - logp_lose_ref_sum)
    loss = -F.logsigmoid(beta * log_ratio).mean()
    return loss


def _ref_log_prob(ref_policy: FixedReferencePolicy, features: torch.Tensor, lambda_idx: torch.Tensor) -> torch.Tensor:
    logits = ref_policy(features)
    return F.log_softmax(logits, dim=-1).gather(1, lambda_idx.unsqueeze(-1)).squeeze(-1)


def policy_entropy(policy: StepLambdaPolicy, features: torch.Tensor) -> torch.Tensor:
    """Compute policy entropy H(π(·|s)) for each state → [B]."""
    logits = policy(features)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def argmax_distribution(
    policy: StepLambdaPolicy,
    features: torch.Tensor,
    n_actions: int,
) -> np.ndarray:
    """Return argmax counts per action → [n_actions]."""
    policy.eval()
    with torch.no_grad():
        logits = policy(features)
        argmax = torch.argmax(logits, dim=-1).cpu().numpy()
    counts = np.bincount(argmax, minlength=n_actions)
    return counts.astype(np.float32)


def evaluate_policy_metrics(
    policy: StepLambdaPolicy,
    ref_policy: FixedReferencePolicy,
    win_features: torch.Tensor,
    win_lambdas: torch.Tensor,
    lose_features: torch.Tensor,
    lose_lambdas: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, Any]:
    """Compute evaluation metrics for a batch of preference pairs."""
    policy.eval()
    with torch.no_grad():
        loss = dpo_loss(
            policy, ref_policy,
            win_features, win_lambdas, lose_features, lose_lambdas, beta=beta,
        )

        # Accuracy: policy assigns higher probability to winner
        B, K, D = win_features.shape
        wf = win_features.reshape(B * K, D)
        lf = lose_features.reshape(B * K, D)
        wl = win_lambdas.reshape(B * K)
        ll = lose_lambdas.reshape(B * K)

        logp_win = policy.log_prob(wf, wl).reshape(B, K).sum(dim=-1)
        logp_lose = policy.log_prob(lf, ll).reshape(B, K).sum(dim=-1)
        accuracy = (logp_win > logp_lose).float().mean().item()

        # Policy entropy
        all_features = torch.cat([wf, lf], dim=0)
        entropy = policy_entropy(policy, all_features).mean().item()

    return {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "entropy": entropy,
    }
