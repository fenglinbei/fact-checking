"""Trajectory data structures and step-wise MMR runner for DPO training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fact_checking.build.candidates import ChunkMMRSample, compute_hybrid_scores
from fact_checking.retrieval.mmr import maximal_marginal_relevance_stepwise


@dataclass
class MMRStep:
    """Single step of MMR selection."""
    step_idx: int
    lambda_val: float
    selected_idx: int
    hybrid_score: float
    max_sim_to_selected: float
    mmr_score: float


@dataclass
class Trajectory:
    """A complete MMR selection trajectory."""
    event_id: str
    claim: str
    gold_label: str
    steps: list[MMRStep]
    selected_ids: list[int]
    lambda_schedule: list[float]
    schedule_type: str       # "fixed" | "handcrafted" | "random"
    utility: float | None = None
    evidence_set_key: str | None = None  # sorted selected_ids as string
    state_features: list[np.ndarray] | None = None  # per-step feature vectors

    @classmethod
    def from_chunk_sample(
        cls,
        sample: ChunkMMRSample,
        lambda_schedule: list[float],
        schedule_type: str,
        *,
        alpha_dense: float = 0.70,
        alpha_lexical: float = 0.20,
        alpha_bm25: float = 0.10,
        state_features: list[np.ndarray] | None = None,
    ) -> "Trajectory":
        """Run step-wise MMR on a ChunkMMRSample and return a Trajectory."""
        scored = compute_hybrid_scores(sample, alpha_dense, alpha_lexical, alpha_bm25)
        n = int(scored["n"])
        if n == 0:
            return cls(
                event_id=sample.event_id,
                claim=sample.claim,
                gold_label=sample.label,
                steps=[],
                selected_ids=[],
                lambda_schedule=list(lambda_schedule),
                schedule_type=schedule_type,
                state_features=state_features,
            )

        hybrid_scores = scored["hybrid_scores"]
        chunk_emb = scored["chunk_emb"]
        top_k = min(len(lambda_schedule), n)
        effective_schedule = lambda_schedule[:top_k]

        selected_indices, step_records = maximal_marginal_relevance_stepwise(
            query_scores=hybrid_scores,
            sentence_vectors=chunk_emb,
            lambda_weights=effective_schedule,
        )

        steps = [
            MMRStep(
                step_idx=r["step_idx"],
                lambda_val=r["lambda_val"],
                selected_idx=r["selected_idx"],
                hybrid_score=r["hybrid_score"],
                max_sim_to_selected=r["max_sim_to_selected"],
                mmr_score=r["mmr_score"],
            )
            for r in step_records
        ]

        sorted_ids = sorted(selected_indices)
        evidence_set_key = "_".join(str(i) for i in sorted_ids)

        return cls(
            event_id=sample.event_id,
            claim=sample.claim,
            gold_label=sample.label,
            steps=steps,
            selected_ids=selected_indices,
            lambda_schedule=list(effective_schedule),
            schedule_type=schedule_type,
            evidence_set_key=evidence_set_key,
            state_features=state_features,
        )


@dataclass
class PreferencePair:
    """A preference pair for DPO training: τ⁺ ≻ τ⁻."""
    event_id: str
    traj_win: Trajectory
    traj_lose: Trajectory
    utility_gap: float
    evidence_set_diff: bool


# ---------------------------------------------------------------------------
# Hand-crafted λ schedules (from experiment plan §7.4)
# ---------------------------------------------------------------------------

HANDCRAFTED_SCHEDULES: list[list[float]] = [
    [0.7, 0.7, 0.7, 0.7, 0.7],   # fixed baseline
    [0.3, 0.3, 0.3, 0.3, 0.3],   # constant low
    [0.5, 0.5, 0.5, 0.5, 0.5],   # constant mid
    [0.9, 0.7, 0.5, 0.3, 0.3],   # decreasing
    [1.0, 0.7, 0.5, 0.3, 0.1],   # steep decreasing
    [0.5, 0.5, 0.7, 0.7, 0.9],   # increasing
    [0.7, 0.5, 0.3, 0.5, 0.7],   # U-shaped
]

LAMBDA_GRID: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]


def generate_random_schedules(
    n_schedules: int,
    top_k: int = 5,
    lambda_grid: list[float] | None = None,
    seed: int = 42,
) -> list[list[float]]:
    """Generate random λ schedules for trajectory exploration."""
    grid = lambda_grid or LAMBDA_GRID
    rng = np.random.default_rng(seed)
    schedules: list[list[float]] = []
    for _ in range(n_schedules):
        sched = [float(grid[int(rng.integers(0, len(grid)))]) for _ in range(top_k)]
        schedules.append(sched)
    return schedules
