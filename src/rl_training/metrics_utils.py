from __future__ import annotations

from collections import Counter
from typing import Iterable

import art


def _flatten_trajectories(
    train_groups: Iterable[art.TrajectoryGroup],
) -> list[art.Trajectory]:
    """Flatten a list of trajectory groups into a list of trajectories."""
    trajectories: list[art.Trajectory] = []
    for group in train_groups:
        trajectories.extend(group.trajectories)
    return trajectories


def compute_role_based_metrics(
    train_groups: Iterable[art.TrajectoryGroup],
) -> dict[str, float]:
    """
    Aggregate win counts and win rates grouped by the trainable agent's role.

    Returns:
        Dict of metrics that can be logged directly to W&B.
    """
    metrics: dict[str, float] = {}
    trajectories = _flatten_trajectories(train_groups)
    if not trajectories:
        return metrics

    role_counter = Counter(
        (t.metadata.get("trainable_role") or "unknown") for t in trajectories
    )
    total = len(trajectories)
    for role, count in role_counter.items():
        metrics[f"role/{role}/count"] = float(count)
        metrics[f"role/{role}/fraction"] = count / total
        wins = sum(1 for t in trajectories if (t.metadata.get("trainable_role") or "unknown") == role and t.reward > 0)
        metrics[f"role/{role}/win_rate"] = wins / count if count else 0.0

    winning_team_counter = Counter(
        (t.metadata.get("winning_team") or "unknown") for t in trajectories
    )
    for team, count in winning_team_counter.items():
        metrics[f"winning_team/{team}/count"] = float(count)
        metrics[f"winning_team/{team}/fraction"] = count / total

    return metrics


def compute_oversampling_role_metrics(
    train_groups: Iterable[art.TrajectoryGroup],
) -> dict[str, float]:
    """
    Measure how often the trainable policy starts on the impostor team.
    """
    metrics: dict[str, float] = {}
    trajectories = _flatten_trajectories(train_groups)
    if not trajectories:
        return metrics

    impostor_starts = sum(
        int(t.metrics.get("trainable_impostor_start", 0)) for t in trajectories
    )
    metrics["oversample/impostor_starts"] = float(impostor_starts)
    metrics["oversample/total_games"] = float(len(trajectories))
    metrics["oversample/impostor_ratio"] = (
        impostor_starts / len(trajectories) if trajectories else 0.0
    )
    return metrics
