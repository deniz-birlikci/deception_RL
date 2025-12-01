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


def compute_emdash_metrics(
    train_groups: Iterable[art.TrajectoryGroup],
) -> dict[str, float]:
    """
    Aggregate em dash usage statistics for each agent and the trainable policy.
    """
    metrics: dict[str, float] = {}
    trajectories = _flatten_trajectories(train_groups)
    if not trajectories:
        return metrics

    total_games = len(trajectories)
    aggregated_counts: dict[str, float] = {}
    observed_agents: set[str] = set()
    trainable_total = 0.0
    trainable_games = 0
    non_trainable_total = 0.0
    non_trainable_agents: set[str] = set()

    for trajectory in trajectories:
        emdash_counts = trajectory.metadata.get("emdash_counts") or {}
        trainable_agent_id = trajectory.metadata.get("trainable_agent_id")

        for agent_id, count in emdash_counts.items():
            observed_agents.add(agent_id)
            aggregated_counts[agent_id] = aggregated_counts.get(agent_id, 0.0) + float(count)

        if trainable_agent_id:
            trainable_total += float(emdash_counts.get(trainable_agent_id, 0))
            trainable_games += 1

        for agent_id, count in emdash_counts.items():
            if agent_id != trainable_agent_id:
                non_trainable_total += float(count)
                non_trainable_agents.add(agent_id)

    for agent_id in sorted(observed_agents):
        metrics[f"emdashes/{agent_id}/avg_per_game"] = (
            aggregated_counts.get(agent_id, 0.0) / total_games
        )

    metrics["emdashes/trainable/avg_per_game"] = (
        trainable_total / trainable_games if trainable_games else 0.0
    )
    metrics["emdashes/trainable/total_count"] = trainable_total

    if non_trainable_agents:
        metrics["emdashes/non_trainable/avg_per_agent"] = (
            non_trainable_total / (len(non_trainable_agents) * total_games)
        )

    return metrics
