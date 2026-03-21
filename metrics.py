from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode / decision rollout."""

    used_fallback: bool
    selected_step: int
    action: int
    output_spike_count: int
    hidden_spike_count: int
    pulses_plus: int
    pulses_minus: int
    n_refresh: int
    winner: int
    target: int
    reward: float


@dataclass
class MetricsSummary:
    num_episodes: int
    fallback_rate: float
    mean_decision_step: float
    std_decision_step: float
    mean_output_spikes: float
    mean_hidden_spikes: float
    mean_pulses_plus: float
    mean_pulses_minus: float
    mean_refreshes: float
    action_histogram: Dict[int, int] = field(default_factory=dict)


class SNNMetrics:
    """
    Lightweight metrics collector for the current recurrent memristive SNN.

    Expected rollout_info format (from network debug / decision result):
        {
            "used_fallback": bool,
            "selected_step": int,
            "action": int,
            "hidden_spikes": List[List[int]] | np.ndarray,
            "output_spikes": List[List[int]] | np.ndarray,
        }

    Expected learning_event:
        object with attributes
            n_pulses_plus
            n_pulses_minus
            n_refresh
            reward
            winner
            target
        or None
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.episodes: List[EpisodeMetrics] = []

    @staticmethod
    def _count_spikes(spikes: Any) -> int:
        if spikes is None:
            return 0
        arr = np.asarray(spikes, dtype=int)
        if arr.size == 0:
            return 0
        return int(arr.sum())

    def add_episode(
        self,
        rollout_info: Dict[str, Any],
        learning_event: Optional[Any] = None,
    ) -> EpisodeMetrics:
        used_fallback = bool(rollout_info.get("used_fallback", False))
        selected_step = int(rollout_info.get("selected_step", -1))
        action = int(rollout_info.get("action", -1))

        hidden_spike_count = self._count_spikes(rollout_info.get("hidden_spikes"))
        output_spike_count = self._count_spikes(rollout_info.get("output_spikes"))

        if learning_event is None:
            pulses_plus = 0
            pulses_minus = 0
            n_refresh = 0
            reward = 0.0
            winner = -1
            target = -1
        else:
            pulses_plus = int(getattr(learning_event, "n_pulses_plus", 0))
            pulses_minus = int(getattr(learning_event, "n_pulses_minus", 0))
            n_refresh = int(getattr(learning_event, "n_refresh", 0))
            reward = float(getattr(learning_event, "reward", 0.0))
            winner_raw = getattr(learning_event, "winner", -1)
            target_raw = getattr(learning_event, "target", -1)

            winner = -1 if winner_raw is None else int(winner_raw)
            target = -1 if target_raw is None else int(target_raw)

        ep = EpisodeMetrics(
            used_fallback=used_fallback,
            selected_step=selected_step,
            action=action,
            output_spike_count=output_spike_count,
            hidden_spike_count=hidden_spike_count,
            pulses_plus=pulses_plus,
            pulses_minus=pulses_minus,
            n_refresh=n_refresh,
            winner=winner,
            target=target,
            reward=reward,
        )
        self.episodes.append(ep)
        return ep

    def summary(self) -> MetricsSummary:
        if not self.episodes:
            return MetricsSummary(
                num_episodes=0,
                fallback_rate=0.0,
                mean_decision_step=0.0,
                std_decision_step=0.0,
                mean_output_spikes=0.0,
                mean_hidden_spikes=0.0,
                mean_pulses_plus=0.0,
                mean_pulses_minus=0.0,
                mean_refreshes=0.0,
                action_histogram={},
            )

        decision_steps = np.array([ep.selected_step for ep in self.episodes], dtype=float)
        fallback = np.array([ep.used_fallback for ep in self.episodes], dtype=float)
        output_spikes = np.array([ep.output_spike_count for ep in self.episodes], dtype=float)
        hidden_spikes = np.array([ep.hidden_spike_count for ep in self.episodes], dtype=float)
        pulses_plus = np.array([ep.pulses_plus for ep in self.episodes], dtype=float)
        pulses_minus = np.array([ep.pulses_minus for ep in self.episodes], dtype=float)
        refreshes = np.array([ep.n_refresh for ep in self.episodes], dtype=float)

        hist: Dict[int, int] = {}
        for ep in self.episodes:
            hist[ep.action] = hist.get(ep.action, 0) + 1

        return MetricsSummary(
            num_episodes=len(self.episodes),
            fallback_rate=float(fallback.mean()),
            mean_decision_step=float(decision_steps.mean()),
            std_decision_step=float(decision_steps.std()),
            mean_output_spikes=float(output_spikes.mean()),
            mean_hidden_spikes=float(hidden_spikes.mean()),
            mean_pulses_plus=float(pulses_plus.mean()),
            mean_pulses_minus=float(pulses_minus.mean()),
            mean_refreshes=float(refreshes.mean()),
            action_histogram=hist,
        )

    def summary_dict(self) -> Dict[str, Any]:
        s = self.summary()
        return {
            "num_episodes": s.num_episodes,
            "fallback_rate": s.fallback_rate,
            "mean_decision_step": s.mean_decision_step,
            "std_decision_step": s.std_decision_step,
            "mean_output_spikes": s.mean_output_spikes,
            "mean_hidden_spikes": s.mean_hidden_spikes,
            "mean_pulses_plus": s.mean_pulses_plus,
            "mean_pulses_minus": s.mean_pulses_minus,
            "mean_refreshes": s.mean_refreshes,
            "action_histogram": s.action_histogram,
        }


if __name__ == "__main__":
    # Small self-check only. Safe to remove later.
    metrics = SNNMetrics()

    rollout_info = {
        "used_fallback": False,
        "selected_step": 2,
        "action": 1,
        "hidden_spikes": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        "output_spikes": [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
    }

    class DummyLearning:
        n_pulses_plus = 1
        n_pulses_minus = 0
        n_refresh = 0
        reward = 1.0
        winner = 1
        target = 1

    metrics.add_episode(rollout_info, DummyLearning())
    print(metrics.summary_dict())
