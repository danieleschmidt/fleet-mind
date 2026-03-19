"""CollectiveIntelligence — shared knowledge base for fleet-wide situational awareness."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Observation:
    """A piece of information observed by an agent and reported to the fleet."""
    agent_id: str
    obs_type: str          # e.g. "hazard", "task_complete", "waypoint_clear", "target_found"
    location: Tuple[float, float]
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0–1

    def age(self) -> float:
        """Seconds since this observation was recorded."""
        return time.time() - self.timestamp

    def is_stale(self, ttl: float = 120.0) -> bool:
        return self.age() > ttl


class CollectiveIntelligence:
    """
    Fleet-wide shared knowledge base.

    Agents report observations; the coordinator synthesizes them into:
    - Known hazard map
    - Cleared / scouted waypoints
    - Active targets / points of interest
    - Fleet-wide task completion history
    - Summary situational report

    All data is time-bounded (stale observations are pruned automatically).
    """

    DEFAULT_TTL = 180.0  # 3 minutes

    def __init__(self, ttl: float = DEFAULT_TTL):
        self.ttl = ttl
        self._observations: List[Observation] = []
        self._hazards: Dict[Tuple[float, float], List[Observation]] = defaultdict(list)
        self._cleared: Dict[Tuple[float, float], Observation] = {}
        self._targets: Dict[str, Observation] = {}
        self._task_history: List[dict] = []
        self._agent_last_seen: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, observation: Observation) -> None:
        """Accept a new observation from any agent."""
        self._observations.append(observation)
        self._agent_last_seen[observation.agent_id] = observation.timestamp

        if observation.obs_type == "hazard":
            self._hazards[observation.location].append(observation)
        elif observation.obs_type == "waypoint_clear":
            self._cleared[observation.location] = observation
        elif observation.obs_type == "target_found":
            target_id = observation.payload.get("target_id", str(observation.location))
            self._targets[target_id] = observation
        elif observation.obs_type == "task_complete":
            self._task_history.append({
                "agent_id": observation.agent_id,
                "task_id": observation.payload.get("task_id"),
                "location": observation.location,
                "timestamp": observation.timestamp,
            })

    def ingest_batch(self, observations: List[dict]) -> None:
        """Ingest raw dict observations (as emitted by FleetAgent.drain_observations)."""
        for raw in observations:
            obs = Observation(
                agent_id=raw.get("agent_id", "unknown"),
                obs_type=raw.get("type", "generic"),
                location=raw.get("location", (0.0, 0.0)),
                payload=raw,
                timestamp=raw.get("timestamp", time.time()),
            )
            self.ingest(obs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def hazards_near(self, location: Tuple[float, float], radius: float = 10.0) -> List[Observation]:
        """Return non-stale hazard observations within radius of location."""
        import math
        result = []
        for (hx, hy), obs_list in self._hazards.items():
            dist = math.sqrt((hx - location[0]) ** 2 + (hy - location[1]) ** 2)
            if dist <= radius:
                fresh = [o for o in obs_list if not o.is_stale(self.ttl)]
                result.extend(fresh)
        return result

    def is_cleared(self, location: Tuple[float, float]) -> bool:
        """Return True if a waypoint has been marked clear and observation is fresh."""
        obs = self._cleared.get(location)
        return obs is not None and not obs.is_stale(self.ttl)

    def active_targets(self) -> Dict[str, Observation]:
        """Return targets that haven't gone stale."""
        return {tid: obs for tid, obs in self._targets.items() if not obs.is_stale(self.ttl)}

    def agent_status(self) -> Dict[str, float]:
        """Return last-seen timestamp per agent."""
        return dict(self._agent_last_seen)

    def task_completion_count(self) -> int:
        return len(self._task_history)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """Remove stale observations. Returns count pruned."""
        before = len(self._observations)
        self._observations = [o for o in self._observations if not o.is_stale(self.ttl)]
        self._hazards = defaultdict(list, {
            k: [o for o in v if not o.is_stale(self.ttl)]
            for k, v in self._hazards.items()
            if any(not o.is_stale(self.ttl) for o in v)
        })
        self._targets = {
            k: v for k, v in self._targets.items() if not v.is_stale(self.ttl)
        }
        return before - len(self._observations)

    def situational_report(self) -> str:
        """Generate a human-readable fleet-wide situation summary."""
        self.prune()
        hazard_count = sum(len(v) for v in self._hazards.values())
        target_count = len(self.active_targets())
        cleared_count = sum(1 for v in self._cleared.values() if not v.is_stale(self.ttl))
        agents_seen = len(self._agent_last_seen)
        completed = self.task_completion_count()

        lines = [
            "=== Fleet Situational Report ===",
            f"  Agents reporting:      {agents_seen}",
            f"  Tasks completed:       {completed}",
            f"  Active hazards:        {hazard_count}",
            f"  Active targets/POIs:   {target_count}",
            f"  Cleared waypoints:     {cleared_count}",
            f"  Total observations:    {len(self._observations)}",
        ]

        if target_count > 0:
            lines.append("  Active Targets:")
            for tid, obs in self.active_targets().items():
                lines.append(f"    [{tid}] at {obs.location} (age {obs.age():.0f}s)")

        return "\n".join(lines)
