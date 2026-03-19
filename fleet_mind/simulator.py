"""FleetSimulator — runs simulated fleet missions."""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from .agent import FleetAgent, Position
from .coordinator import FleetCoordinator
from .intelligence import CollectiveIntelligence
from .tasks import Task, TaskPriority, TaskQueue


class MissionType(Enum):
    COVERAGE_PATROL = auto()   # Agents sweep a grid area
    DELIVERY = auto()          # Deliver packages to destinations
    SEARCH_AND_RESCUE = auto() # Locate and extract targets


@dataclass
class MissionResult:
    mission_type: MissionType
    total_tasks: int
    completed_tasks: int
    expired_tasks: int
    total_ticks: int
    agent_completions: dict  # agent_id -> count

    @property
    def completion_rate(self) -> float:
        return self.completed_tasks / self.total_tasks if self.total_tasks else 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*50}",
            f"Mission Complete: {self.mission_type.name}",
            f"{'='*50}",
            f"  Tasks total:     {self.total_tasks}",
            f"  Tasks completed: {self.completed_tasks}",
            f"  Tasks expired:   {self.expired_tasks}",
            f"  Completion rate: {self.completion_rate:.0%}",
            f"  Ticks elapsed:   {self.total_ticks}",
            f"  Agent breakdown:",
        ]
        for agent_id, count in sorted(self.agent_completions.items()):
            lines.append(f"    {agent_id}: {count} task(s)")
        lines.append(f"{'='*50}")
        return "\n".join(lines)


class FleetSimulator:
    """
    Runs end-to-end fleet missions.

    Usage:
        sim = FleetSimulator(seed=42)
        result = sim.run(mission_type=MissionType.DELIVERY, num_agents=5, num_tasks=10)
        print(result.summary())
    """

    def __init__(self, seed: Optional[int] = None, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def run(
        self,
        mission_type: MissionType = MissionType.DELIVERY,
        num_agents: int = 5,
        num_tasks: int = 10,
        arena_size: float = 50.0,
        max_ticks: int = 500,
        auction_interval: int = 5,
    ) -> MissionResult:
        """
        Build a mission, run it to completion or max_ticks, return results.
        """
        intel = CollectiveIntelligence()
        coordinator = FleetCoordinator(intel)
        task_queue = TaskQueue()

        # Create agents
        agents = self._build_agents(num_agents, arena_size, mission_type)
        for agent in agents:
            coordinator.add_agent(agent)

        # Create tasks
        tasks = self._build_tasks(mission_type, num_tasks, arena_size)
        for task in tasks:
            task_queue.add(task)

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Mission: {mission_type.name}")
            print(f"Agents: {num_agents}  Tasks: {num_tasks}  Arena: {arena_size}x{arena_size}")
            print(f"{'='*50}")
            for agent in agents:
                caps = ", ".join(sorted(agent.capabilities))
                print(f"  {agent.agent_id} at {agent.position}  caps=[{caps}]")
            print()

        # Simulation loop
        tick = 0
        while tick < max_ticks:
            # Auction: assign pending tasks periodically
            if tick % auction_interval == 0 and len(task_queue) > 0:
                auction_result = coordinator.run_auction(task_queue, max_tasks=num_agents)
                if self.verbose and auction_result.assignments:
                    print(f"[Tick {tick:03d}] Auction round:")
                    print(auction_result.summary())

            # Step all agents
            events = coordinator.step()
            if self.verbose:
                for event in events:
                    print(f"[Tick {tick:03d}] ✓ {event}")

            tick += 1

            # Check mission complete
            if coordinator.all_idle() and len(task_queue) == 0:
                break

        # Expire any remaining tasks
        expired = task_queue.expire()

        if self.verbose:
            print()
            print(coordinator.fleet_status())
            print()
            print(intel.situational_report())

        return MissionResult(
            mission_type=mission_type,
            total_tasks=num_tasks,
            completed_tasks=coordinator.completion_count(),
            expired_tasks=len(expired),
            total_ticks=tick,
            agent_completions={a.agent_id: a.completed for a in agents},
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_agents(
        self, num_agents: int, arena_size: float, mission_type: MissionType
    ) -> List[FleetAgent]:
        """Spawn agents with capabilities appropriate for the mission."""
        base_caps_pool = {
            MissionType.DELIVERY: ["carry", "navigate"],
            MissionType.COVERAGE_PATROL: ["scan", "navigate"],
            MissionType.SEARCH_AND_RESCUE: ["rescue", "scan", "navigate", "medical"],
        }
        caps_pool = base_caps_pool[mission_type]

        agents = []
        for i in range(num_agents):
            # Agents start clustered near origin (home base)
            sx = self._rng.uniform(0, arena_size * 0.15)
            sy = self._rng.uniform(0, arena_size * 0.15)

            # Vary capabilities a bit
            if mission_type == MissionType.SEARCH_AND_RESCUE:
                # Mix of rescue specialists and scouts
                if i % 3 == 0:
                    caps = {"rescue", "medical", "navigate"}
                else:
                    caps = {"scan", "navigate"}
            else:
                caps = set(caps_pool)

            agent = FleetAgent(
                agent_id=f"A{i+1:02d}",
                position=Position(sx, sy),
                capabilities=caps,
                speed=self._rng.uniform(1.5, 3.0),
            )
            agents.append(agent)
        return agents

    def _build_tasks(
        self, mission_type: MissionType, num_tasks: int, arena_size: float
    ) -> List[Task]:
        """Generate tasks appropriate for the mission type."""
        tasks = []
        now = time.time()

        for i in range(num_tasks):
            tx = self._rng.uniform(arena_size * 0.1, arena_size * 0.9)
            ty = self._rng.uniform(arena_size * 0.1, arena_size * 0.9)

            # Vary priority
            prio_weights = [
                (TaskPriority.CRITICAL, 0.1),
                (TaskPriority.HIGH, 0.25),
                (TaskPriority.NORMAL, 0.5),
                (TaskPriority.LOW, 0.15),
            ]
            prio = self._rng.choices(
                [p for p, _ in prio_weights],
                weights=[w for _, w in prio_weights],
            )[0]

            if mission_type == MissionType.DELIVERY:
                task = Task(
                    task_id=f"D{i+1:03d}",
                    task_type="deliver",
                    location=Position(tx, ty),
                    priority=prio,
                    deadline=now + self._rng.uniform(120, 600),
                    requirements={"carry", "navigate"},
                    execution_time=self._rng.randint(2, 5),
                    description=f"Deliver package #{i+1}",
                )

            elif mission_type == MissionType.COVERAGE_PATROL:
                # Grid-based waypoints
                cols = math.ceil(math.sqrt(num_tasks))
                row, col = divmod(i, cols)
                step = arena_size / (cols + 1)
                tx = step * (col + 1)
                ty = step * (row + 1)
                task = Task(
                    task_id=f"P{i+1:03d}",
                    task_type="patrol",
                    location=Position(tx, ty),
                    priority=TaskPriority.NORMAL,
                    deadline=now + 900,
                    requirements={"scan", "navigate"},
                    execution_time=2,
                    description=f"Patrol waypoint {i+1}",
                )

            elif mission_type == MissionType.SEARCH_AND_RESCUE:
                # Mix of scout (scan) and extraction (rescue) tasks
                if i % 4 == 0:
                    task = Task(
                        task_id=f"R{i+1:03d}",
                        task_type="rescue",
                        location=Position(tx, ty),
                        priority=TaskPriority.HIGH,
                        deadline=now + self._rng.uniform(180, 480),
                        requirements={"rescue", "navigate"},
                        execution_time=self._rng.randint(4, 8),
                        description=f"Extract survivor #{i+1}",
                    )
                else:
                    task = Task(
                        task_id=f"S{i+1:03d}",
                        task_type="scout",
                        location=Position(tx, ty),
                        priority=TaskPriority.NORMAL,
                        deadline=now + 600,
                        requirements={"scan", "navigate"},
                        execution_time=2,
                        description=f"Scout sector {i+1}",
                    )
            else:
                raise ValueError(f"Unknown mission type: {mission_type}")

            tasks.append(task)

        return tasks
