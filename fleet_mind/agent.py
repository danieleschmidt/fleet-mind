"""FleetAgent — represents one autonomous agent in the fleet."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Set, List, Tuple


class AgentState(Enum):
    IDLE = auto()
    MOVING = auto()
    EXECUTING = auto()
    RETURNING = auto()
    OFFLINE = auto()


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def move_toward(self, target: "Position", speed: float) -> "Position":
        dist = self.distance_to(target)
        if dist <= speed:
            return Position(target.x, target.y)
        ratio = speed / dist
        return Position(
            self.x + (target.x - self.x) * ratio,
            self.y + (target.y - self.y) * ratio,
        )

    def __str__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class FleetAgent:
    """
    Represents one agent (drone, vehicle, robot) in the fleet.

    Attributes:
        agent_id:     Unique identifier.
        position:     Current 2D position.
        capabilities: Set of task types this agent can perform.
        speed:        Units per timestep.
        state:        Current operational state.
        current_task: Task the agent is currently working on (or None).
        load:         Number of tasks assigned (including current).
        battery:      0–100 % charge level.
        completed:    Count of successfully completed tasks.
        observations: Local observations reported to collective intelligence.
    """

    agent_id: str
    position: Position
    capabilities: Set[str]
    speed: float = 1.0
    state: AgentState = AgentState.IDLE
    current_task: Optional[object] = field(default=None, repr=False)
    load: int = 0
    battery: float = 100.0
    completed: int = 0
    observations: List[dict] = field(default_factory=list, repr=False)

    # Task backlog: tasks queued but not yet being executed
    task_backlog: List[object] = field(default_factory=list, repr=False)

    # Home base for returning
    home: Optional[Position] = field(default=None, repr=False)

    def __post_init__(self):
        if self.home is None:
            self.home = Position(self.position.x, self.position.y)

    # ------------------------------------------------------------------
    # Auction mechanics
    # ------------------------------------------------------------------

    def bid(self, task) -> Optional[float]:
        """
        Compute a bid (lower = more eager) for a task.

        Bid = distance_to_task + load_penalty + capability_bonus
        Returns None if this agent cannot perform the task.
        """
        if task.requirements and not task.requirements.issubset(self.capabilities):
            return None  # can't do it

        if self.battery < 20:
            return None  # not enough battery

        distance = self.position.distance_to(task.location)
        load_penalty = self.load * 5.0  # each pending task adds 5 units of cost
        urgency_discount = max(0.0, (task.deadline - time.time()) / 60.0)  # seconds remaining (capped)
        urgency_discount = min(urgency_discount, 30.0)  # cap at 30s advantage

        return distance + load_penalty - urgency_discount

    def assign_task(self, task) -> None:
        """Accept a task assignment (queued if agent is already busy)."""
        self.load += 1
        task.assign(self.agent_id)
        if self.current_task is None:
            self.current_task = task
            self.state = AgentState.MOVING
        else:
            self.task_backlog.append(task)

    def step(self) -> Optional[str]:
        """
        Advance the agent one simulation timestep.

        Returns a completion event string if a task is finished, else None.
        """
        if self.state == AgentState.OFFLINE:
            return None

        # Passive battery drain
        self.battery = max(0.0, self.battery - 0.2)

        if self.current_task is None:
            self.state = AgentState.IDLE
            return None

        task = self.current_task

        # Phase 1: Move to task location
        if self.state == AgentState.MOVING:
            self.position = self.position.move_toward(task.location, self.speed)
            if self.position.distance_to(task.location) < 0.1:
                self.state = AgentState.EXECUTING
                task.start_execution()

        # Phase 2: Execute task (takes execution_time steps)
        elif self.state == AgentState.EXECUTING:
            task.execution_remaining -= 1
            if task.execution_remaining <= 0:
                self._complete_task()
                return f"Agent {self.agent_id} completed task {task.task_id}"

        return None

    def _complete_task(self) -> None:
        task = self.current_task
        task.complete()
        self.completed += 1
        self.load = max(0, self.load - 1)

        # Record observation
        self.observations.append({
            "agent_id": self.agent_id,
            "task_id": task.task_id,
            "location": (task.location.x, task.location.y),
            "timestamp": time.time(),
            "type": "task_complete",
        })

        # Pick up next queued task, if any
        if self.task_backlog:
            next_task = self.task_backlog.pop(0)
            self.current_task = next_task
            self.state = AgentState.MOVING
        else:
            self.current_task = None
            self.state = AgentState.IDLE

    def observe(self, observation: dict) -> None:
        """Record a local observation (hazard, point of interest, etc.)."""
        observation["agent_id"] = self.agent_id
        observation["timestamp"] = time.time()
        self.observations.append(observation)

    def drain_observations(self) -> List[dict]:
        """Return and clear pending observations."""
        obs = list(self.observations)
        self.observations.clear()
        return obs

    def status_line(self) -> str:
        task_info = f"task={self.current_task.task_id}" if self.current_task else "idle"
        return (
            f"[{self.agent_id}] pos={self.position} state={self.state.name} "
            f"battery={self.battery:.0f}% load={self.load} done={self.completed} {task_info}"
        )
