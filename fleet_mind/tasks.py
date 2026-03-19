"""TaskQueue — priority queue of tasks with requirements, deadlines, and locations."""

from __future__ import annotations

import heapq
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum, Enum, auto
from typing import Optional, Set

from .agent import Position


class TaskPriority(IntEnum):
    """Lower number = higher priority (heapq is a min-heap)."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class TaskStatus(Enum):
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    EXPIRED = auto()
    FAILED = auto()


@dataclass
class Task:
    """
    A work unit that can be assigned to a FleetAgent.

    Attributes:
        task_id:            Unique task identifier.
        task_type:          Category string (e.g. "deliver", "scan", "rescue").
        location:           Target 2D position.
        priority:           TaskPriority value.
        deadline:           Unix timestamp by which the task must start.
        requirements:       Set of capability strings the agent must have.
        execution_time:     Timesteps needed to complete once on-site.
        description:        Human-readable description.
    """

    task_id: str
    task_type: str
    location: Position
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: float = field(default_factory=lambda: time.time() + 300)
    requirements: Set[str] = field(default_factory=set)
    execution_time: int = 3
    description: str = ""

    # Runtime fields
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    assigned_to: Optional[str] = field(default=None, init=False)
    created_at: float = field(default_factory=time.time, init=False)
    completed_at: Optional[float] = field(default=None, init=False)
    execution_remaining: int = field(default=0, init=False)

    def __post_init__(self):
        self.execution_remaining = self.execution_time

    # Comparison for heapq: sort by (priority, deadline, task_id)
    def __lt__(self, other: "Task") -> bool:
        return (self.priority, self.deadline, self.task_id) < (
            other.priority, other.deadline, other.task_id
        )

    def assign(self, agent_id: str) -> None:
        self.assigned_to = agent_id
        self.status = TaskStatus.ASSIGNED

    def start_execution(self) -> None:
        self.status = TaskStatus.IN_PROGRESS

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.completed_at = time.time()

    def is_expired(self) -> bool:
        return time.time() > self.deadline and self.status == TaskStatus.PENDING

    def time_remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def __str__(self) -> str:
        return (
            f"Task({self.task_id} type={self.task_type} "
            f"loc={self.location} prio={self.priority.name} status={self.status.name})"
        )


class TaskQueue:
    """
    Thread-safe priority queue for fleet tasks.

    Uses a min-heap keyed on (priority, deadline). Supports:
    - add / push
    - pop_next (next highest-priority pending task)
    - peek
    - expire (remove overdue tasks)
    - stats
    """

    def __init__(self):
        self._heap: list = []
        self._all: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        """Add a task to the queue."""
        self._all[task.task_id] = task
        heapq.heappush(self._heap, task)

    def push(self, task: Task) -> None:
        """Alias for add."""
        self.add(task)

    def pop_next(self) -> Optional[Task]:
        """
        Return the highest-priority pending task, or None if empty.
        Skips non-pending tasks (assigned, completed, etc.).
        """
        while self._heap:
            task = heapq.heappop(self._heap)
            if task.status == TaskStatus.PENDING and not task.is_expired():
                return task
            if task.is_expired():
                task.status = TaskStatus.EXPIRED
        return None

    def peek(self) -> Optional[Task]:
        """Look at the next task without removing it."""
        for task in sorted(self._heap):
            if task.status == TaskStatus.PENDING and not task.is_expired():
                return task
        return None

    def expire(self) -> list[Task]:
        """Mark and return all expired tasks."""
        expired = []
        for task in self._all.values():
            if task.is_expired():
                task.status = TaskStatus.EXPIRED
                expired.append(task)
        return expired

    def pending_count(self) -> int:
        return sum(1 for t in self._all.values() if t.status == TaskStatus.PENDING)

    def stats(self) -> dict:
        counts = {}
        for status in TaskStatus:
            counts[status.name] = sum(1 for t in self._all.values() if t.status == status)
        return counts

    def all_tasks(self) -> list[Task]:
        return list(self._all.values())

    def __len__(self) -> int:
        return self.pending_count()

    def __repr__(self) -> str:
        return f"TaskQueue(pending={self.pending_count()}, total={len(self._all)})"
