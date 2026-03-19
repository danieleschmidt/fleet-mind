"""Fleet-Mind: Collective Intelligence for Autonomous Agent Fleets.

A pure-Python fleet coordination system featuring:
- Auction-based task allocation
- Distributed situational awareness
- Priority-driven task queuing
- Mission simulation for patrol, delivery, and search-and-rescue scenarios
"""

from .agent import FleetAgent, AgentState
from .coordinator import FleetCoordinator
from .tasks import TaskQueue, Task, TaskPriority, TaskStatus
from .intelligence import CollectiveIntelligence, Observation
from .simulator import FleetSimulator, MissionType

__version__ = "1.0.0"
__all__ = [
    "FleetAgent",
    "AgentState",
    "FleetCoordinator",
    "TaskQueue",
    "Task",
    "TaskPriority",
    "TaskStatus",
    "CollectiveIntelligence",
    "Observation",
    "FleetSimulator",
    "MissionType",
]
