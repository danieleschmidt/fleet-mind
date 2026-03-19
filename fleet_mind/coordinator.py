"""FleetCoordinator — orchestrates the fleet via auction-based task allocation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .agent import FleetAgent, AgentState
from .intelligence import CollectiveIntelligence
from .tasks import Task, TaskQueue, TaskStatus


class AuctionResult:
    """Outcome of a single auction round."""

    def __init__(self):
        self.assignments: List[Tuple[Task, FleetAgent, float]] = []  # (task, agent, bid)
        self.unassigned: List[Task] = []
        self.no_bidders: List[Task] = []

    def summary(self) -> str:
        lines = [f"  Assigned:    {len(self.assignments)} task(s)"]
        for task, agent, bid in self.assignments:
            lines.append(f"    {task.task_id} → {agent.agent_id} (bid={bid:.2f})")
        if self.no_bidders:
            lines.append(f"  No bidders:  {[t.task_id for t in self.no_bidders]}")
        if self.unassigned:
            lines.append(f"  Unassigned:  {[t.task_id for t in self.unassigned]}")
        return "\n".join(lines)


class FleetCoordinator:
    """
    Orchestrates a fleet of FleetAgents.

    Responsibilities:
    1. Run auction rounds: pull pending tasks, solicit bids, assign to winners.
    2. Step all agents forward each simulation tick.
    3. Collect agent observations into CollectiveIntelligence.
    4. Report fleet-wide status.

    Auction mechanics:
    - Each pending task is auctioned independently.
    - All idle (or low-load) agents submit sealed bids.
    - Lowest bid wins (bid = travel distance + load penalty - urgency discount).
    - Agents exceeding max_concurrent_tasks are excluded from bidding.
    """

    def __init__(
        self,
        intelligence: CollectiveIntelligence,
        max_concurrent_tasks: int = 2,
    ):
        self.intelligence = intelligence
        self.max_concurrent_tasks = max_concurrent_tasks
        self.agents: Dict[str, FleetAgent] = {}
        self.events: List[str] = []
        self.tick: int = 0

    # ------------------------------------------------------------------
    # Fleet management
    # ------------------------------------------------------------------

    def add_agent(self, agent: FleetAgent) -> None:
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str) -> None:
        self.agents.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Auction
    # ------------------------------------------------------------------

    def run_auction(self, task_queue: TaskQueue, max_tasks: int = 10) -> AuctionResult:
        """
        Pull up to max_tasks from the queue and run a single auction round.

        Each task is assigned to the agent with the lowest bid.
        """
        result = AuctionResult()

        tasks_to_assign = []
        for _ in range(max_tasks):
            task = task_queue.pop_next()
            if task is None:
                break
            tasks_to_assign.append(task)

        for task in tasks_to_assign:
            # Re-evaluate eligibility per task (load may have changed after prior assignments)
            eligible_agents = [
                a for a in self.agents.values()
                if a.state != AgentState.OFFLINE and a.load < self.max_concurrent_tasks
            ]
            bids = {}
            for agent in eligible_agents:
                bid_val = agent.bid(task)
                if bid_val is not None:
                    bids[agent.agent_id] = bid_val

            if not bids:
                result.no_bidders.append(task)
                # Put back in queue if no takers
                task.status = TaskStatus.PENDING
                task_queue.add(task)
                continue

            winner_id = min(bids, key=bids.__getitem__)
            winner = self.agents[winner_id]
            winner.assign_task(task)
            result.assignments.append((task, winner, bids[winner_id]))

        return result

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> List[str]:
        """
        Advance all agents one tick.

        Returns list of completion event strings.
        """
        self.tick += 1
        events = []

        for agent in self.agents.values():
            event = agent.step()
            if event:
                events.append(event)
                self.events.append(event)

            # Collect observations into collective intelligence
            obs = agent.drain_observations()
            if obs:
                self.intelligence.ingest_batch(obs)

        return events

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def fleet_status(self) -> str:
        lines = [f"=== Fleet Status (tick={self.tick}) ==="]
        for agent in sorted(self.agents.values(), key=lambda a: a.agent_id):
            lines.append("  " + agent.status_line())
        return "\n".join(lines)

    def all_idle(self) -> bool:
        return all(
            a.state == AgentState.IDLE or a.state == AgentState.OFFLINE
            for a in self.agents.values()
        )

    def completion_count(self) -> int:
        return sum(a.completed for a in self.agents.values())

    def total_tasks_assigned(self) -> int:
        return len(self.events)
