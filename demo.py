#!/usr/bin/env python3
"""
Fleet-Mind Demo

Runs a 5-agent, 10-task fleet mission showing:
- Auction-based task allocation
- Collective intelligence (shared situational awareness)
- Mission completion across delivery, patrol, and search-and-rescue scenarios
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from fleet_mind import (
    FleetAgent,
    FleetCoordinator,
    TaskQueue,
    Task,
    TaskPriority,
    CollectiveIntelligence,
    FleetSimulator,
    MissionType,
    Observation,
)
from fleet_mind.agent import Position


def demo_auction_mechanics():
    """Demonstrate manual auction setup and bidding."""
    print("\n" + "=" * 60)
    print("DEMO 1: Auction Mechanics")
    print("=" * 60)

    intel = CollectiveIntelligence()
    coordinator = FleetCoordinator(intel, max_concurrent_tasks=2)

    # Three agents with different capabilities and positions
    agents = [
        FleetAgent("Scout-1", Position(0, 0),   {"scan", "navigate"},           speed=3.0),
        FleetAgent("Hauler-1", Position(10, 5),  {"carry", "navigate"},          speed=2.0),
        FleetAgent("Rescue-1", Position(5, 5),   {"rescue", "scan", "navigate"}, speed=2.5),
    ]
    for a in agents:
        coordinator.add_agent(a)

    # A few tasks with varying requirements
    tq = TaskQueue()
    import time
    now = time.time()

    tq.add(Task("T001", "deliver",  Position(20, 10), TaskPriority.HIGH,   now + 180,
                requirements={"carry"}, execution_time=3))
    tq.add(Task("T002", "scout",    Position(30, 30), TaskPriority.NORMAL, now + 300,
                requirements={"scan"}, execution_time=2))
    tq.add(Task("T003", "rescue",   Position(15, 20), TaskPriority.CRITICAL, now + 120,
                requirements={"rescue"}, execution_time=5))

    print(f"\nTask queue: {tq}")
    print(f"\nRunning auction...")
    result = coordinator.run_auction(tq, max_tasks=5)
    print(result.summary())

    print("\nAgent states after auction:")
    for agent in agents:
        print(f"  {agent.status_line()}")


def demo_collective_intelligence():
    """Demonstrate shared situational awareness."""
    print("\n" + "=" * 60)
    print("DEMO 2: Collective Intelligence")
    print("=" * 60)

    intel = CollectiveIntelligence()

    # Agents report observations
    intel.ingest(Observation("A01", "hazard",        (25, 35), {"severity": "high"}))
    intel.ingest(Observation("A02", "waypoint_clear",(10, 10), {}))
    intel.ingest(Observation("A03", "target_found",  (40, 20), {"target_id": "SURVIVOR_1"}))
    intel.ingest(Observation("A01", "waypoint_clear",(20, 20), {}))
    intel.ingest(Observation("A04", "task_complete", (30, 10), {"task_id": "T007"}))

    print(intel.situational_report())

    print(f"\nHazards near (25,35) radius=15: {len(intel.hazards_near((25,35), 15))} obs")
    print(f"Is (10,10) cleared: {intel.is_cleared((10,10))}")
    print(f"Active targets: {list(intel.active_targets().keys())}")


def demo_full_mission(mission_type: MissionType, num_agents=5, num_tasks=10):
    """Run a full simulated fleet mission."""
    print("\n" + "=" * 60)
    print(f"DEMO 3: Full Mission — {mission_type.name}")
    print("=" * 60)

    sim = FleetSimulator(seed=42, verbose=True)
    result = sim.run(
        mission_type=mission_type,
        num_agents=num_agents,
        num_tasks=num_tasks,
        arena_size=50.0,
        max_ticks=300,
        auction_interval=5,
    )
    print(result.summary())
    return result


if __name__ == "__main__":
    demo_auction_mechanics()
    demo_collective_intelligence()

    # Run all three mission types
    results = {}
    for mtype in MissionType:
        results[mtype] = demo_full_mission(mtype)

    # Final summary
    print("\n" + "=" * 60)
    print("ALL MISSIONS SUMMARY")
    print("=" * 60)
    for mtype, result in results.items():
        print(f"  {mtype.name:<25} {result.completed_tasks}/{result.total_tasks} "
              f"tasks  ({result.completion_rate:.0%})  in {result.total_ticks} ticks")
