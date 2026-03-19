# Fleet-Mind

**Collective intelligence for autonomous agent fleets.**

Fleet-Mind is a pure-Python fleet coordination system built on auction-based task allocation and distributed situational awareness. Five modules, no external dependencies.

---

## Architecture

```
FleetSimulator
    └── FleetCoordinator ──── CollectiveIntelligence
            ├── FleetAgent (×N)
            └── TaskQueue
```

| Module | Role |
|--------|------|
| `FleetAgent` | One autonomous unit — position, capability set, battery, bid computation, task execution |
| `FleetCoordinator` | Runs auction rounds, steps agents forward, collects observations |
| `TaskQueue` | Min-heap priority queue (priority → deadline → id) with expiry |
| `CollectiveIntelligence` | Shared knowledge base — hazard map, cleared waypoints, active targets, situational report |
| `FleetSimulator` | End-to-end mission runner — patrol, delivery, search-and-rescue |

---

## Auction-Based Task Allocation

Each tick the coordinator pulls pending tasks and runs sealed-bid auctions:

```
bid = travel_distance + load_penalty - urgency_discount
```

- **Travel distance** — Euclidean distance from agent's current position to task location.
- **Load penalty** — `5.0 × current_load` discourages overloading busy agents.
- **Urgency discount** — Tasks nearing their deadline attract more competitive bids.
- Agents that lack required capabilities or have low battery submit no bid.
- The lowest bid wins and the task is immediately assigned.

---

## Quick Start

```bash
# No dependencies needed — stdlib only
python3 demo.py
```

### Run a specific mission

```python
from fleet_mind import FleetSimulator, MissionType

sim = FleetSimulator(seed=42)
result = sim.run(
    mission_type=MissionType.SEARCH_AND_RESCUE,
    num_agents=5,
    num_tasks=10,
    arena_size=50.0,
)
print(result.summary())
```

### Manual fleet setup

```python
from fleet_mind import (
    FleetAgent, FleetCoordinator, TaskQueue, Task,
    TaskPriority, CollectiveIntelligence
)
from fleet_mind.agent import Position

intel = CollectiveIntelligence()
coord = FleetCoordinator(intel, max_concurrent_tasks=2)

coord.add_agent(FleetAgent("Drone-1", Position(0, 0), {"carry", "navigate"}, speed=3.0))
coord.add_agent(FleetAgent("Drone-2", Position(5, 5), {"scan",  "navigate"}, speed=2.5))

tq = TaskQueue()
tq.add(Task("T001", "deliver", Position(20, 15), TaskPriority.HIGH, requirements={"carry"}))
tq.add(Task("T002", "scan",    Position(30, 10), TaskPriority.NORMAL, requirements={"scan"}))

# Auction — assigns tasks to best-fit agents
result = coord.run_auction(tq)
print(result.summary())

# Step simulation forward
for _ in range(50):
    events = coord.step()
    for e in events:
        print(e)

# Fleet-wide situational awareness
print(intel.situational_report())
```

---

## Mission Types

| Mission | Task Type | Capabilities Required |
|---------|-----------|----------------------|
| `COVERAGE_PATROL` | Sweep grid waypoints | `scan`, `navigate` |
| `DELIVERY` | Deliver packages to destinations | `carry`, `navigate` |
| `SEARCH_AND_RESCUE` | Scout sectors + extract survivors | `scan`/`rescue`, `navigate` |

---

## Tests

```bash
python3 -m pytest tests/ -v
```

27 tests covering positions, agent lifecycle, task queue priority/expiry, auction mechanics, collective intelligence, and full-mission integration.

---

## Design Principles

- **No external dependencies** — stdlib only (`heapq`, `dataclasses`, `math`, `time`, `random`, `uuid`)
- **Composable** — each module works standalone; the simulator just wires them together
- **Observable** — agents emit structured observations; coordinator feeds them to shared intelligence
- **Extensible** — add new `MissionType`, `TaskPriority`, capability strings, or bid functions without changing core logic
