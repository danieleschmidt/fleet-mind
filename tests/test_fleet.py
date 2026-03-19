"""Tests for fleet-mind core components."""

import math
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_mind.agent import FleetAgent, Position, AgentState
from fleet_mind.tasks import Task, TaskQueue, TaskPriority, TaskStatus
from fleet_mind.coordinator import FleetCoordinator
from fleet_mind.intelligence import CollectiveIntelligence, Observation
from fleet_mind.simulator import FleetSimulator, MissionType


# ─────────────────────────────────────────────────────────────────────
# Position
# ─────────────────────────────────────────────────────────────────────
class TestPosition(unittest.TestCase):

    def test_distance(self):
        p1 = Position(0, 0)
        p2 = Position(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_move_toward_reaches_target(self):
        p1 = Position(0, 0)
        p2 = Position(1, 0)
        moved = p1.move_toward(p2, speed=10)
        self.assertAlmostEqual(moved.x, 1.0)
        self.assertAlmostEqual(moved.y, 0.0)

    def test_move_toward_partial(self):
        p1 = Position(0, 0)
        p2 = Position(10, 0)
        moved = p1.move_toward(p2, speed=3)
        self.assertAlmostEqual(moved.x, 3.0)
        self.assertAlmostEqual(moved.y, 0.0)


# ─────────────────────────────────────────────────────────────────────
# FleetAgent
# ─────────────────────────────────────────────────────────────────────
class TestFleetAgent(unittest.TestCase):

    def _make_agent(self, caps=None):
        return FleetAgent(
            agent_id="test-agent",
            position=Position(0, 0),
            capabilities=caps or {"carry", "navigate"},
            speed=100.0,  # fast so it arrives in 1 step
        )

    def _make_task(self, reqs=None, execution_time=1):
        return Task(
            task_id="T1",
            task_type="deliver",
            location=Position(1, 0),
            requirements=reqs or {"carry"},
            execution_time=execution_time,
            deadline=time.time() + 300,
        )

    def test_bid_returns_none_missing_capability(self):
        agent = self._make_agent(caps={"scan"})
        task = self._make_task(reqs={"carry"})
        self.assertIsNone(agent.bid(task))

    def test_bid_returns_float_when_capable(self):
        agent = self._make_agent(caps={"carry", "navigate"})
        task = self._make_task(reqs={"carry"})
        bid = agent.bid(task)
        self.assertIsNotNone(bid)
        self.assertIsInstance(bid, float)

    def test_assign_and_step_to_completion(self):
        agent = self._make_agent()
        task = self._make_task(execution_time=1)

        agent.assign_task(task)
        self.assertEqual(agent.state, AgentState.MOVING)

        # Step 1: moves to location
        event = agent.step()
        self.assertEqual(agent.state, AgentState.EXECUTING)
        self.assertIsNone(event)

        # Step 2: executes
        event = agent.step()
        self.assertIsNotNone(event)
        self.assertIn("completed", event)
        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertEqual(agent.completed, 1)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_battery_drain(self):
        agent = self._make_agent()
        initial_battery = agent.battery
        agent.step()
        self.assertLess(agent.battery, initial_battery)

    def test_observations_drained(self):
        agent = self._make_agent()
        task = self._make_task(execution_time=1)
        agent.assign_task(task)
        agent.step()  # move
        agent.step()  # execute + complete => records observation

        obs = agent.drain_observations()
        self.assertTrue(any(o.get("type") == "task_complete" for o in obs))
        # Second drain should be empty
        self.assertEqual(agent.drain_observations(), [])


# ─────────────────────────────────────────────────────────────────────
# TaskQueue
# ─────────────────────────────────────────────────────────────────────
class TestTaskQueue(unittest.TestCase):

    def _task(self, tid, priority=TaskPriority.NORMAL, deadline_offset=300):
        return Task(
            task_id=tid,
            task_type="test",
            location=Position(0, 0),
            priority=priority,
            deadline=time.time() + deadline_offset,
        )

    def test_priority_ordering(self):
        tq = TaskQueue()
        tq.add(self._task("low",  TaskPriority.LOW))
        tq.add(self._task("high", TaskPriority.HIGH))
        tq.add(self._task("crit", TaskPriority.CRITICAL))
        tq.add(self._task("norm", TaskPriority.NORMAL))

        first = tq.pop_next()
        self.assertEqual(first.task_id, "crit")
        second = tq.pop_next()
        self.assertEqual(second.task_id, "high")

    def test_empty_queue_returns_none(self):
        tq = TaskQueue()
        self.assertIsNone(tq.pop_next())

    def test_expired_task_skipped(self):
        tq = TaskQueue()
        tq.add(self._task("expired", deadline_offset=-1))  # already expired
        tq.add(self._task("fresh",   deadline_offset=300))
        result = tq.pop_next()
        self.assertEqual(result.task_id, "fresh")

    def test_stats(self):
        tq = TaskQueue()
        tq.add(self._task("t1"))
        tq.add(self._task("t2"))
        stats = tq.stats()
        self.assertEqual(stats["PENDING"], 2)

    def test_len(self):
        tq = TaskQueue()
        for i in range(5):
            tq.add(self._task(f"t{i}"))
        self.assertEqual(len(tq), 5)


# ─────────────────────────────────────────────────────────────────────
# FleetCoordinator
# ─────────────────────────────────────────────────────────────────────
class TestFleetCoordinator(unittest.TestCase):

    def setUp(self):
        self.intel = CollectiveIntelligence()
        self.coord = FleetCoordinator(self.intel)
        self.agent = FleetAgent("A1", Position(0, 0), {"carry", "navigate"}, speed=100)
        self.coord.add_agent(self.agent)

    def test_auction_assigns_task(self):
        tq = TaskQueue()
        tq.add(Task("T1", "deliver", Position(5, 5),
                    requirements={"carry"}, deadline=time.time() + 300))
        result = self.coord.run_auction(tq)
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(self.agent.load, 1)

    def test_no_bidders_when_no_capability(self):
        tq = TaskQueue()
        tq.add(Task("T1", "scan", Position(5, 5),
                    requirements={"scan"}, deadline=time.time() + 300))
        result = self.coord.run_auction(tq)
        self.assertEqual(len(result.assignments), 0)
        self.assertEqual(len(result.no_bidders), 1)

    def test_step_advances_agents(self):
        tq = TaskQueue()
        task = Task("T1", "deliver", Position(1, 0),
                    requirements={"carry"}, execution_time=1,
                    deadline=time.time() + 300)
        tq.add(task)
        self.coord.run_auction(tq)

        # Step until complete
        events = []
        for _ in range(10):
            e = self.coord.step()
            events.extend(e)

        self.assertTrue(any("completed" in e for e in events))

    def test_all_idle_when_done(self):
        self.assertTrue(self.coord.all_idle())


# ─────────────────────────────────────────────────────────────────────
# CollectiveIntelligence
# ─────────────────────────────────────────────────────────────────────
class TestCollectiveIntelligence(unittest.TestCase):

    def setUp(self):
        self.intel = CollectiveIntelligence(ttl=60.0)

    def test_ingest_and_query_hazard(self):
        obs = Observation("A1", "hazard", (10.0, 10.0), {"severity": "medium"})
        self.intel.ingest(obs)
        hazards = self.intel.hazards_near((10.0, 10.0), radius=5.0)
        self.assertEqual(len(hazards), 1)

    def test_hazard_outside_radius_not_returned(self):
        obs = Observation("A1", "hazard", (50.0, 50.0), {})
        self.intel.ingest(obs)
        hazards = self.intel.hazards_near((0.0, 0.0), radius=5.0)
        self.assertEqual(len(hazards), 0)

    def test_waypoint_clear(self):
        obs = Observation("A1", "waypoint_clear", (5.0, 5.0), {})
        self.intel.ingest(obs)
        self.assertTrue(self.intel.is_cleared((5.0, 5.0)))
        self.assertFalse(self.intel.is_cleared((99.0, 99.0)))

    def test_target_found(self):
        obs = Observation("A2", "target_found", (20.0, 30.0),
                          {"target_id": "SURVIVOR_1"})
        self.intel.ingest(obs)
        targets = self.intel.active_targets()
        self.assertIn("SURVIVOR_1", targets)

    def test_ingest_batch(self):
        raw_obs = [
            {"agent_id": "A1", "type": "task_complete", "location": (3.0, 4.0),
             "task_id": "T42", "timestamp": time.time()},
        ]
        self.intel.ingest_batch(raw_obs)
        self.assertEqual(self.intel.task_completion_count(), 1)

    def test_situational_report_is_string(self):
        report = self.intel.situational_report()
        self.assertIsInstance(report, str)
        self.assertIn("Fleet Situational Report", report)


# ─────────────────────────────────────────────────────────────────────
# FleetSimulator (integration)
# ─────────────────────────────────────────────────────────────────────
class TestFleetSimulator(unittest.TestCase):

    def _run(self, mission_type):
        sim = FleetSimulator(seed=7, verbose=False)
        return sim.run(
            mission_type=mission_type,
            num_agents=5,
            num_tasks=10,
            arena_size=50.0,
            max_ticks=500,
        )

    def test_delivery_mission_completes_tasks(self):
        result = self._run(MissionType.DELIVERY)
        self.assertGreater(result.completed_tasks, 0)
        self.assertEqual(result.total_tasks, 10)

    def test_patrol_mission(self):
        result = self._run(MissionType.COVERAGE_PATROL)
        self.assertGreater(result.completed_tasks, 0)

    def test_sar_mission(self):
        result = self._run(MissionType.SEARCH_AND_RESCUE)
        self.assertGreater(result.completed_tasks, 0)

    def test_completion_rate_range(self):
        for mtype in MissionType:
            result = self._run(mtype)
            self.assertGreaterEqual(result.completion_rate, 0.0)
            self.assertLessEqual(result.completion_rate, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
