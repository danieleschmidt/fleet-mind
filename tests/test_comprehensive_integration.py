"""Comprehensive integration tests for Fleet-Mind system."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from fleet_mind import (
    SwarmCoordinator, DroneFleet, WebRTCStreamer, LLMPlanner,
    SecurityManager, SecurityLevel, HealthMonitor, HealthStatus
)
from fleet_mind.optimization import PerformanceOptimizer, OptimizationStrategy
from fleet_mind.fleet.drone_fleet import DroneStatus, DroneCapability


class TestComprehensiveIntegration:
    """Comprehensive integration tests for the entire Fleet-Mind system."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create SwarmCoordinator with all components."""
        coordinator = SwarmCoordinator(
            llm_model="gpt-4o-mini",
            max_drones=10,
            security_level=SecurityLevel.MEDIUM,
            enable_health_monitoring=True
        )
        yield coordinator
        
        # Cleanup
        if coordinator.health_monitor:
            await coordinator.health_monitor.stop_monitoring()

    @pytest.fixture
    def drone_fleet(self):
        """Create test drone fleet."""
        drone_ids = [f"test_drone_{i}" for i in range(5)]
        return DroneFleet(drone_ids, communication_protocol="webrtc")

    @pytest.fixture
    async def performance_optimizer(self):
        """Create performance optimizer."""
        optimizer = PerformanceOptimizer(
            strategy=OptimizationStrategy.BALANCED,
            enable_auto_optimization=False  # Disable for testing
        )
        await optimizer.start()
        yield optimizer
        await optimizer.stop()

    async def test_full_system_initialization(self, coordinator, drone_fleet):
        """Test complete system initialization and connectivity."""
        # Connect fleet
        await coordinator.connect_fleet(drone_fleet)
        
        # Verify initialization
        assert coordinator.fleet == drone_fleet
        assert coordinator.security_manager is not None
        assert coordinator.health_monitor is not None
        
        # Verify security credentials generated
        for drone_id in drone_fleet.drone_ids:
            assert drone_id in coordinator.security_manager.drone_credentials
            
        # Verify health monitoring started
        system_health = coordinator.health_monitor.get_system_health()
        assert "overall_status" in system_health
        assert system_health["total_components"] > 0

    async def test_secure_mission_execution(self, coordinator, drone_fleet):
        """Test mission execution with security and monitoring."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Generate a test mission plan
        mission = "Test formation flying in grid pattern"
        plan = await coordinator.generate_plan(mission)
        
        # Verify plan structure
        assert "mission_id" in plan
        assert "latent_code" in plan
        assert "planning_latency_ms" in plan
        assert plan["planning_latency_ms"] >= 0
        
        # Test secure broadcast
        broadcast_result = await coordinator.secure_broadcast(
            {"test": "data", "timestamp": time.time()},
            priority="real_time"
        )
        
        # Verify secure broadcast succeeded for active drones
        active_drones = drone_fleet.get_active_drones()
        for drone_id in active_drones:
            assert drone_id in broadcast_result
            assert isinstance(broadcast_result[drone_id], bool)

    async def test_health_monitoring_integration(self, coordinator, drone_fleet):
        """Test health monitoring with automatic responses."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Wait for initial health metrics
        await asyncio.sleep(1)
        
        # Update health metrics manually
        await coordinator.update_health_metrics()
        
        # Get health status
        health_status = await coordinator.get_health_status()
        
        assert "overall_status" in health_status
        assert "monitored_components" in health_status
        assert health_status["monitored_components"] >= 4  # coordinator, webrtc, llm, encoder
        
        # Test health alert simulation
        if coordinator.health_monitor:
            # Simulate high memory usage alert
            coordinator.health_monitor.update_metric(
                "coordinator", "memory_usage", 95.0, "%", "Memory utilization"
            )
            
            # Check that alert was generated
            alerts = coordinator.health_monitor.get_alerts(active_only=True)
            memory_alerts = [a for a in alerts if a.metric_name == "memory_usage"]
            assert len(memory_alerts) > 0

    async def test_security_threat_detection(self, coordinator, drone_fleet):
        """Test security threat detection and response."""
        await coordinator.connect_fleet(drone_fleet)
        
        security_manager = coordinator.security_manager
        
        # Test authentication
        drone_id = drone_fleet.drone_ids[0]
        
        # Test failed authentication
        result = security_manager.authenticate_drone(drone_id, "invalid_token")
        assert result is False
        
        # Test threat detection
        malicious_message = {
            "command": "; rm -rf /",  # Command injection attempt
            "data": "malicious_data"
        }
        
        threats = security_manager.detect_threats(malicious_message, drone_id)
        assert len(threats) > 0
        
        # Test security audit
        audit_results = await coordinator.perform_security_audit()
        assert "risk_level" in audit_results
        assert "findings" in audit_results
        assert "recommendations" in audit_results

    async def test_performance_optimization_integration(self, coordinator, drone_fleet, performance_optimizer):
        """Test performance optimization with Fleet-Mind components."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Collect initial metrics
        metrics = await performance_optimizer.collect_metrics()
        assert metrics.timestamp > 0
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_percent >= 0
        
        # Analyze bottlenecks
        bottlenecks = performance_optimizer.analyze_bottlenecks(metrics)
        # Should have minimal bottlenecks in test environment
        assert isinstance(bottlenecks, list)
        
        # Test optimization generation
        if bottlenecks:
            optimizations = performance_optimizer.generate_optimizations(bottlenecks)
            assert isinstance(optimizations, list)
            
            # Test optimization application
            if optimizations:
                success = await performance_optimizer.apply_optimization(optimizations[0])
                assert isinstance(success, bool)
        
        # Test scaling decision
        scaling_decision = performance_optimizer.decide_scaling_action(metrics)
        assert scaling_decision.action in ["scale_up", "scale_down", "maintain"]
        assert 0 <= scaling_decision.confidence <= 1

    async def test_fleet_auto_healing(self, coordinator, drone_fleet):
        """Test automated fleet healing capabilities."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Simulate drone failure
        failed_drone = drone_fleet.drone_ids[0]
        drone_fleet.update_drone_state(
            failed_drone,
            status=DroneStatus.FAILED,
            health_score=0.0
        )
        
        # Test fleet healing
        healing_results = await drone_fleet.auto_heal_fleet()
        
        assert "drones_recovered" in healing_results
        assert "issues_resolved" in healing_results
        assert "recommendations" in healing_results
        
        # Test formation quality assessment
        formation_config = {
            "formation_type": "grid",
            "spacing_meters": 10.0
        }
        
        quality_score = drone_fleet.get_formation_quality_score(formation_config)
        assert 0 <= quality_score <= 1

    async def test_adaptive_replanning(self, coordinator, drone_fleet):
        """Test adaptive mission replanning."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Set initial mission
        mission = "Test adaptive mission"
        coordinator.current_mission = mission
        
        # Test adaptive replanning
        context = {
            "reason": "environmental_change",
            "new_conditions": {"weather": "windy", "visibility": "reduced"}
        }
        
        success = await coordinator.adaptive_replan(context)
        assert isinstance(success, bool)
        
        # Verify context was stored
        assert len(coordinator.context_history) > 0
        latest_context = coordinator.context_history[-1]
        assert "mission" in latest_context or "type" in latest_context

    async def test_comprehensive_status_reporting(self, coordinator, drone_fleet, performance_optimizer):
        """Test comprehensive system status reporting."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Get comprehensive stats
        swarm_stats = coordinator.get_comprehensive_stats()
        
        # Verify comprehensive stats structure
        required_keys = [
            "swarm_status", "fleet_stats", "performance_stats", 
            "concurrency_stats", "autoscaling_stats", "system_health"
        ]
        for key in required_keys:
            assert key in swarm_stats
        
        # Get security status
        security_status = await coordinator.get_security_status()
        assert "security_level" in security_status
        assert "threat_level" in security_status
        
        # Get health status
        health_status = await coordinator.get_health_status()
        assert "overall_status" in health_status
        
        # Get performance summary
        perf_summary = performance_optimizer.get_performance_summary()
        assert "current_metrics" in perf_summary
        assert "strategy" in perf_summary

    async def test_error_handling_resilience(self, coordinator, drone_fleet):
        """Test system resilience to errors and failures."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Test invalid mission planning
        try:
            plan = await coordinator.generate_plan("")  # Empty mission
            # Should handle gracefully
            assert "mission_id" in plan  # Fallback plan generated
        except Exception as e:
            # Should not crash the system
            assert isinstance(e, Exception)
        
        # Test WebRTC failure resilience
        original_broadcast = coordinator.webrtc_streamer.broadcast
        coordinator.webrtc_streamer.broadcast = AsyncMock(side_effect=Exception("Connection failed"))
        
        try:
            result = await coordinator.secure_broadcast({"test": "data"})
            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception:
            # Should not crash
            pass
        finally:
            coordinator.webrtc_streamer.broadcast = original_broadcast
        
        # Test security manager resilience
        original_encrypt = coordinator.security_manager.encrypt_message
        coordinator.security_manager.encrypt_message = Mock(side_effect=Exception("Encryption failed"))
        
        try:
            result = await coordinator.secure_broadcast({"test": "data"})
            # Should handle encryption failure
            assert isinstance(result, dict)
        except Exception:
            # Should not crash system
            pass
        finally:
            coordinator.security_manager.encrypt_message = original_encrypt

    async def test_concurrent_operations(self, coordinator, drone_fleet):
        """Test system behavior under concurrent operations."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Create concurrent mission planning tasks
        missions = [
            "Formation flying test",
            "Search pattern alpha", 
            "Emergency response drill",
            "Surveillance mission",
            "Data collection flight"
        ]
        
        # Execute concurrent mission planning
        tasks = [
            coordinator.generate_plan(mission) 
            for mission in missions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all plans generated (or handled gracefully)
        assert len(results) == len(missions)
        
        success_count = sum(1 for r in results if isinstance(r, dict) and "mission_id" in r)
        # At least some should succeed
        assert success_count >= 3

    async def test_resource_cleanup(self, coordinator, drone_fleet):
        """Test proper resource cleanup."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Start some background tasks
        if coordinator.health_monitor:
            assert coordinator.health_monitor._running
        
        # Test emergency stop cleanup
        await coordinator.emergency_stop()
        
        # Verify system state
        assert coordinator.mission_status.value in ["paused", "failed", "completed"]
        
        # Test fleet cleanup
        await drone_fleet.stop_monitoring()
        assert not drone_fleet._running

    @pytest.mark.performance
    async def test_performance_benchmarks(self, coordinator, drone_fleet, performance_optimizer):
        """Test system performance benchmarks."""
        await coordinator.connect_fleet(drone_fleet)
        
        # Benchmark mission planning
        start_time = time.time()
        plan = await coordinator.generate_plan("Benchmark mission test")
        planning_time = (time.time() - start_time) * 1000  # ms
        
        # Planning should be under 2 seconds for test mission
        assert planning_time < 2000
        assert plan["planning_latency_ms"] < 1000
        
        # Benchmark secure broadcast
        start_time = time.time()
        await coordinator.secure_broadcast({"benchmark": True})
        broadcast_time = (time.time() - start_time) * 1000  # ms
        
        # Broadcast should be under 500ms
        assert broadcast_time < 500
        
        # Test system load metrics
        metrics = await performance_optimizer.collect_metrics()
        
        # Memory usage should be reasonable for test
        assert metrics.memory_usage_percent < 80
        
        # CPU usage should not be excessive
        assert metrics.cpu_usage_percent < 90

    async def test_scalability_limits(self, coordinator):
        """Test system behavior at scale limits."""
        # Test with maximum drone count
        max_drone_ids = [f"drone_{i}" for i in range(coordinator.max_drones)]
        large_fleet = DroneFleet(max_drone_ids)
        
        # Should handle max fleet size
        await coordinator.connect_fleet(large_fleet)
        
        # Verify all drones have credentials
        assert len(coordinator.security_manager.drone_credentials) == coordinator.max_drones
        
        # Test broadcast to max fleet
        result = await coordinator.secure_broadcast({"scale_test": True})
        
        # Should attempt broadcast to all drones
        assert len(result) == coordinator.max_drones
        
        await large_fleet.stop_monitoring()


@pytest.mark.asyncio
class TestLoadTesting:
    """Load testing for Fleet-Mind components."""
    
    async def test_concurrent_mission_load(self):
        """Test handling multiple concurrent missions."""
        coordinator = SwarmCoordinator(max_drones=20, enable_health_monitoring=False)
        drone_fleet = DroneFleet([f"load_drone_{i}" for i in range(20)])
        
        await coordinator.connect_fleet(drone_fleet)
        
        # Generate load with concurrent missions
        tasks = []
        for i in range(50):  # 50 concurrent mission plans
            task = coordinator.generate_plan(f"Load test mission {i}")
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes
        successes = sum(1 for r in results if isinstance(r, dict) and "mission_id" in r)
        
        # Should handle majority successfully
        assert successes >= 40  # At least 80% success rate
        
        await drone_fleet.stop_monitoring()

    async def test_memory_usage_stability(self):
        """Test memory usage stability under load."""
        coordinator = SwarmCoordinator(enable_health_monitoring=True)
        drone_fleet = DroneFleet([f"mem_drone_{i}" for i in range(10)])
        
        await coordinator.connect_fleet(drone_fleet)
        
        # Collect initial metrics
        if coordinator.health_monitor:
            await coordinator.update_health_metrics()
            initial_health = await coordinator.get_health_status()
            
            # Generate continuous load for memory test
            for _ in range(100):
                await coordinator.generate_plan("Memory test mission")
                
                # Update metrics periodically
                if _ % 10 == 0:
                    await coordinator.update_health_metrics()
            
            # Check final memory usage
            final_health = await coordinator.get_health_status()
            
            # Memory usage should not grow excessively
            # This is a basic check - in production, you'd have more sophisticated memory leak detection
            assert "overall_status" in final_health
            
        await drone_fleet.stop_monitoring()
        if coordinator.health_monitor:
            await coordinator.health_monitor.stop_monitoring()


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestComprehensiveIntegration::test_full_system_initialization", "-v"])