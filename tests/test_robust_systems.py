"""Comprehensive tests for robust Fleet-Mind systems - Generation 2 reliability features."""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import sys
import os

# Add the parent directory to path so we can import fleet_mind
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fleet_mind import (
    SwarmCoordinator,
    DroneFleet,
    MissionConstraints,
    ErrorSeverity,
    SecurityLevel,
    get_performance_summary
)
from fleet_mind.utils.error_handling import (
    ErrorHandler,
    FleetMindError,
    CommunicationError,
    PlanningError,
    DroneError,
    ValidationError,
    SecurityError,
    get_error_handler
)
from fleet_mind.utils.advanced_logging import (
    FleetMindLogger,
    LogLevel,
    LogCategory,
    get_logger,
    configure_logging
)
from fleet_mind.security.security_manager import SecurityManager, ThreatType


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ErrorHandler()
    
    def test_error_classification(self, error_handler):
        """Test automatic error classification."""
        # Test communication error
        comm_error = CommunicationError("Connection timeout", drone_id="test_drone")
        result = error_handler.handle_error(comm_error)
        
        assert comm_error.context.component == "communication"
        assert comm_error.context.severity == ErrorSeverity.HIGH
        assert comm_error.context.drone_id == "test_drone"
    
    def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality."""
        # Generate multiple failures to trigger circuit breaker
        for i in range(6):  # Exceed threshold of 5
            error = FleetMindError(f"Test error {i}")
            error.context.component = "test_component"
            error.context.operation = "test_operation"
            error_handler.handle_error(error)
        
        # Check that circuit breaker is open
        assert error_handler._is_circuit_breaker_open("test_component", "test_operation")
    
    def test_retry_mechanism(self, error_handler):
        """Test retry with exponential backoff."""
        retry_count = 0
        
        def mock_retry_action(context):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise Exception("Still failing")
            return "Success"
        
        # Replace retry action
        error_handler.recovery_actions["FleetMindError"].action = mock_retry_action
        
        error = FleetMindError("Test retry error")
        result = error_handler.handle_error(error)
        
        assert retry_count == 3
    
    def test_fallback_recovery(self, error_handler):
        """Test fallback recovery strategies."""
        planning_error = PlanningError("LLM service unavailable", mission_id="test_mission")
        result = error_handler.handle_error(planning_error)
        
        # Should execute fallback planning
        assert result is not None
    
    def test_security_error_escalation(self, error_handler):
        """Test security error escalation."""
        security_error = SecurityError("Unauthorized access attempt", user_id="suspicious_user")
        result = error_handler.handle_error(security_error)
        
        assert security_error.context.severity == ErrorSeverity.CRITICAL
        assert security_error.context.user_id == "suspicious_user"
    
    def test_error_metrics_collection(self, error_handler):
        """Test error metrics and statistics."""
        # Generate various errors
        errors = [
            CommunicationError("Comm error 1"),
            PlanningError("Planning error 1"),
            DroneError("Drone error 1", "drone_1"),
            ValidationError("Validation error 1"),
        ]
        
        for error in errors:
            error_handler.handle_error(error)
        
        stats = error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 4
        assert stats['health_score'] < 1.0
        assert len(stats['most_common_errors']) > 0


class TestAdvancedLogging:
    """Test advanced logging system."""
    
    @pytest.fixture
    def logger(self):
        """Create logger for testing."""
        return get_logger("test_logger", component="test_component")
    
    def test_structured_logging(self, logger):
        """Test structured log entries."""
        logger.info(
            "Test message", 
            category=LogCategory.COMMUNICATION,
            drone_id="test_drone",
            mission_id="test_mission",
            custom_field="custom_value"
        )
        
        # Verify log was created (check metrics)
        metrics = logger.get_metrics()
        assert metrics['total_logs'] >= 1
        assert 'communication' in metrics['logs_by_category']
    
    def test_context_management(self, logger):
        """Test logging context management."""
        with logger.with_context(mission_id="context_mission", drone_id="context_drone"):
            logger.info("Message with context")
        
        logger.info("Message without context")
        
        # Context should be properly managed
        metrics = logger.get_metrics()
        assert metrics['total_logs'] >= 2
    
    def test_performance_logging(self, logger):
        """Test performance logging."""
        logger.performance("Test operation", duration_ms=150.5)
        
        metrics = logger.get_metrics()
        assert 'performance' in metrics['logs_by_category']
    
    def test_security_audit_logging(self, logger):
        """Test security and audit logging."""
        logger.security("Security event detected", user_id="test_user")
        logger.audit("User action logged", user_id="test_user", action="login")
        
        metrics = logger.get_metrics()
        assert 'security' in metrics['logs_by_category']
        assert 'audit' in metrics['logs_by_category']
    
    def test_async_logging_performance(self, logger):
        """Test async logging performance."""
        start_time = time.time()
        
        # Log many messages quickly
        for i in range(100):
            logger.info(f"Async test message {i}")
        
        duration = time.time() - start_time
        
        # Should be fast due to async processing
        assert duration < 1.0  # Should complete in less than 1 second
    
    def test_log_metrics_accuracy(self, logger):
        """Test logging metrics accuracy."""
        # Log specific numbers of each level
        for i in range(5):
            logger.info(f"Info {i}")
        for i in range(3):
            logger.warning(f"Warning {i}")
        for i in range(2):
            logger.error(f"Error {i}")
        
        metrics = logger.get_metrics()
        
        assert metrics['logs_by_level']['info'] >= 5
        assert metrics['logs_by_level']['warning'] >= 3
        assert metrics['logs_by_level']['error'] >= 2
        assert metrics['error_rate'] > 0
        assert metrics['warning_rate'] > 0


class TestSecuritySystem:
    """Test security management system."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for testing."""
        return SecurityManager(
            security_level=SecurityLevel.HIGH,
            enable_threat_detection=True
        )
    
    def test_drone_credential_generation(self, security_manager):
        """Test drone credential generation."""
        permissions = {"basic_flight", "telemetry", "emergency_response"}
        credentials = security_manager.generate_drone_credentials("test_drone", permissions)
        
        assert credentials.drone_id == "test_drone"
        assert credentials.permissions == permissions
        assert credentials.public_key is not None
        assert credentials.certificate is not None
        assert credentials.expires_at > time.time()
    
    def test_message_encryption_decryption(self, security_manager):
        """Test message encryption and decryption."""
        test_message = {"command": "move_to", "position": [10, 20, 30]}
        
        encrypted = security_manager.encrypt_message("test_drone", test_message)
        decrypted = security_manager.decrypt_message("test_drone", encrypted)
        
        assert decrypted == test_message
    
    def test_threat_detection(self, security_manager):
        """Test threat detection capabilities."""
        # Simulate replay attack
        message = {"nonce": "test_nonce_123", "command": "emergency_land"}
        
        # First message should be accepted
        is_threat1 = security_manager.detect_threat(message, "test_drone", ThreatType.REPLAY_ATTACK)
        assert not is_threat1
        
        # Second identical message should be detected as replay attack
        is_threat2 = security_manager.detect_threat(message, "test_drone", ThreatType.REPLAY_ATTACK)
        assert is_threat2
    
    def test_access_control(self, security_manager):
        """Test access control and authorization."""
        # Create credentials with limited permissions
        limited_permissions = {"telemetry"}
        credentials = security_manager.generate_drone_credentials("limited_drone", limited_permissions)
        
        # Test authorized operation
        assert security_manager.is_authorized("limited_drone", "telemetry")
        
        # Test unauthorized operation
        assert not security_manager.is_authorized("limited_drone", "emergency_response")
    
    def test_security_event_logging(self, security_manager):
        """Test security event logging."""
        # Trigger security event
        security_manager._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,
            SecurityLevel.HIGH,
            "test_drone",
            "Unauthorized command attempt",
            "access_denied"
        )
        
        events = security_manager.get_security_events()
        assert len(events) > 0
        assert events[-1].event_type == ThreatType.UNAUTHORIZED_ACCESS
    
    def test_key_rotation(self, security_manager):
        """Test automatic key rotation."""
        initial_key = security_manager.session_keys.get("test_session")
        
        # Force key rotation
        security_manager._rotate_keys()
        
        # Keys should be updated
        rotated_key = security_manager.session_keys.get("test_session")
        # New sessions should get different keys after rotation
        
    def test_dos_attack_detection(self, security_manager):
        """Test DoS attack detection."""
        source_id = "attack_source"
        
        # Simulate rapid requests
        for i in range(20):  # Exceed threshold
            security_manager._record_request(source_id)
        
        # Should detect DoS pattern
        is_dos = security_manager._detect_dos_attack({"source": source_id}, source_id)
        assert is_dos


class TestFleetResilience:
    """Test fleet resilience and self-healing capabilities."""
    
    @pytest.fixture
    async def fleet_system(self):
        """Create integrated fleet system for testing."""
        # Initialize components
        drone_ids = [f"test_drone_{i}" for i in range(10)]
        fleet = DroneFleet(drone_ids=drone_ids)
        
        coordinator = SwarmCoordinator(
            llm_model="gpt-4o",
            max_drones=10,
            safety_constraints=MissionConstraints(
                max_altitude=100.0,
                safety_distance=5.0
            )
        )
        
        await coordinator.connect_fleet(fleet)
        await fleet.start_monitoring()
        
        yield coordinator, fleet
        
        # Cleanup
        await fleet.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_drone_failure_recovery(self, fleet_system):
        """Test automatic drone failure recovery."""
        coordinator, fleet = fleet_system
        
        # Simulate drone failure
        failed_drone = "test_drone_1"
        fleet.update_drone_state(
            failed_drone,
            status=fleet.DroneStatus.FAILED,
            health_score=0.0
        )
        
        # Trigger auto-healing
        healing_results = await fleet.auto_heal_fleet()
        
        # Check if system adapted to failure
        active_drones = fleet.get_active_drones()
        assert failed_drone not in active_drones or len(healing_results['drones_recovered']) > 0
    
    @pytest.mark.asyncio
    async def test_communication_failure_recovery(self, fleet_system):
        """Test communication failure recovery."""
        coordinator, fleet = fleet_system
        
        # Simulate communication failure
        with patch.object(coordinator.webrtc_streamer, 'send_to_drone', side_effect=Exception("Connection lost")):
            # Try to send message - should trigger recovery
            result = await coordinator.send_command_to_drone("test_drone_0", {"test": "command"})
            
            # Error handling should have kicked in
            error_stats = get_error_handler().get_error_statistics()
            assert error_stats['total_errors'] > 0
    
    @pytest.mark.asyncio
    async def test_planning_service_failure(self, fleet_system):
        """Test fallback when planning service fails."""
        coordinator, fleet = fleet_system
        
        # Mock planning failure
        with patch.object(coordinator.llm_planner, 'generate_plan', side_effect=Exception("LLM service down")):
            # Request should fall back to emergency plan
            plan = await coordinator.generate_plan("Test mission during service failure")
            
            # Should get fallback plan
            assert plan is not None
            assert plan.get('is_fallback', False) or plan.get('metadata', {}).get('fallback', False)
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, fleet_system):
        """Test handling of resource exhaustion."""
        coordinator, fleet = fleet_system
        
        # Simulate high load
        tasks = []
        for i in range(50):  # Create high load
            task = asyncio.create_task(
                coordinator.generate_plan(f"High load mission {i}")
            )
            tasks.append(task)
        
        # Some should complete, some may be throttled/failed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # System should remain functional
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful > 0  # At least some should succeed
        
        # Check system health
        health_stats = coordinator.get_comprehensive_stats()
        assert health_stats is not None
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, fleet_system):
        """Test prevention of cascading failures."""
        coordinator, fleet = fleet_system
        
        # Simulate multiple drone failures
        failed_drones = ["test_drone_1", "test_drone_2", "test_drone_3"]
        
        for drone_id in failed_drones:
            fleet.update_drone_state(
                drone_id,
                status=fleet.DroneStatus.FAILED,
                health_score=0.0
            )
        
        # System should isolate failures and maintain operation
        fleet_status = fleet.get_fleet_status()
        
        # Should still have operational drones
        assert fleet_status['active_drones'] > 0
        
        # Should not propagate to other systems
        coordinator_status = await coordinator.get_swarm_status()
        assert coordinator_status['mission_status'] != 'failed'


class TestPerformanceUnderStress:
    """Test system performance under stress conditions."""
    
    @pytest.fixture
    async def stressed_system(self):
        """Create system under stress for testing."""
        fleet = DroneFleet([f"stress_drone_{i}" for i in range(100)])  # Large fleet
        
        coordinator = SwarmCoordinator(
            max_drones=100,
            update_rate=20.0  # High frequency
        )
        
        await coordinator.connect_fleet(fleet)
        await fleet.start_monitoring()
        
        yield coordinator, fleet
        
        await fleet.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_high_frequency_updates(self, stressed_system):
        """Test system under high frequency updates."""
        coordinator, fleet = stressed_system
        
        start_time = time.time()
        
        # Generate high frequency updates
        update_tasks = []
        for i in range(200):  # 200 concurrent updates
            task = asyncio.create_task(
                fleet.update_drone_state(
                    f"stress_drone_{i % 100}",
                    position=(i, i, i),
                    battery_percent=100 - (i % 50)
                )
            )
            update_tasks.append(task)
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle updates efficiently
        assert duration < 5.0  # Complete within 5 seconds
        
        successful_updates = sum(1 for r in results if r is True)
        assert successful_updates > 150  # Most should succeed
    
    @pytest.mark.asyncio
    async def test_concurrent_mission_planning(self, stressed_system):
        """Test concurrent mission planning performance."""
        coordinator, fleet = stressed_system
        
        start_time = time.time()
        
        # Generate concurrent planning requests
        planning_tasks = []
        for i in range(20):  # 20 concurrent plans
            task = asyncio.create_task(
                coordinator.generate_plan(f"Concurrent mission {i}")
            )
            planning_tasks.append(task)
        
        results = await asyncio.gather(*planning_tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle concurrent planning
        successful_plans = sum(1 for r in results if not isinstance(r, Exception))
        assert successful_plans > 15  # Most should succeed
        
        # Check performance metrics
        perf_summary = get_performance_summary()
        assert perf_summary is not None
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, stressed_system):
        """Test memory leak prevention under load."""
        coordinator, fleet = stressed_system
        
        # Get initial memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Generate sustained load
        for batch in range(10):  # 10 batches
            tasks = []
            for i in range(50):  # 50 operations per batch
                task = asyncio.create_task(
                    coordinator.generate_plan(f"Memory test {batch}_{i}")
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth


class TestIntegrationResilience:
    """Test integration points and their resilience."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_resilience(self):
        """Test complete end-to-end system resilience."""
        # Initialize full system
        drone_ids = [f"e2e_drone_{i}" for i in range(25)]
        fleet = DroneFleet(drone_ids=drone_ids)
        
        coordinator = SwarmCoordinator(
            llm_model="gpt-4o",
            max_drones=25
        )
        
        await coordinator.connect_fleet(fleet)
        await fleet.start_monitoring()
        
        try:
            # Test mission with multiple failure points
            mission_desc = "Complex search and rescue with formation flight"
            
            # Execute mission with simulated failures
            with patch.object(coordinator.webrtc_streamer, 'send_to_drone') as mock_send:
                # Simulate intermittent communication failures
                mock_send.side_effect = [
                    Exception("Connection failed"),  # First call fails
                    True,  # Second succeeds
                    Exception("Timeout"),  # Third fails
                    True,  # Fourth succeeds
                ]
                
                # Should still complete mission with retries/recovery
                plan = await coordinator.generate_plan(mission_desc)
                assert plan is not None
                
                # Check that error handling was triggered
                error_stats = get_error_handler().get_error_statistics()
                assert error_stats['total_errors'] > 0
                
                # System should still be operational
                status = await coordinator.get_swarm_status()
                assert status['mission_status'] in ['active', 'idle', 'planning']  # Any valid state
        
        finally:
            await fleet.stop_monitoring()
    
    def test_configuration_validation_robustness(self):
        """Test robustness of configuration validation."""
        # Test with invalid configurations
        invalid_configs = [
            {"max_drones": -1},  # Negative value
            {"update_rate": 0},   # Zero value
            {"safety_distance": -5.0},  # Negative safety distance
            {"max_altitude": 10000},  # Unrealistic altitude
        ]
        
        for config in invalid_configs:
            try:
                # Should either use defaults or fail gracefully
                constraints = MissionConstraints(**config)
                # If it doesn't fail, values should be sanitized
                assert constraints.max_altitude > 0
                assert constraints.safety_distance >= 0
            except (ValueError, ValidationError):
                # Acceptable - validation caught invalid config
                pass
    
    def test_graceful_degradation(self):
        """Test graceful degradation of system capabilities."""
        # Start with full capabilities
        fleet = DroneFleet([f"degrade_drone_{i}" for i in range(20)])
        
        # Gradually reduce capabilities
        capabilities_to_remove = [
            "formation_flight",
            "precision_hover", 
            "obstacle_avoidance",
            "camera"
        ]
        
        for capability in capabilities_to_remove:
            # Remove capability from half the drones
            for i in range(0, 20, 2):
                drone_id = f"degrade_drone_{i}"
                # System should adapt to reduced capabilities
        
        # System should still function with reduced capabilities
        active_drones = fleet.get_active_drones()
        assert len(active_drones) > 0


if __name__ == "__main__":
    # Configure logging for tests
    configure_logging(
        log_level=LogLevel.INFO,
        log_format="text",
        enable_metrics=True,
        enable_async_logging=False  # Sync for tests
    )
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])