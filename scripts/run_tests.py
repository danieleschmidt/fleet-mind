#!/usr/bin/env python3
"""Simple test runner to validate Fleet-Mind implementation."""

import sys
import os
import importlib.util
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all main modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        'fleet_mind',
        'fleet_mind.coordination.swarm_coordinator',
        'fleet_mind.communication.webrtc_streamer',
        'fleet_mind.communication.latent_encoder',
        'fleet_mind.planning.llm_planner',
        'fleet_mind.fleet.drone_fleet',
        'fleet_mind.ros2_integration.fleet_manager_node',
        'fleet_mind.utils.logging',
        'fleet_mind.utils.validation',
        'fleet_mind.utils.security',
        'fleet_mind.utils.error_handling',
        'fleet_mind.optimization.performance_monitor',
        'fleet_mind.optimization.cache_manager',
        'fleet_mind.cli',
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {module_name}: {e}")
    
    print(f"\nImport test results: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints
        from fleet_mind.fleet.drone_fleet import DroneFleet
        from fleet_mind.communication.latent_encoder import LatentEncoder
        from fleet_mind.utils.validation import validate_mission_constraints
        from fleet_mind.utils.security import SecurityManager
        from fleet_mind.optimization.performance_monitor import PerformanceMonitor
        
        # Test SwarmCoordinator initialization
        constraints = MissionConstraints(max_altitude=100.0, safety_distance=5.0)
        coordinator = SwarmCoordinator(
            llm_model="test-model",
            latent_dim=64,
            safety_constraints=constraints
        )
        print("✓ SwarmCoordinator initialization")
        
        # Test DroneFleet initialization
        fleet = DroneFleet(
            drone_ids=["drone_1", "drone_2", "drone_3"],
            communication_protocol="webrtc"
        )
        print("✓ DroneFleet initialization")
        
        # Test LatentEncoder
        encoder = LatentEncoder(input_dim=512, latent_dim=64)
        encoded = encoder.encode("test action sequence")
        print("✓ LatentEncoder functionality")
        
        # Test validation
        test_constraints = {
            'max_altitude': 120.0,
            'battery_time': 30.0,
            'safety_distance': 5.0,
        }
        results = validate_mission_constraints(test_constraints)
        print("✓ Validation system")
        
        # Test SecurityManager
        security = SecurityManager()
        encrypted = security.encrypt_message("test message")
        decrypted = security.decrypt_message(encrypted)
        print("✓ Security system")
        
        # Test PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.record_latency("test_operation", 50.0)
        print("✓ Performance monitoring")
        
        print("\nBasic functionality tests: ALL PASSED")
        return True
        
    except Exception as e:
        print(f"\nBasic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration and data validation."""
    print("\nTesting configuration validation...")
    
    try:
        from fleet_mind.utils.validation import FleetValidator
        
        validator = FleetValidator()
        
        # Test mission constraints validation
        valid_constraints = {
            'max_altitude': 100.0,
            'battery_time': 25.0,
            'safety_distance': 5.0,
        }
        
        results = validator.validate_mission_constraints(valid_constraints)
        has_errors = any(not r.is_valid for r in results if r.severity.value in ['error', 'critical'])
        
        if not has_errors:
            print("✓ Mission constraints validation")
        else:
            print("✗ Mission constraints validation failed")
            return False
        
        # Test drone state validation
        valid_state = {
            'drone_id': 'test_drone_1',
            'position': (10.0, 20.0, 50.0),
            'velocity': (2.0, 1.0, 0.0),
            'orientation': (0.1, 0.05, 1.57),
            'battery_percent': 85.0,
            'health_score': 0.95,
        }
        
        results = validator.validate_drone_state(valid_state)
        has_errors = any(not r.is_valid for r in results if r.severity.value in ['error', 'critical'])
        
        if not has_errors:
            print("✓ Drone state validation")
        else:
            print("✗ Drone state validation failed")
            return False
        
        # Test fleet configuration validation
        valid_config = {
            'num_drones': 10,
            'min_separation': 5.0,
            'communication_range': 1000.0,
        }
        
        results = validator.validate_fleet_configuration(valid_config)
        has_errors = any(not r.is_valid for r in results if r.severity.value in ['error', 'critical'])
        
        if not has_errors:
            print("✓ Fleet configuration validation")
        else:
            print("✗ Fleet configuration validation failed")
            return False
        
        print("\nConfiguration validation tests: ALL PASSED")
        return True
        
    except Exception as e:
        print(f"\nConfiguration validation test failed: {e}")
        traceback.print_exc()
        return False

def test_security_features():
    """Test security features."""
    print("\nTesting security features...")
    
    try:
        from fleet_mind.utils.security import SecurityManager, SecurityContext, ThreatDetection
        
        security = SecurityManager()
        
        # Test encryption/decryption
        original_message = "This is a test message for encryption"
        encrypted = security.encrypt_message(original_message)
        decrypted = security.decrypt_message(encrypted)
        
        if decrypted.decode('utf-8') == original_message:
            print("✓ Message encryption/decryption")
        else:
            print("✗ Message encryption/decryption failed")
            return False
        
        # Test digital signatures
        message_bytes = original_message.encode('utf-8')
        signature = security.sign_message(message_bytes)
        is_valid = security.verify_signature(message_bytes, signature)
        
        if is_valid:
            print("✓ Digital signatures")
        else:
            print("✗ Digital signature validation failed")
            return False
        
        # Test JWT tokens
        payload = {'user_id': 'test_user', 'role': 'operator'}
        token = security.create_secure_token(payload)
        decoded = security.verify_token(token)
        
        if decoded and decoded['user_id'] == 'test_user':
            print("✓ JWT token creation/verification")
        else:
            print("✗ JWT token validation failed")
            return False
        
        # Test threat detection
        test_data = {
            'user_input': 'SELECT * FROM users WHERE id = 1',
            'command': 'normal_command'
        }
        
        threats = security.detect_threats(test_data)
        
        if isinstance(threats, list):
            print("✓ Threat detection system")
        else:
            print("✗ Threat detection failed")
            return False
        
        print("\nSecurity feature tests: ALL PASSED")
        return True
        
    except Exception as e:
        print(f"\nSecurity test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\nTesting performance monitoring...")
    
    try:
        from fleet_mind.optimization.performance_monitor import PerformanceMonitor, MetricType, PerformanceMetric
        
        monitor = PerformanceMonitor(history_size=100, enable_auto_optimization=False)
        
        # Test metric recording
        monitor.record_latency("api_call", 45.5, "web_service")
        monitor.record_throughput("requests", 120.0, "api_server")
        monitor.record_error_rate("authentication", 1.2, "auth_service")
        
        # Check metrics were recorded
        current_metrics = monitor.get_current_metrics()
        
        if len(current_metrics) >= 3:
            print("✓ Metric recording")
        else:
            print("✗ Metric recording failed")
            return False
        
        # Test statistics calculation
        for i in range(10):
            monitor.record_latency("test_op", float(i * 10))
        
        stats = monitor.get_metric_statistics("latency.test_op")
        
        if stats and 'mean' in stats and 'p95' in stats:
            print("✓ Statistical analysis")
        else:
            print("✗ Statistical analysis failed")
            return False
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        
        if 'current_metrics' in summary and 'analysis' in summary:
            print("✓ Performance summary generation")
        else:
            print("✗ Performance summary failed")
            return False
        
        print("\nPerformance monitoring tests: ALL PASSED")
        return True
        
    except Exception as e:
        print(f"\nPerformance monitoring test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("Fleet-Mind Implementation Validation")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Validation", test_configuration_validation),
        ("Security Features", test_security_features),
        ("Performance Monitoring", test_performance_monitoring),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            print(f"✗ {test_name}: EXCEPTION - {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Fleet-Mind implementation is valid!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Review implementation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)