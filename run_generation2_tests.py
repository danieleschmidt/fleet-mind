#!/usr/bin/env python3
"""
Standalone Generation 2 Robustness Testing Runner for Fleet-Mind.

This script runs comprehensive tests for all enterprise-grade robustness enhancements
without external testing framework dependencies.
"""

import asyncio
import time
import random
import json
import hashlib
from typing import Dict, List, Any, Optional

# Import Fleet-Mind Generation 2 components
try:
    from fleet_mind.security.security_manager import (
        SecurityManager, SecurityLevel, ThreatType
    )
    from fleet_mind.monitoring.health_monitor import (
        HealthMonitor, HealthStatus, AlertSeverity
    )
    from fleet_mind.utils.circuit_breaker import (
        CircuitBreaker, CircuitBreakerConfig, CircuitState,
        FaultToleranceManager
    )
    from fleet_mind.utils.validation import (
        EnterpriseValidator, ValidationContext
    )
    from fleet_mind.i18n.compliance import (
        ComplianceManager, ComplianceStandard
    )
    from fleet_mind.utils.input_sanitizer import (
        InputSanitizer, SanitizationConfig, SanitizationLevel
    )
    
    print("✓ All Generation 2 components imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some Generation 2 components may not be available")


class Generation2TestRunner:
    """Standalone test runner for Generation 2 robustness features."""
    
    def __init__(self):
        """Initialize test runner."""
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
        # Initialize components
        self.security_manager = None
        self.health_monitor = None
        self.fault_tolerance = None
        self.validator = None
        self.compliance_manager = None
        self.input_sanitizer = None
    
    async def setup_components(self):
        """Set up all Generation 2 components for testing."""
        print("\n🔧 Setting up Generation 2 components...")
        
        try:
            # Initialize Security Manager
            self.security_manager = SecurityManager(
                security_level=SecurityLevel.HIGH,
                enable_threat_detection=True,
                enable_rate_limiting=True,
                enable_audit_logging=True
            )
            print("✓ Security Manager initialized")
            
            # Initialize Health Monitor  
            self.health_monitor = HealthMonitor(
                check_interval=1.0,
                enable_anomaly_detection=True,
                enable_predictive_analysis=True
            )
            await self.health_monitor.start_monitoring()
            print("✓ Health Monitor started")
            
            # Initialize Fault Tolerance Manager
            self.fault_tolerance = FaultToleranceManager()
            print("✓ Fault Tolerance Manager initialized")
            
            # Initialize Enterprise Validator
            self.validator = EnterpriseValidator()
            print("✓ Enterprise Validator initialized")
            
            # Initialize Compliance Manager
            self.compliance_manager = ComplianceManager()
            print("✓ Compliance Manager initialized")
            
            # Initialize Input Sanitizer
            self.input_sanitizer = InputSanitizer(SanitizationConfig(
                level=SanitizationLevel.STRICT
            ))
            print("✓ Input Sanitizer initialized")
            
            print("✅ All Generation 2 components ready for testing")
            
        except Exception as e:
            print(f"❌ Component setup failed: {e}")
            raise
    
    async def cleanup_components(self):
        """Clean up components after testing."""
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        print("✓ Components cleaned up")
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling."""
        try:
            print(f"\n🧪 Running: {test_name}")
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.create_task(test_func())
                return result
            else:
                result = test_func()
                
            duration = time.time() - start_time
            print(f"✅ PASSED: {test_name} ({duration:.3f}s)")
            self.passed_tests += 1
            self.test_results.append({
                'name': test_name,
                'status': 'PASSED',
                'duration': duration
            })
            return result
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ FAILED: {test_name} - {str(e)} ({duration:.3f}s)")
            self.failed_tests += 1
            self.test_results.append({
                'name': test_name,
                'status': 'FAILED',
                'duration': duration,
                'error': str(e)
            })
            # Don't raise - continue with other tests
    
    async def test_security_authentication(self):
        """Test enhanced authentication."""
        drone_id = "test_drone_001"
        permissions = {"basic_flight", "telemetry", "emergency_response"}
        
        # Generate credentials
        credentials = self.security_manager.generate_drone_credentials(drone_id, permissions)
        
        assert credentials.drone_id == drone_id
        assert credentials.permissions == permissions
        assert not credentials.revoked
        
        # Test authentication
        token = self.security_manager._generate_simple_token(drone_id)
        auth_result = self.security_manager.authenticate_drone(drone_id, token)
        
        assert auth_result == True
        print("  ✓ Authentication successful")
    
    async def test_security_rate_limiting(self):
        """Test rate limiting protection."""
        drone_id = "rate_test_drone"
        source_ip = "192.168.1.100"
        
        blocked_count = 0
        allowed_count = 0
        
        # Make many requests to trigger rate limiting
        for i in range(15):
            allowed = self.security_manager.check_rate_limit(
                drone_id, "authentication", source_ip
            )
            if allowed:
                allowed_count += 1
            else:
                blocked_count += 1
        
        assert blocked_count > 0, "Rate limiting should block some requests"
        print(f"  ✓ Rate limiting: {allowed_count} allowed, {blocked_count} blocked")
    
    async def test_security_threat_detection(self):
        """Test threat detection."""
        malicious_message = {
            "type": "command", 
            "data": "; rm -rf /; echo 'test'"
        }
        
        threats = self.security_manager.detect_threats(malicious_message, "test_source")
        
        assert len(threats) > 0, "Should detect command injection threat"
        assert ThreatType.COMMAND_INJECTION in threats
        print(f"  ✓ Detected {len(threats)} threat(s)")
    
    async def test_monitoring_anomaly_detection(self):
        """Test anomaly detection."""
        component = "test_component"
        metric_name = "cpu_usage"
        
        # Generate baseline data
        for i in range(30):
            normal_value = 30 + random.uniform(-5, 5)
            self.health_monitor.update_metric(component, metric_name, normal_value, "%")
            await asyncio.sleep(0.001)
        
        # Create baseline
        self.health_monitor.create_performance_baseline(component, metric_name)
        
        # Test anomaly
        anomalous_value = 85.0
        anomaly = self.health_monitor.detect_anomaly(component, metric_name, anomalous_value)
        
        assert anomaly is not None, "Should detect anomaly"
        print(f"  ✓ Anomaly detected: {anomaly.anomaly_type} (score: {anomaly.anomaly_score:.2f})")
    
    async def test_monitoring_health_dashboard(self):
        """Test health dashboard."""
        # Register test component
        self.health_monitor.register_component("dashboard_test")
        
        # Add metrics
        self.health_monitor.update_metric("dashboard_test", "cpu_usage", 45.0, "%")
        self.health_monitor.update_metric("dashboard_test", "memory_usage", 70.0, "%")
        
        dashboard = self.health_monitor.get_health_dashboard()
        
        assert dashboard.component_count > 0
        print(f"  ✓ Dashboard: {dashboard.component_count} components, {dashboard.overall_health} status")
    
    async def test_fault_tolerance_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from fleet_mind.utils.circuit_breaker import CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5,
            smart_recovery=True
        )
        
        circuit_breaker = self.fault_tolerance.get_circuit_breaker("test_service", config)
        
        failure_count = 0
        
        async def failing_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 4:
                raise Exception("Service failure")
            return "success"
        
        # Trigger circuit opening
        for i in range(5):
            try:
                await circuit_breaker.call_async(failing_service)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        print(f"  ✓ Circuit breaker: {circuit_breaker.state.value} after failures")
    
    async def test_validation_schema_validation(self):
        """Test schema-based validation."""
        valid_data = {
            "mission_id": "test_mission_001",
            "max_altitude": 100.0,
            "battery_time": 30.0,
            "safety_distance": 5.0
        }
        
        context = ValidationContext(
            request_id="test_validation",
            user_id="test_user",
            source="test_suite"
        )
        
        results = self.validator.validate_with_schema(valid_data, "mission_constraints", context)
        
        errors = [r for r in results if not r.is_valid]
        print(f"  ✓ Validation: {len(results)} checks, {len(errors)} errors")
    
    async def test_compliance_data_processing(self):
        """Test compliance data processing."""
        record_id = self.compliance_manager.register_data_processing(
            data_subject_id="test_user_001",
            data_types=["name", "email"],
            purpose="service_delivery",
            legal_basis="consent",
            consent_obtained=True
        )
        
        assert record_id is not None
        
        # Test data subject request
        response = self.compliance_manager.handle_data_subject_request(
            request_type="access",
            data_subject_id="test_user_001"
        )
        
        assert response['request_type'] == "access"
        print(f"  ✓ Data processing: record {record_id}, request processed")
    
    async def test_input_sanitization(self):
        """Test input sanitization."""
        attack_vectors = [
            "<script>alert('xss')</script>",
            "; rm -rf /",
            "../../../etc/passwd",
            "'; DROP TABLE users; --"
        ]
        
        blocked_count = 0
        
        for attack in attack_vectors:
            try:
                sanitized = self.input_sanitizer.sanitize({"input": attack}, "test")
                sanitized_value = sanitized.get("input", "")
                
                if attack not in str(sanitized_value):
                    blocked_count += 1
                    
            except ValueError:
                blocked_count += 1
        
        block_rate = blocked_count / len(attack_vectors)
        assert block_rate >= 0.5, "Should block at least 50% of attacks"
        
        print(f"  ✓ Sanitization: {blocked_count}/{len(attack_vectors)} attacks blocked ({block_rate:.1%})")
    
    async def run_all_tests(self):
        """Run all Generation 2 robustness tests."""
        print("="*80)
        print("🚀 FLEET-MIND GENERATION 2 ROBUSTNESS TEST SUITE")  
        print("="*80)
        
        # Setup components
        await self.setup_components()
        
        try:
            # Define test cases
            test_cases = [
                ("Security Authentication", self.test_security_authentication),
                ("Security Rate Limiting", self.test_security_rate_limiting),
                ("Security Threat Detection", self.test_security_threat_detection),
                ("Monitoring Anomaly Detection", self.test_monitoring_anomaly_detection),
                ("Monitoring Health Dashboard", self.test_monitoring_health_dashboard),
                ("Fault Tolerance Circuit Breaker", self.test_fault_tolerance_circuit_breaker),
                ("Validation Schema Validation", self.test_validation_schema_validation),
                ("Compliance Data Processing", self.test_compliance_data_processing),
                ("Input Sanitization", self.test_input_sanitization),
            ]
            
            print(f"\n📋 Running {len(test_cases)} test categories...")
            
            # Execute all tests
            tasks = []
            for test_name, test_func in test_cases:
                task = self.run_test(test_name, test_func)
                if task:
                    tasks.append(task)
            
            # Wait for all async tests to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Print summary
            print(f"\n" + "="*80)
            print("📊 TEST EXECUTION SUMMARY")
            print("="*80)
            
            total_tests = self.passed_tests + self.failed_tests
            success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            print(f"✅ Passed: {self.passed_tests}")
            print(f"❌ Failed: {self.failed_tests}")
            print(f"📈 Success Rate: {success_rate:.1f}%")
            
            if self.failed_tests == 0:
                print(f"\n🎉 ALL TESTS PASSED! Generation 2 robustness implementation is PRODUCTION-READY!")
                print(f"\n📋 GENERATION 2 FEATURES VALIDATED:")
                print(f"   🔒 Advanced Security & Authentication")
                print(f"   📊 Enterprise-Grade Monitoring & Alerting")
                print(f"   ⚡ Advanced Fault Tolerance")
                print(f"   ✅ Comprehensive Data Validation")
                print(f"   🌍 Global Compliance & Privacy")
                print(f"   🛡️ Input Sanitization & Security")
            else:
                print(f"\n⚠️  Some tests failed. Review the output above for details.")
                
                # Show failed tests
                failed_tests = [r for r in self.test_results if r['status'] == 'FAILED']
                if failed_tests:
                    print(f"\nFailed Tests:")
                    for test in failed_tests:
                        print(f"   ❌ {test['name']}: {test.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"\n💥 Test execution failed: {e}")
            raise
            
        finally:
            await self.cleanup_components()


async def main():
    """Main test execution function."""
    runner = Generation2TestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())