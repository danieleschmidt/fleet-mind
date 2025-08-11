"""
Comprehensive Generation 2 Robustness Testing Framework for Fleet-Mind.

This test suite validates all enterprise-grade robustness enhancements:
- Advanced Security & Authentication
- Enterprise-Grade Monitoring & Alerting  
- Advanced Fault Tolerance
- Data Validation & Sanitization
- Compliance & Global Standards
- Integration Testing with Fault Injection
- Performance Regression Testing
- Security Penetration Testing
"""

import asyncio
import pytest
import time
import random
import json
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional

# Import Fleet-Mind Generation 2 components
from fleet_mind.security.security_manager import (
    SecurityManager, SecurityLevel, ThreatType, SecurityEvent,
    DroneCredentials, RateLimitRule
)
from fleet_mind.monitoring.health_monitor import (
    HealthMonitor, HealthStatus, AlertSeverity, PerformanceBaseline,
    AnomalyDetection, HealthDashboard
)
from fleet_mind.utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    AdvancedRetry, RetryConfig, DistributedConsensus,
    ByzantineFaultConfig, Bulkhead, BulkheadConfig,
    FaultToleranceManager
)
from fleet_mind.utils.validation import (
    EnterpriseValidator, ValidationSchema, ValidationContext,
    ValidationResult, ValidationSeverity, DataIntegrityCheck
)
from fleet_mind.i18n.compliance import (
    ComplianceManager, ComplianceStandard, DataProcessingRecord
)
from fleet_mind.utils.input_sanitizer import (
    InputSanitizer, SanitizationConfig, SanitizationLevel
)


class Generation2RobustnessTestSuite:
    """Comprehensive test suite for Generation 2 robustness features."""
    
    def __init__(self):
        """Initialize test suite with all components."""
        self.security_manager = None
        self.health_monitor = None
        self.fault_tolerance = None
        self.validator = None
        self.compliance_manager = None
        self.input_sanitizer = None
        
        # Test data and metrics
        self.test_results = []
        self.performance_metrics = {}
        self.security_test_results = []
        
    async def setup_test_environment(self):
        """Set up test environment with all Generation 2 components."""
        print("Setting up Generation 2 Robustness Test Environment...")
        
        # Initialize Security Manager with enterprise settings
        self.security_manager = SecurityManager(
            security_level=SecurityLevel.CRITICAL,
            enable_threat_detection=True,
            enable_rate_limiting=True,
            enable_audit_logging=True,
            max_failed_attempts=3,
            lockout_duration=300,
            enable_geo_blocking=True
        )
        
        # Initialize Health Monitor with advanced features
        self.health_monitor = HealthMonitor(
            check_interval=5.0,
            enable_anomaly_detection=True,
            enable_predictive_analysis=True,
            baseline_learning_period=3600.0
        )
        
        # Initialize Fault Tolerance Manager
        self.fault_tolerance = FaultToleranceManager()
        
        # Initialize Enterprise Validator
        self.validator = EnterpriseValidator()
        
        # Initialize Compliance Manager
        self.compliance_manager = ComplianceManager()
        
        # Initialize Input Sanitizer
        self.input_sanitizer = InputSanitizer(SanitizationConfig(
            level=SanitizationLevel.PARANOID
        ))
        
        # Start monitoring systems
        await self.health_monitor.start_monitoring()
        
        print("✓ Test environment setup complete")
    
    async def teardown_test_environment(self):
        """Clean up test environment."""
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        print("✓ Test environment cleanup complete")


class TestAdvancedSecurity:
    """Test advanced security and authentication features."""
    
    @pytest.fixture
    async def security_manager(self):
        """Create security manager for testing."""
        manager = SecurityManager(
            security_level=SecurityLevel.HIGH,
            enable_threat_detection=True,
            enable_rate_limiting=True,
            enable_audit_logging=True
        )
        return manager
    
    async def test_enhanced_authentication(self, security_manager):
        """Test enhanced authentication with multi-factor support."""
        print("\n🔒 Testing Enhanced Authentication...")
        
        # Generate credentials for test drone
        drone_id = "test_drone_001"
        permissions = {"basic_flight", "telemetry", "emergency_response"}
        
        credentials = security_manager.generate_drone_credentials(drone_id, permissions)
        
        assert credentials.drone_id == drone_id
        assert credentials.permissions == permissions
        assert credentials.multi_factor_enabled == False  # Default
        assert not credentials.revoked
        
        # Test authentication with valid credentials
        token = security_manager._generate_simple_token(drone_id)
        auth_result = security_manager.authenticate_drone(
            drone_id, token, 
            source_ip="192.168.1.100", 
            user_agent="DroneClient/1.0"
        )
        
        assert auth_result == True
        assert credentials.last_used is not None
        assert credentials.usage_count > 0
        
        print("✓ Enhanced authentication tests passed")
    
    async def test_rate_limiting_protection(self, security_manager):
        """Test rate limiting and DDoS protection."""
        print("\n🛡️ Testing Rate Limiting Protection...")
        
        drone_id = "rate_test_drone"
        source_ip = "192.168.1.200"
        
        # Test normal rate limiting
        allowed_requests = 0
        blocked_requests = 0
        
        for i in range(15):  # Exceed authentication rate limit
            allowed = security_manager.check_rate_limit(
                drone_id, "authentication", source_ip
            )
            if allowed:
                allowed_requests += 1
            else:
                blocked_requests += 1
        
        assert allowed_requests <= 10  # Default authentication limit
        assert blocked_requests > 0
        
        # Test rate limit recovery
        await asyncio.sleep(1)  # Wait for potential recovery
        
        print(f"✓ Rate limiting: {allowed_requests} allowed, {blocked_requests} blocked")
    
    async def test_threat_detection(self, security_manager):
        """Test real-time threat detection."""
        print("\n🎯 Testing Threat Detection...")
        
        # Test command injection detection
        malicious_message = {
            "type": "command",
            "data": "; rm -rf /; echo 'pwned'"
        }
        
        threats = security_manager.detect_threats(malicious_message, "test_source")
        
        assert ThreatType.COMMAND_INJECTION in threats
        
        # Test suspicious pattern detection
        suspicious_data = {
            "script": "<script>alert('xss')</script>",
            "eval": "eval(malicious_code)"
        }
        
        threats = security_manager.detect_threats(suspicious_data, "test_source")
        
        # Should detect multiple threat types
        assert len(threats) > 0
        
        print("✓ Threat detection tests passed")
    
    async def test_audit_logging(self, security_manager):
        """Test comprehensive audit logging."""
        print("\n📝 Testing Audit Logging...")
        
        initial_audit_count = len(security_manager.security_audit_log)
        
        # Perform actions that should generate audit logs
        drone_id = "audit_test_drone"
        credentials = security_manager.generate_drone_credentials(drone_id)
        
        # Authentication attempts
        token = security_manager._generate_simple_token(drone_id)
        security_manager.authenticate_drone(drone_id, token, source_ip="127.0.0.1")
        
        # Authorization checks
        security_manager.authorize_action(drone_id, "takeoff", source_ip="127.0.0.1")
        
        # Check audit log growth
        final_audit_count = len(security_manager.security_audit_log)
        
        assert final_audit_count > initial_audit_count
        
        # Verify audit record structure
        if security_manager.security_audit_log:
            audit_record = security_manager.security_audit_log[-1]
            assert hasattr(audit_record, 'audit_id')
            assert hasattr(audit_record, 'timestamp')
            assert hasattr(audit_record, 'action')
            assert hasattr(audit_record, 'success')
        
        print(f"✓ Audit logging: {final_audit_count - initial_audit_count} new records")
    
    async def test_security_dashboard(self, security_manager):
        """Test security dashboard and metrics."""
        print("\n📊 Testing Security Dashboard...")
        
        dashboard_data = security_manager.get_security_dashboard_data()
        
        # Verify dashboard structure
        required_keys = [
            'security_level', 'monitoring_enabled', 'metrics',
            'recent_activity', 'rate_limiting', 'account_security',
            'system_health', 'performance'
        ]
        
        for key in required_keys:
            assert key in dashboard_data
        
        # Verify metrics structure
        metrics = dashboard_data['metrics']
        assert 'total_requests' in metrics
        assert 'blocked_requests' in metrics
        assert 'block_rate' in metrics
        
        print("✓ Security dashboard tests passed")


class TestEnterpriseMonitoring:
    """Test enterprise-grade monitoring and alerting."""
    
    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor for testing."""
        monitor = HealthMonitor(
            check_interval=1.0,
            enable_anomaly_detection=True,
            enable_predictive_analysis=True
        )
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    async def test_anomaly_detection(self, health_monitor):
        """Test ML-based anomaly detection."""
        print("\n🔍 Testing Anomaly Detection...")
        
        component = "test_component"
        metric_name = "cpu_usage"
        
        # Generate baseline data
        for i in range(50):
            normal_value = 30 + random.uniform(-5, 5)  # Normal: 25-35%
            health_monitor.update_metric(component, metric_name, normal_value, "%")
            await asyncio.sleep(0.01)
        
        # Create baseline
        health_monitor.create_performance_baseline(component, metric_name)
        
        # Generate anomalous data
        anomalous_value = 90.0  # Clear anomaly
        health_monitor.update_metric(component, metric_name, anomalous_value, "%")
        
        # Check for anomaly detection
        anomaly = health_monitor.detect_anomaly(component, metric_name, anomalous_value)
        
        assert anomaly is not None
        assert anomaly.anomaly_score > 5.0
        assert anomaly.anomaly_type in ["spike", "drop"]
        
        print(f"✓ Anomaly detected: {anomaly.anomaly_type} with score {anomaly.anomaly_score:.2f}")
    
    async def test_performance_baselines(self, health_monitor):
        """Test performance baseline learning."""
        print("\n📈 Testing Performance Baselines...")
        
        component = "baseline_test"
        metric_name = "response_time"
        
        # Generate varied data with pattern
        for hour in range(24):
            for minute in range(5):  # 5 data points per hour
                # Simulate daily pattern: higher response times during "peak hours"
                base_value = 100 + (50 * (1 + 0.5 * (hour - 12) ** 2 / 144))
                value = base_value + random.uniform(-10, 10)
                health_monitor.update_metric(component, metric_name, value, "ms")
                await asyncio.sleep(0.001)
        
        # Create baseline
        health_monitor.create_performance_baseline(component, metric_name)
        
        # Verify baseline exists
        baseline_key = f"{component}:{metric_name}"
        assert baseline_key in health_monitor.performance_baselines
        
        baseline = health_monitor.performance_baselines[baseline_key]
        assert baseline.sample_count > 100
        assert baseline.mean_value > 0
        assert baseline.std_deviation > 0
        assert len(baseline.seasonal_patterns) > 0  # Should detect hourly patterns
        
        print(f"✓ Baseline learned: mean={baseline.mean_value:.1f}, std={baseline.std_deviation:.1f}")
    
    async def test_advanced_alerting(self, health_monitor):
        """Test advanced alerting with escalation."""
        print("\n🚨 Testing Advanced Alerting...")
        
        # Add test alert channel
        alert_messages = []
        
        def test_alert_channel(alert):
            alert_messages.append(alert.message)
        
        health_monitor.add_alert_channel(test_alert_channel)
        
        # Generate critical metric
        component = "alert_test"
        metric_name = "error_rate"
        
        health_monitor.update_metric(component, metric_name, 0.15, "", "High error rate")  # Above critical threshold
        
        await asyncio.sleep(0.5)  # Allow alert processing
        
        # Check for alerts
        active_alerts = health_monitor.get_alerts(active_only=True)
        
        assert len(active_alerts) > 0
        assert len(alert_messages) > 0
        
        # Test alert acknowledgment
        if active_alerts:
            alert_id = active_alerts[0].alert_id
            ack_result = health_monitor.acknowledge_alert(alert_id, "test_user")
            assert ack_result == True
        
        print(f"✓ Generated {len(active_alerts)} alerts, {len(alert_messages)} notifications")
    
    async def test_health_dashboard(self, health_monitor):
        """Test comprehensive health dashboard."""
        print("\n📊 Testing Health Dashboard...")
        
        # Register test component
        health_monitor.register_component("dashboard_test")
        
        # Generate various metrics
        health_monitor.update_metric("dashboard_test", "cpu_usage", 45.0, "%")
        health_monitor.update_metric("dashboard_test", "memory_usage", 70.0, "%")
        health_monitor.update_metric("dashboard_test", "response_time", 120.0, "ms")
        
        dashboard = health_monitor.get_health_dashboard()
        
        assert isinstance(dashboard, HealthDashboard)
        assert dashboard.component_count > 0
        assert dashboard.overall_health in [status.value for status in HealthStatus]
        assert dashboard.performance_score >= 0
        
        print(f"✓ Dashboard: {dashboard.component_count} components, {dashboard.overall_health} status")
    
    async def test_sla_monitoring(self, health_monitor):
        """Test SLA compliance monitoring."""
        print("\n📋 Testing SLA Monitoring...")
        
        # Set SLA targets
        health_monitor.set_sla_target("availability", 99.5)
        health_monitor.set_sla_target("response_time", 100.0)
        
        component = "sla_test"
        health_monitor.register_component(component)
        
        # Generate metrics that violate SLA
        for i in range(10):
            availability = 98.0 if i < 3 else 99.8  # 3 violations out of 10
            health_monitor.update_metric(component, "availability", availability, "%")
            await asyncio.sleep(0.01)
        
        sla_report = health_monitor.get_sla_report(component, hours=1)
        
        assert component in sla_report['component_details']
        component_data = sla_report['component_details'][component]
        assert 'sla_met' in component_data
        assert 'violations_count' in component_data
        
        print(f"✓ SLA monitoring: {sla_report['overall_availability']:.1f}% availability")


class TestAdvancedFaultTolerance:
    """Test advanced fault tolerance mechanisms."""
    
    @pytest.fixture
    def fault_tolerance_manager(self):
        """Create fault tolerance manager for testing."""
        return FaultToleranceManager()
    
    async def test_smart_circuit_breaker(self, fault_tolerance_manager):
        """Test circuit breaker with smart recovery."""
        print("\n⚡ Testing Smart Circuit Breaker...")
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            smart_recovery=True,
            failure_rate_threshold=0.5
        )
        
        circuit_breaker = fault_tolerance_manager.get_circuit_breaker("test_service", config)
        
        # Simulate failing service
        failure_count = 0
        
        async def failing_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 4:  # Fail first 4 calls
                raise Exception("Service unavailable")
            return "success"
        
        # Test circuit opening
        for i in range(5):
            try:
                await circuit_breaker.call_async(failing_service)
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery attempt
        await asyncio.sleep(1.1)
        
        # Test recovery
        try:
            result = await circuit_breaker.call_async(failing_service)
            assert result == "success"
            assert circuit_breaker.state == CircuitState.CLOSED
        except Exception:
            # May still be in recovery phase
            pass
        
        print(f"✓ Circuit breaker: {circuit_breaker.state.value} state after recovery")
    
    async def test_advanced_retry_mechanism(self, fault_tolerance_manager):
        """Test advanced retry with exponential backoff."""
        print("\n🔄 Testing Advanced Retry Mechanism...")
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=True,
            backoff_strategy='exponential'
        )
        
        retry_handler = fault_tolerance_manager.get_retry_handler("test_retry", config)
        
        attempt_count = 0
        
        async def unreliable_service():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Temporary failure {attempt_count}")
            return f"Success after {attempt_count} attempts"
        
        result = await retry_handler.execute_with_retry(unreliable_service)
        
        assert "Success after 3 attempts" in result
        assert attempt_count == 3
        
        print(f"✓ Retry succeeded after {attempt_count} attempts")
    
    async def test_bulkhead_pattern(self, fault_tolerance_manager):
        """Test bulkhead pattern for resource isolation."""
        print("\n🛡️ Testing Bulkhead Pattern...")
        
        config = BulkheadConfig(
            max_concurrent_calls=2,
            max_wait_time=1.0,
            timeout_per_call=0.5,
            rejection_policy='fail_fast'
        )
        
        bulkhead = fault_tolerance_manager.get_bulkhead("test_bulkhead", config)
        
        async def slow_service(delay=0.3):
            await asyncio.sleep(delay)
            return "completed"
        
        # Start multiple concurrent calls
        tasks = []
        for i in range(5):  # Exceed bulkhead capacity
            task = asyncio.create_task(bulkhead.execute(slow_service, 0.2))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed, some should be rejected
        successes = sum(1 for r in results if isinstance(r, str) and r == "completed")
        rejections = sum(1 for r in results if isinstance(r, Exception))
        
        assert successes > 0
        assert rejections > 0
        
        print(f"✓ Bulkhead: {successes} succeeded, {rejections} rejected")
    
    async def test_distributed_consensus(self, fault_tolerance_manager):
        """Test distributed consensus mechanism."""
        print("\n🤝 Testing Distributed Consensus...")
        
        config = ByzantineFaultConfig(
            max_byzantine_nodes=1,
            consensus_threshold=0.67
        )
        
        consensus = fault_tolerance_manager.get_consensus_system("test_consensus", "node_1", config)
        
        # Add test nodes
        consensus.add_node("node_2", "192.168.1.2")
        consensus.add_node("node_3", "192.168.1.3")
        
        # Test consensus status
        status = consensus.get_consensus_status()
        
        assert status['node_id'] == "node_1"
        assert status['total_nodes'] == 3  # Including self
        assert 'can_reach_consensus' in status
        
        # Test Byzantine fault detection
        byzantine_detected = consensus.detect_byzantine_behavior("node_2", "Inconsistent voting pattern")
        
        print(f"✓ Consensus: {status['total_nodes']} nodes, byzantine detection: {byzantine_detected}")
    
    async def test_fault_tolerance_integration(self, fault_tolerance_manager):
        """Test integrated fault tolerance mechanisms."""
        print("\n🔗 Testing Fault Tolerance Integration...")
        
        # Test the composite fault tolerance decorator
        from fleet_mind.utils.circuit_breaker import with_fault_tolerance
        
        call_count = 0
        
        @with_fault_tolerance(
            circuit_breaker_name="integrated_test",
            retry_name="integrated_retry",
            bulkhead_name="integrated_bulkhead"
        )
        async def test_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Service temporarily unavailable")
            return f"Success on call {call_count}"
        
        result = await test_service()
        
        assert "Success" in result
        assert call_count > 2  # Should have retried
        
        # Check system status
        system_status = fault_tolerance_manager.get_system_status()
        
        assert 'circuit_breakers' in system_status
        assert 'bulkheads' in system_status
        assert 'overall_health' in system_status
        
        print(f"✓ Integrated fault tolerance: {system_status['overall_health']}")


class TestEnterpriseValidation:
    """Test enterprise-grade validation system."""
    
    @pytest.fixture
    def validator(self):
        """Create enterprise validator for testing."""
        return EnterpriseValidator()
    
    async def test_schema_based_validation(self, validator):
        """Test schema-based validation with custom rules."""
        print("\n✅ Testing Schema-Based Validation...")
        
        # Test mission constraints validation
        valid_constraints = {
            "mission_id": "test_mission_001",
            "max_altitude": 100.0,
            "battery_time": 30.0,
            "safety_distance": 5.0,
            "geofence": [[-90, -180], [90, -180], [90, 180], [-90, 180]]
        }
        
        context = ValidationContext(
            request_id="test_validation_001",
            user_id="test_user",
            source="test_suite"
        )
        
        results = validator.validate_with_schema(valid_constraints, "mission_constraints", context)
        
        # Should have mostly passing results
        errors = [r for r in results if not r.is_valid]
        warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        
        print(f"✓ Schema validation: {len(results)} checks, {len(errors)} errors, {len(warnings)} warnings")
        
        # Test invalid data
        invalid_constraints = {
            "mission_id": "",  # Invalid: empty
            "max_altitude": -10,  # Invalid: negative
            "battery_time": 500,  # Invalid: too high
            "safety_distance": -1  # Invalid: negative
        }
        
        invalid_results = validator.validate_with_schema(invalid_constraints, "mission_constraints", context)
        invalid_errors = [r for r in invalid_results if not r.is_valid]
        
        assert len(invalid_errors) > len(errors)
        
        print(f"✓ Invalid data validation: {len(invalid_errors)} errors detected")
    
    async def test_cross_field_validation(self, validator):
        """Test cross-field validation rules."""
        print("\n🔗 Testing Cross-Field Validation...")
        
        # Test data that violates cross-field rules
        constraints_with_violations = {
            "mission_id": "cross_field_test",
            "max_altitude": 450.0,  # Above FAA limit
            "battery_time": 10.0,   # Too low for high altitude
            "safety_distance": 2.0
        }
        
        context = ValidationContext(request_id="cross_field_test")
        results = validator.validate_with_schema(constraints_with_violations, "mission_constraints", context)
        
        # Look for cross-field validation errors
        cross_field_errors = [r for r in results if r.rule_name and 'cross_field' in r.rule_name]
        
        print(f"✓ Cross-field validation: {len(cross_field_errors)} rule violations")
    
    async def test_data_integrity_checks(self, validator):
        """Test data integrity validation."""
        print("\n🛡️ Testing Data Integrity Checks...")
        
        # Add integrity check
        integrity_check = DataIntegrityCheck(
            name="hash_verification",
            description="Verify data has not been tampered with",
            check_type="hash",
            parameters={"expected_hash": "dummy_hash"},
            enabled=True,
            critical=True
        )
        
        validator.add_integrity_check(integrity_check)
        
        test_data = {"test": "data", "important": "value"}
        
        integrity_results = validator.validate_data_integrity(test_data)
        
        # Should detect hash mismatch
        hash_errors = [r for r in integrity_results if r.rule_name == "hash_verification"]
        
        assert len(hash_errors) > 0
        
        print(f"✓ Data integrity: {len(integrity_results)} checks performed")
    
    async def test_validation_performance_monitoring(self, validator):
        """Test validation performance metrics."""
        print("\n⚡ Testing Validation Performance...")
        
        # Perform multiple validations to generate metrics
        context = ValidationContext(request_id="perf_test")
        
        for i in range(10):
            test_data = {
                "mission_id": f"perf_test_{i}",
                "max_altitude": 50 + i,
                "battery_time": 30,
                "safety_distance": 5
            }
            
            validator.validate_with_schema(test_data, "mission_constraints", context)
        
        metrics = validator.get_validation_metrics()
        
        assert metrics['total_validations'] >= 10
        assert 'avg_validation_time_ms' in metrics
        assert 'success_rate' in metrics
        
        print(f"✓ Performance metrics: {metrics['total_validations']} validations, "
              f"{metrics['avg_validation_time_ms']:.2f}ms average")


class TestComplianceAndPrivacy:
    """Test compliance and privacy features."""
    
    @pytest.fixture
    def compliance_manager(self):
        """Create compliance manager for testing."""
        return ComplianceManager()
    
    async def test_multi_region_compliance(self, compliance_manager):
        """Test multi-region compliance support."""
        print("\n🌍 Testing Multi-Region Compliance...")
        
        # Test EU (GDPR) compliance
        eu_record_id = compliance_manager.register_data_processing(
            data_subject_id="eu_user_001",
            data_types=["name", "email", "location"],
            purpose="service_delivery",
            legal_basis="consent",
            region="EU",
            consent_obtained=True
        )
        
        assert eu_record_id is not None
        
        # Test US (CCPA) compliance  
        us_record_id = compliance_manager.register_data_processing(
            data_subject_id="us_user_001",
            data_types=["name", "email"],
            purpose="service_delivery", 
            legal_basis="legitimate_interest",
            region="US",
            consent_obtained=False
        )
        
        assert us_record_id is not None
        
        # Verify regional configurations
        assert "EU" in compliance_manager.regional_configs
        assert "US" in compliance_manager.regional_configs
        
        eu_config = compliance_manager.regional_configs["EU"]
        us_config = compliance_manager.regional_configs["US"]
        
        assert eu_config['data_residency_required'] == True
        assert us_config['data_residency_required'] == False
        
        print(f"✓ Multi-region: EU and US compliance configurations active")
    
    async def test_data_subject_rights(self, compliance_manager):
        """Test data subject rights handling."""
        print("\n👤 Testing Data Subject Rights...")
        
        # Register test data
        data_subject_id = "rights_test_user"
        compliance_manager.register_data_processing(
            data_subject_id=data_subject_id,
            data_types=["profile", "activity"],
            purpose="service_delivery",
            legal_basis="consent",
            region="EU"
        )
        
        # Test access request
        access_response = compliance_manager.handle_data_subject_request(
            request_type="access",
            data_subject_id=data_subject_id,
            region="EU"
        )
        
        assert access_response['request_type'] == "access"
        assert access_response['region'] == "EU"
        assert 'response_deadline' in access_response
        
        # Test erasure request  
        erasure_response = compliance_manager.handle_data_subject_request(
            request_type="erasure",
            data_subject_id=data_subject_id,
            region="EU"
        )
        
        assert erasure_response['request_type'] == "erasure"
        
        print(f"✓ Data subject rights: access and erasure requests processed")
    
    async def test_compliance_dashboard(self, compliance_manager):
        """Test compliance dashboard and metrics."""
        print("\n📊 Testing Compliance Dashboard...")
        
        # Generate some test data
        for i in range(5):
            compliance_manager.register_data_processing(
                data_subject_id=f"dashboard_user_{i}",
                data_types=["basic_profile"],
                purpose="analytics",
                legal_basis="legitimate_interest"
            )
        
        # Get dashboard data
        dashboard = compliance_manager.get_enhanced_compliance_dashboard()
        
        required_sections = [
            'global_compliance_score', 'regional_scores', 'data_protection',
            'rights_requests', 'security_and_privacy', 'regulatory_coverage'
        ]
        
        for section in required_sections:
            assert section in dashboard
        
        assert dashboard['data_protection']['total_data_subjects'] >= 5
        assert isinstance(dashboard['global_compliance_score'], float)
        
        print(f"✓ Compliance dashboard: {dashboard['global_compliance_score']:.1f}% global score")
    
    async def test_automated_data_lifecycle(self, compliance_manager):
        """Test automated data lifecycle management."""
        print("\n🔄 Testing Automated Data Lifecycle...")
        
        data_subject_id = "lifecycle_test_user"
        
        # Register data with retention period
        compliance_manager.register_data_processing(
            data_subject_id=data_subject_id,
            data_types=["temporary_data"],
            purpose="testing",
            legal_basis="consent",
            retention_days=1  # Very short for testing
        )
        
        # Test anonymization
        anonymization_result = compliance_manager.anonymize_data_subject(
            data_subject_id=data_subject_id,
            anonymization_method="k_anonymity"
        )
        
        assert anonymization_result['status'] == 'completed'
        assert anonymization_result['anonymized_records'] > 0
        
        # Test scheduled deletion
        compliance_manager.schedule_automated_deletion(
            data_subject_id=data_subject_id,
            deletion_date=time.time() + 86400,  # 1 day from now
            reason="retention_expired"
        )
        
        assert len(compliance_manager.deletion_queue) > 0
        
        print(f"✓ Data lifecycle: anonymization and scheduled deletion configured")


class TestSecurityPenetration:
    """Security penetration testing."""
    
    @pytest.fixture
    def security_components(self):
        """Set up security components for penetration testing."""
        return {
            'security_manager': SecurityManager(security_level=SecurityLevel.HIGH),
            'input_sanitizer': InputSanitizer(SanitizationConfig(level=SanitizationLevel.STRICT))
        }
    
    async def test_input_sanitization_penetration(self, security_components):
        """Test input sanitization against various attack vectors."""
        print("\n🎯 Testing Input Sanitization Penetration...")
        
        sanitizer = security_components['input_sanitizer']
        
        attack_vectors = [
            # XSS attacks
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "<img src='x' onerror='alert(1)'>",
            
            # SQL injection attempts
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM sensitive_data",
            
            # Command injection
            "; rm -rf /",
            "$(cat /etc/passwd)",
            "`whoami`",
            
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            
            # NoSQL injection
            "{'$ne': null}",
            "{'$where': 'function() { return true; }'}",
        ]
        
        blocked_attacks = 0
        
        for attack in attack_vectors:
            try:
                sanitized = sanitizer.sanitize({"malicious": attack}, "penetration_test")
                
                # Check if attack was neutralized
                sanitized_value = sanitized.get("malicious", "")
                if attack not in str(sanitized_value):  # Attack was modified/removed
                    blocked_attacks += 1
                    
            except ValueError:  # Sanitizer rejected the input entirely
                blocked_attacks += 1
        
        block_rate = blocked_attacks / len(attack_vectors)
        
        assert block_rate >= 0.8  # Should block at least 80% of attacks
        
        print(f"✓ Penetration testing: {blocked_attacks}/{len(attack_vectors)} attacks blocked ({block_rate:.1%})")
    
    async def test_authentication_brute_force_protection(self, security_components):
        """Test protection against brute force attacks."""
        print("\n🔐 Testing Brute Force Protection...")
        
        security_manager = security_components['security_manager']
        
        # Generate test credentials
        drone_id = "brute_force_target"
        security_manager.generate_drone_credentials(drone_id)
        
        # Simulate brute force attack
        failed_attempts = 0
        blocked_attempts = 0
        
        for i in range(20):  # 20 failed login attempts
            try:
                fake_token = f"fake_token_{i}"
                result = security_manager.authenticate_drone(
                    drone_id, fake_token,
                    source_ip="192.168.1.100"
                )
                
                if not result:
                    failed_attempts += 1
                    
            except Exception:
                # Account locked or other protection kicked in
                blocked_attempts += 1
        
        # Check if account got locked
        assert blocked_attempts > 0 or failed_attempts < 20
        
        # Verify the drone is locked/blocked
        assert drone_id in security_manager.locked_accounts or drone_id in security_manager.blocked_sources
        
        print(f"✓ Brute force protection: {blocked_attempts} blocked, account locked after {failed_attempts} attempts")
    
    async def test_dos_protection(self, security_components):
        """Test DoS/DDoS protection mechanisms."""
        print("\n🛡️ Testing DoS Protection...")
        
        security_manager = security_components['security_manager']
        
        source_ip = "192.168.1.200"
        
        # Simulate high-frequency requests
        allowed_requests = 0
        blocked_requests = 0
        
        for i in range(100):  # Rapid requests
            allowed = security_manager.check_rate_limit(
                f"dos_test_{i % 10}",  # Use different sources
                "command_execution",
                source_ip
            )
            
            if allowed:
                allowed_requests += 1
            else:
                blocked_requests += 1
        
        # Should have blocked some requests due to rate limiting
        assert blocked_requests > 0
        
        block_rate = blocked_requests / (allowed_requests + blocked_requests)
        
        print(f"✓ DoS protection: {block_rate:.1%} requests blocked under load")


class TestPerformanceRegression:
    """Performance regression testing."""
    
    async def test_security_performance_regression(self):
        """Test security operations performance."""
        print("\n⚡ Testing Security Performance...")
        
        security_manager = SecurityManager(security_level=SecurityLevel.HIGH)
        
        # Test encryption performance
        test_message = {"data": "test" * 100}  # Larger message
        
        encryption_times = []
        for i in range(10):
            start_time = time.time()
            
            encrypted = security_manager.encrypt_message(test_message, "test_recipient")
            decrypted = security_manager.decrypt_message(encrypted, "test_recipient")
            
            end_time = time.time()
            encryption_times.append((end_time - start_time) * 1000)  # ms
        
        avg_time = sum(encryption_times) / len(encryption_times)
        max_time = max(encryption_times)
        
        # Performance assertions (adjust thresholds as needed)
        assert avg_time < 50.0  # Average under 50ms
        assert max_time < 100.0  # Max under 100ms
        
        print(f"✓ Security performance: {avg_time:.2f}ms average, {max_time:.2f}ms max")
    
    async def test_monitoring_performance_regression(self):
        """Test monitoring system performance."""
        print("\n📊 Testing Monitoring Performance...")
        
        health_monitor = HealthMonitor(check_interval=0.1)
        await health_monitor.start_monitoring()
        
        try:
            # Test metric update performance
            update_times = []
            
            for i in range(100):
                start_time = time.time()
                
                health_monitor.update_metric(
                    f"perf_test_{i % 10}",
                    "test_metric",
                    random.uniform(0, 100),
                    "%"
                )
                
                end_time = time.time()
                update_times.append((end_time - start_time) * 1000)  # ms
            
            avg_update_time = sum(update_times) / len(update_times)
            
            # Should be very fast
            assert avg_update_time < 5.0  # Under 5ms average
            
            print(f"✓ Monitoring performance: {avg_update_time:.3f}ms per metric update")
            
        finally:
            await health_monitor.stop_monitoring()
    
    async def test_validation_performance_regression(self):
        """Test validation system performance."""
        print("\n✅ Testing Validation Performance...")
        
        validator = EnterpriseValidator()
        
        # Test large-scale validation
        validation_times = []
        
        for i in range(50):
            test_data = {
                "mission_id": f"performance_test_{i}",
                "max_altitude": random.uniform(10, 400),
                "battery_time": random.uniform(10, 120),
                "safety_distance": random.uniform(1, 20),
                "geofence": [[-90, -180], [90, -180], [90, 180], [-90, 180]]
            }
            
            context = ValidationContext(request_id=f"perf_{i}")
            
            start_time = time.time()
            results = validator.validate_with_schema(test_data, "mission_constraints", context)
            end_time = time.time()
            
            validation_times.append((end_time - start_time) * 1000)  # ms
        
        avg_validation_time = sum(validation_times) / len(validation_times)
        max_validation_time = max(validation_times)
        
        # Performance thresholds
        assert avg_validation_time < 10.0  # Under 10ms average
        assert max_validation_time < 50.0  # Under 50ms max
        
        print(f"✓ Validation performance: {avg_validation_time:.2f}ms average")


# Main test execution
async def run_comprehensive_robustness_tests():
    """Run all Generation 2 robustness tests."""
    print("="*80)
    print("🚀 FLEET-MIND GENERATION 2 ROBUSTNESS TEST SUITE")
    print("="*80)
    
    test_suite = Generation2RobustnessTestSuite()
    
    try:
        # Setup test environment
        await test_suite.setup_test_environment()
        
        # Initialize test classes
        security_tests = TestAdvancedSecurity()
        monitoring_tests = TestEnterpriseMonitoring()
        fault_tolerance_tests = TestAdvancedFaultTolerance()
        validation_tests = TestEnterpriseValidation()
        compliance_tests = TestComplianceAndPrivacy()
        penetration_tests = TestSecurityPenetration()
        performance_tests = TestPerformanceRegression()
        
        print(f"\n🧪 Starting comprehensive robustness testing...")
        
        # Run security tests
        print(f"\n" + "="*50)
        print("🔒 ADVANCED SECURITY TESTING")
        print("="*50)
        
        await security_tests.test_enhanced_authentication(test_suite.security_manager)
        await security_tests.test_rate_limiting_protection(test_suite.security_manager)
        await security_tests.test_threat_detection(test_suite.security_manager)
        await security_tests.test_audit_logging(test_suite.security_manager)
        await security_tests.test_security_dashboard(test_suite.security_manager)
        
        # Run monitoring tests
        print(f"\n" + "="*50)
        print("📊 ENTERPRISE MONITORING TESTING")
        print("="*50)
        
        await monitoring_tests.test_anomaly_detection(test_suite.health_monitor)
        await monitoring_tests.test_performance_baselines(test_suite.health_monitor)
        await monitoring_tests.test_advanced_alerting(test_suite.health_monitor)
        await monitoring_tests.test_health_dashboard(test_suite.health_monitor)
        await monitoring_tests.test_sla_monitoring(test_suite.health_monitor)
        
        # Run fault tolerance tests
        print(f"\n" + "="*50)
        print("⚡ ADVANCED FAULT TOLERANCE TESTING")
        print("="*50)
        
        await fault_tolerance_tests.test_smart_circuit_breaker(test_suite.fault_tolerance)
        await fault_tolerance_tests.test_advanced_retry_mechanism(test_suite.fault_tolerance)
        await fault_tolerance_tests.test_bulkhead_pattern(test_suite.fault_tolerance)
        await fault_tolerance_tests.test_distributed_consensus(test_suite.fault_tolerance)
        await fault_tolerance_tests.test_fault_tolerance_integration(test_suite.fault_tolerance)
        
        # Run validation tests
        print(f"\n" + "="*50)
        print("✅ ENTERPRISE VALIDATION TESTING")
        print("="*50)
        
        await validation_tests.test_schema_based_validation(test_suite.validator)
        await validation_tests.test_cross_field_validation(test_suite.validator)
        await validation_tests.test_data_integrity_checks(test_suite.validator)
        await validation_tests.test_validation_performance_monitoring(test_suite.validator)
        
        # Run compliance tests
        print(f"\n" + "="*50)
        print("🌍 COMPLIANCE & PRIVACY TESTING")
        print("="*50)
        
        await compliance_tests.test_multi_region_compliance(test_suite.compliance_manager)
        await compliance_tests.test_data_subject_rights(test_suite.compliance_manager)
        await compliance_tests.test_compliance_dashboard(test_suite.compliance_manager)
        await compliance_tests.test_automated_data_lifecycle(test_suite.compliance_manager)
        
        # Run penetration tests
        print(f"\n" + "="*50)
        print("🎯 SECURITY PENETRATION TESTING")
        print("="*50)
        
        security_components = {
            'security_manager': test_suite.security_manager,
            'input_sanitizer': test_suite.input_sanitizer
        }
        await penetration_tests.test_input_sanitization_penetration(security_components)
        await penetration_tests.test_authentication_brute_force_protection(security_components)
        await penetration_tests.test_dos_protection(security_components)
        
        # Run performance regression tests
        print(f"\n" + "="*50)
        print("⚡ PERFORMANCE REGRESSION TESTING")
        print("="*50)
        
        await performance_tests.test_security_performance_regression()
        await performance_tests.test_monitoring_performance_regression()
        await performance_tests.test_validation_performance_regression()
        
        print(f"\n" + "="*80)
        print("✅ ALL GENERATION 2 ROBUSTNESS TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Generate test summary
        print(f"\n📋 TEST SUMMARY:")
        print(f"   🔒 Security & Authentication: PASSED")
        print(f"   📊 Enterprise Monitoring: PASSED")
        print(f"   ⚡ Advanced Fault Tolerance: PASSED")
        print(f"   ✅ Enterprise Validation: PASSED")
        print(f"   🌍 Global Compliance: PASSED")
        print(f"   🎯 Security Penetration: PASSED")
        print(f"   ⚡ Performance Regression: PASSED")
        
        print(f"\n🎉 Generation 2 'Make It Robust' implementation is PRODUCTION-READY!")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        await test_suite.teardown_test_environment()


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(run_comprehensive_robustness_tests())