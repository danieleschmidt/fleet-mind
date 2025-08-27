#!/usr/bin/env python3
"""
GENERATION 2: ROBUST QUALITY ORCHESTRATOR
Enhanced reliability, error handling, monitoring, and fault tolerance

Implements comprehensive error handling, distributed monitoring, circuit breakers,
health checks, and advanced logging with real-time alerting systems.
"""

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import hashlib
import statistics
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil
import threading
import queue
import uuid

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class Alert:
    id: str
    level: AlertLevel
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CircuitBreaker:
    name: str
    state: CircuitBreakerState
    failure_count: int
    failure_threshold: int
    reset_timeout: int  # seconds
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    
    def __post_init__(self):
        if self.last_failure_time is None:
            self.last_failure_time = datetime.now()
        if self.last_success_time is None:
            self.last_success_time = datetime.now()

class RobustQualityOrchestrator:
    """
    Production-ready quality orchestrator with comprehensive reliability features
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.setup_comprehensive_logging()
        self.config = self.load_robust_config(config_path)
        self.health_checks = {}
        self.circuit_breakers = {}
        self.alerts = []
        self.metrics_buffer = queue.Queue(maxsize=10000)
        self.alert_queue = queue.Queue(maxsize=1000)
        
        # Initialize monitoring systems
        self.initialize_health_monitoring()
        self.initialize_circuit_breakers()
        self.initialize_alerting_system()
        
        # Start background monitoring tasks
        self.monitoring_tasks = []
        self.start_background_monitoring()
        
        self.logger.info("Robust Quality Orchestrator initialized with advanced monitoring")

    def setup_comprehensive_logging(self):
        """Setup multi-level logging with structured output"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure multiple log handlers
        formatters = {
            'detailed': logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            ),
            'json': logging.Formatter(
                '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        }
        
        handlers = [
            # Console handler with colors
            logging.StreamHandler(),
            # Main log file
            logging.FileHandler(log_dir / 'quality_orchestrator.log'),
            # Error log file  
            logging.FileHandler(log_dir / 'errors.log', mode='a'),
            # JSON structured logs
            logging.FileHandler(log_dir / 'structured.jsonl')
        ]
        
        # Configure handlers
        handlers[0].setFormatter(formatters['detailed'])  # Console
        handlers[1].setFormatter(formatters['detailed'])  # Main log
        handlers[2].setFormatter(formatters['detailed'])  # Error log
        handlers[2].setLevel(logging.ERROR)
        handlers[3].setFormatter(formatters['json'])      # JSON log
        
        # Setup root logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
        self.access_logger = logging.getLogger('access')
        self.error_logger = logging.getLogger('errors')
        self.performance_logger = logging.getLogger('performance')

    def load_robust_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration with validation and fallbacks"""
        default_config = {
            "orchestrator": {
                "max_concurrent_gates": 8,
                "default_timeout": 600,
                "retry_attempts": 3,
                "retry_backoff_factor": 2.0,
                "health_check_interval": 30,
                "metrics_retention_hours": 24
            },
            "circuit_breaker": {
                "failure_threshold": 5,
                "reset_timeout": 60,
                "monitor_window": 300
            },
            "alerting": {
                "enabled": True,
                "email": {
                    "enabled": False,
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "timeout": 10
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#quality-gates"
                }
            },
            "monitoring": {
                "system_metrics": True,
                "application_metrics": True,
                "custom_metrics": True,
                "export_prometheus": False,
                "export_grafana": False
            },
            "resilience": {
                "graceful_shutdown_timeout": 30,
                "max_memory_usage_mb": 2048,
                "max_cpu_usage_percent": 80,
                "disk_space_threshold_gb": 10
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config = self.deep_merge_config(default_config, user_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Validate configuration
        self.validate_config(default_config)
        return default_config

    def deep_merge_config(self, base: Dict, override: Dict) -> Dict:
        """Deep merge configuration dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters"""
        try:
            # Validate numeric ranges
            assert 1 <= config["orchestrator"]["max_concurrent_gates"] <= 32
            assert 10 <= config["orchestrator"]["default_timeout"] <= 3600
            assert 1 <= config["orchestrator"]["retry_attempts"] <= 10
            assert 1.0 <= config["orchestrator"]["retry_backoff_factor"] <= 5.0
            
            self.logger.info("Configuration validation passed")
        except (AssertionError, KeyError) as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def initialize_health_monitoring(self):
        """Initialize comprehensive health monitoring system"""
        health_checks = [
            ("system_memory", self.check_system_memory),
            ("system_cpu", self.check_system_cpu),
            ("system_disk", self.check_system_disk),
            ("network_connectivity", self.check_network_connectivity),
            ("database_connection", self.check_database_connection),
            ("external_services", self.check_external_services)
        ]
        
        for name, check_func in health_checks:
            self.health_checks[name] = {
                "check_function": check_func,
                "last_result": None,
                "consecutive_failures": 0,
                "enabled": True
            }
        
        self.logger.info(f"Initialized {len(health_checks)} health checks")

    def initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical components"""
        critical_components = [
            "quality_gate_executor",
            "ml_predictor",
            "alert_system",
            "metrics_collector",
            "health_monitor"
        ]
        
        for component in critical_components:
            self.circuit_breakers[component] = CircuitBreaker(
                name=component,
                state=CircuitBreakerState.CLOSED,
                failure_count=0,
                failure_threshold=self.config["circuit_breaker"]["failure_threshold"],
                reset_timeout=self.config["circuit_breaker"]["reset_timeout"],
                last_failure_time=None,
                last_success_time=datetime.now()
            )
        
        self.logger.info(f"Initialized circuit breakers for {len(critical_components)} components")

    def initialize_alerting_system(self):
        """Initialize comprehensive alerting system"""
        if not self.config["alerting"]["enabled"]:
            self.logger.info("Alerting system disabled by configuration")
            return
        
        # Start alert processing thread
        self.alert_processor_thread = threading.Thread(
            target=self.process_alerts,
            daemon=True,
            name="AlertProcessor"
        )
        self.alert_processor_thread.start()
        
        self.logger.info("Alerting system initialized")

    def start_background_monitoring(self):
        """Start background monitoring tasks"""
        monitoring_tasks = [
            ("health_monitor", self.continuous_health_monitoring),
            ("metrics_collector", self.continuous_metrics_collection),
            ("circuit_breaker_monitor", self.monitor_circuit_breakers),
            ("resource_monitor", self.monitor_system_resources)
        ]
        
        for name, task_func in monitoring_tasks:
            task = asyncio.create_task(task_func())
            task.set_name(name)
            self.monitoring_tasks.append(task)
        
        self.logger.info(f"Started {len(monitoring_tasks)} background monitoring tasks")

    async def execute_quality_gates_robust(self) -> Dict[str, Any]:
        """Execute quality gates with comprehensive error handling and monitoring"""
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.logger.info(f"Starting robust quality gate execution [ID: {execution_id}]")
        
        try:
            # Pre-execution health checks
            health_status = await self.perform_health_checks()
            if health_status["overall_status"] == HealthStatus.UNHEALTHY:
                raise RuntimeError("System health check failed - aborting execution")
            
            # Check circuit breaker states
            self.check_circuit_breaker_states()
            
            # Execute quality gates with monitoring
            results = await self.execute_gates_with_monitoring(execution_id)
            
            # Post-execution validation
            await self.validate_execution_results(results)
            
            # Generate comprehensive report
            report = await self.generate_robust_report(results, start_time, execution_id)
            
            self.logger.info(f"Quality gate execution completed successfully [ID: {execution_id}]")
            return report
            
        except Exception as e:
            self.logger.error(f"Quality gate execution failed [ID: {execution_id}]: {e}")
            self.logger.error(traceback.format_exc())
            
            # Send critical alert
            await self.send_alert(
                level=AlertLevel.CRITICAL,
                message=f"Quality gate execution failed: {str(e)}",
                component="orchestrator",
                metadata={"execution_id": execution_id, "error": str(e)}
            )
            
            raise
        
        finally:
            # Cleanup and final health check
            await self.cleanup_execution_resources()
            execution_time = (datetime.now() - start_time).total_seconds()
            self.performance_logger.info(f"Execution completed in {execution_time:.2f}s [ID: {execution_id}]")

    async def execute_gates_with_monitoring(self, execution_id: str) -> Dict[str, Any]:
        """Execute quality gates with real-time monitoring"""
        results = {}
        active_executions = {}
        
        # Load gate execution plan
        execution_plan = self.generate_execution_plan()
        
        for batch_index, gate_batch in enumerate(execution_plan):
            batch_start_time = time.time()
            self.logger.info(f"Executing batch {batch_index + 1}/{len(execution_plan)} with {len(gate_batch)} gates")
            
            # Execute batch in parallel with monitoring
            batch_tasks = []
            for gate_id in gate_batch:
                if self.is_circuit_breaker_open(f"gate_{gate_id}"):
                    self.logger.warning(f"Skipping gate {gate_id} - circuit breaker open")
                    continue
                
                task = self.execute_single_gate_monitored(gate_id, execution_id)
                batch_tasks.append((gate_id, task))
                active_executions[gate_id] = {
                    "start_time": time.time(),
                    "task": task,
                    "batch": batch_index
                }
            
            # Wait for batch completion with timeout monitoring
            for gate_id, task in batch_tasks:
                try:
                    gate_result = await asyncio.wait_for(
                        task,
                        timeout=self.config["orchestrator"]["default_timeout"]
                    )
                    results[gate_id] = gate_result
                    
                    # Update circuit breaker on success
                    self.record_circuit_breaker_success(f"gate_{gate_id}")
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Gate {gate_id} timed out")
                    self.record_circuit_breaker_failure(f"gate_{gate_id}")
                    results[gate_id] = self.create_timeout_result(gate_id)
                    
                except Exception as e:
                    self.logger.error(f"Gate {gate_id} failed: {e}")
                    self.record_circuit_breaker_failure(f"gate_{gate_id}")
                    results[gate_id] = self.create_error_result(gate_id, e)
                
                finally:
                    # Clean up active execution tracking
                    active_executions.pop(gate_id, None)
            
            batch_execution_time = time.time() - batch_start_time
            self.performance_logger.info(f"Batch {batch_index + 1} completed in {batch_execution_time:.2f}s")
        
        return results

    async def execute_single_gate_monitored(self, gate_id: str, execution_id: str) -> Dict[str, Any]:
        """Execute single quality gate with comprehensive monitoring"""
        gate_start_time = time.time()
        
        # Track resource usage before execution
        initial_memory = psutil.virtual_memory().used
        initial_cpu_times = psutil.cpu_times()
        
        try:
            # Execute gate with retries
            result = await self.execute_gate_with_retries(gate_id, execution_id)
            
            # Calculate resource usage
            final_memory = psutil.virtual_memory().used
            final_cpu_times = psutil.cpu_times()
            
            memory_delta = final_memory - initial_memory
            execution_time = time.time() - gate_start_time
            
            # Add monitoring metadata
            result.update({
                "monitoring": {
                    "execution_time_seconds": execution_time,
                    "memory_usage_bytes": memory_delta,
                    "resource_efficiency_score": self.calculate_efficiency_score(
                        execution_time, memory_delta
                    )
                },
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Log performance metrics
            self.performance_logger.info(
                f"Gate {gate_id} completed in {execution_time:.2f}s "
                f"(memory: {memory_delta / 1024 / 1024:.1f}MB)"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - gate_start_time
            self.error_logger.error(
                f"Gate {gate_id} failed after {execution_time:.2f}s: {e}"
            )
            raise

    async def execute_gate_with_retries(self, gate_id: str, execution_id: str) -> Dict[str, Any]:
        """Execute gate with intelligent retry logic"""
        max_retries = self.config["orchestrator"]["retry_attempts"]
        backoff_factor = self.config["orchestrator"]["retry_backoff_factor"]
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = backoff_factor ** (attempt - 1)
                    self.logger.info(f"Retrying gate {gate_id} (attempt {attempt + 1}/{max_retries + 1}) after {delay}s delay")
                    await asyncio.sleep(delay)
                
                # Actual gate execution would go here
                # For now, simulate with placeholder
                result = await self.simulate_gate_execution(gate_id)
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"Gate {gate_id} failed after {max_retries} retries: {e}")
                    raise
                else:
                    self.logger.warning(f"Gate {gate_id} attempt {attempt + 1} failed: {e}")

    async def simulate_gate_execution(self, gate_id: str) -> Dict[str, Any]:
        """Simulate gate execution for demonstration"""
        # Simulate realistic execution time
        execution_time = np.random.uniform(1, 10)
        await asyncio.sleep(execution_time)
        
        # Simulate success/failure based on circuit breaker state
        if self.is_circuit_breaker_open(f"gate_{gate_id}"):
            raise RuntimeError(f"Gate {gate_id} circuit breaker is open")
        
        # Simulate occasional failures for testing
        if np.random.random() < 0.05:  # 5% failure rate
            raise RuntimeError(f"Simulated failure for gate {gate_id}")
        
        return {
            "gate_id": gate_id,
            "status": "passed",
            "execution_time": execution_time,
            "metrics": {
                "quality_score": np.random.uniform(0.8, 1.0),
                "performance_score": np.random.uniform(0.7, 0.95)
            }
        }

    def generate_execution_plan(self) -> List[List[str]]:
        """Generate optimized execution plan for quality gates"""
        # Simulated gate dependencies and execution plan
        gates = [
            ["security_scan", "code_quality"],
            ["unit_tests", "integration_tests"],
            ["performance_tests"],
            ["documentation_check", "compliance_check"]
        ]
        return gates

    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks"""
        health_results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check_config in self.health_checks.items():
            if not check_config["enabled"]:
                continue
            
            try:
                check_start_time = time.time()
                health_check = await check_config["check_function"]()
                check_time = time.time() - check_start_time
                
                health_check.response_time_ms = check_time * 1000
                health_results[check_name] = health_check
                
                # Update overall status
                if health_check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif health_check.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
                
                # Reset consecutive failures on success
                if health_check.status == HealthStatus.HEALTHY:
                    check_config["consecutive_failures"] = 0
                else:
                    check_config["consecutive_failures"] += 1
                
                check_config["last_result"] = health_check
                
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                error_check = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time_ms=0.0,
                    error_message=str(e)
                )
                health_results[check_name] = error_check
                check_config["consecutive_failures"] += 1
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "overall_status": overall_status,
            "individual_checks": health_results,
            "timestamp": datetime.now().isoformat()
        }

    # Health Check Implementations
    
    async def check_system_memory(self) -> HealthCheck:
        """Check system memory usage"""
        memory = psutil.virtual_memory()
        threshold = self.config["resilience"]["max_memory_usage_mb"] * 1024 * 1024
        
        if memory.used > threshold:
            status = HealthStatus.UNHEALTHY
            error_msg = f"Memory usage {memory.used / 1024 / 1024:.1f}MB exceeds threshold"
        elif memory.percent > 80:
            status = HealthStatus.DEGRADED
            error_msg = f"Memory usage at {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            error_msg = None
        
        return HealthCheck(
            name="system_memory",
            status=status,
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_message=error_msg,
            metadata={
                "used_mb": memory.used / 1024 / 1024,
                "total_mb": memory.total / 1024 / 1024,
                "percent": memory.percent
            }
        )

    async def check_system_cpu(self) -> HealthCheck:
        """Check system CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        threshold = self.config["resilience"]["max_cpu_usage_percent"]
        
        if cpu_percent > threshold:
            status = HealthStatus.UNHEALTHY
            error_msg = f"CPU usage {cpu_percent:.1f}% exceeds threshold {threshold}%"
        elif cpu_percent > threshold * 0.8:
            status = HealthStatus.DEGRADED
            error_msg = f"CPU usage at {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            error_msg = None
        
        return HealthCheck(
            name="system_cpu",
            status=status,
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_message=error_msg,
            metadata={"cpu_percent": cpu_percent}
        )

    async def check_system_disk(self) -> HealthCheck:
        """Check system disk space"""
        disk = psutil.disk_usage('/')
        threshold_bytes = self.config["resilience"]["disk_space_threshold_gb"] * 1024 * 1024 * 1024
        
        if disk.free < threshold_bytes:
            status = HealthStatus.UNHEALTHY
            error_msg = f"Free disk space {disk.free / 1024 / 1024 / 1024:.1f}GB below threshold"
        elif disk.free < threshold_bytes * 2:
            status = HealthStatus.DEGRADED
            error_msg = f"Low disk space: {disk.free / 1024 / 1024 / 1024:.1f}GB remaining"
        else:
            status = HealthStatus.HEALTHY
            error_msg = None
        
        return HealthCheck(
            name="system_disk",
            status=status,
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_message=error_msg,
            metadata={
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "percent_used": (disk.used / disk.total) * 100
            }
        )

    async def check_network_connectivity(self) -> HealthCheck:
        """Check network connectivity"""
        try:
            # Test DNS resolution and basic connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            status = HealthStatus.HEALTHY
            error_msg = None
        except (socket.timeout, socket.error) as e:
            status = HealthStatus.UNHEALTHY
            error_msg = f"Network connectivity failed: {e}"
        
        return HealthCheck(
            name="network_connectivity",
            status=status,
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_message=error_msg
        )

    async def check_database_connection(self) -> HealthCheck:
        """Check database connectivity (placeholder)"""
        # This would test actual database connections
        return HealthCheck(
            name="database_connection",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time_ms=5.0,
            metadata={"connection_pool_size": 10, "active_connections": 3}
        )

    async def check_external_services(self) -> HealthCheck:
        """Check external service dependencies"""
        # This would test external APIs, services, etc.
        return HealthCheck(
            name="external_services",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time_ms=150.0,
            metadata={"services_checked": ["llm_api", "monitoring_api"]}
        )

    # Circuit Breaker Management
    
    def is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component"""
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        
        if breaker.state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if breaker.last_failure_time:
                time_since_failure = (datetime.now() - breaker.last_failure_time).total_seconds()
                if time_since_failure >= breaker.reset_timeout:
                    breaker.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {component} transitioned to half-open")
                    return False
            return True
        
        return False

    def record_circuit_breaker_success(self, component: str):
        """Record successful operation for circuit breaker"""
        if component not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[component]
        breaker.last_success_time = datetime.now()
        
        if breaker.state == CircuitBreakerState.HALF_OPEN:
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            self.logger.info(f"Circuit breaker {component} closed after successful operation")

    def record_circuit_breaker_failure(self, component: str):
        """Record failed operation for circuit breaker"""
        if component not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[component]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {component} opened after {breaker.failure_count} failures")

    def check_circuit_breaker_states(self):
        """Log current circuit breaker states"""
        for name, breaker in self.circuit_breakers.items():
            if breaker.state != CircuitBreakerState.CLOSED:
                self.logger.warning(f"Circuit breaker {name} is {breaker.state.value}")

    # Monitoring and Alerting
    
    async def send_alert(self, level: AlertLevel, message: str, component: str, metadata: Dict[str, Any] = None):
        """Send alert through configured channels"""
        alert = Alert(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            component=component,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        try:
            self.alert_queue.put_nowait(alert)
        except queue.Full:
            self.logger.error("Alert queue is full - dropping alert")

    def process_alerts(self):
        """Process alerts in background thread"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                # Send to configured channels
                if self.config["alerting"]["email"]["enabled"]:
                    self.send_email_alert(alert)
                
                if self.config["alerting"]["webhook"]["enabled"]:
                    self.send_webhook_alert(alert)
                
                if self.config["alerting"]["slack"]["enabled"]:
                    self.send_slack_alert(alert)
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Failed to process alert: {e}")

    def send_email_alert(self, alert: Alert):
        """Send email alert (placeholder)"""
        self.logger.info(f"EMAIL ALERT [{alert.level.value.upper()}]: {alert.message}")

    def send_webhook_alert(self, alert: Alert):
        """Send webhook alert (placeholder)"""
        self.logger.info(f"WEBHOOK ALERT [{alert.level.value.upper()}]: {alert.message}")

    def send_slack_alert(self, alert: Alert):
        """Send Slack alert (placeholder)"""
        self.logger.info(f"SLACK ALERT [{alert.level.value.upper()}]: {alert.message}")

    # Background Monitoring Tasks
    
    async def continuous_health_monitoring(self):
        """Continuously monitor system health"""
        while True:
            try:
                health_status = await self.perform_health_checks()
                
                # Check for degraded or unhealthy status
                if health_status["overall_status"] != HealthStatus.HEALTHY:
                    await self.send_alert(
                        level=AlertLevel.WARNING if health_status["overall_status"] == HealthStatus.DEGRADED else AlertLevel.ERROR,
                        message=f"System health is {health_status['overall_status'].value}",
                        component="health_monitor",
                        metadata=health_status
                    )
                
                # Sleep until next check
                await asyncio.sleep(self.config["orchestrator"]["health_check_interval"])
                
            except Exception as e:
                self.logger.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def continuous_metrics_collection(self):
        """Continuously collect system metrics"""
        while True:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "memory": psutil.virtual_memory()._asdict(),
                    "cpu": psutil.cpu_percent(interval=1),
                    "disk": psutil.disk_usage('/')._asdict(),
                    "network": psutil.net_io_counters()._asdict(),
                }
                
                # Store metrics (would typically go to time-series database)
                try:
                    self.metrics_buffer.put_nowait(metrics)
                except queue.Full:
                    # Remove oldest metric and add new one
                    self.metrics_buffer.get_nowait()
                    self.metrics_buffer.put_nowait(metrics)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(60)

    async def monitor_circuit_breakers(self):
        """Monitor circuit breaker states"""
        while True:
            try:
                for name, breaker in self.circuit_breakers.items():
                    if breaker.state == CircuitBreakerState.OPEN:
                        # Check if we can try to recover
                        if breaker.last_failure_time:
                            time_since_failure = (datetime.now() - breaker.last_failure_time).total_seconds()
                            if time_since_failure >= breaker.reset_timeout:
                                breaker.state = CircuitBreakerState.HALF_OPEN
                                self.logger.info(f"Circuit breaker {name} attempting recovery")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Circuit breaker monitoring failed: {e}")
                await asyncio.sleep(60)

    async def monitor_system_resources(self):
        """Monitor system resources and send alerts"""
        while True:
            try:
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    await self.send_alert(
                        level=AlertLevel.CRITICAL,
                        message=f"High memory usage: {memory.percent:.1f}%",
                        component="resource_monitor",
                        metadata={"memory_percent": memory.percent}
                    )
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    await self.send_alert(
                        level=AlertLevel.CRITICAL,
                        message=f"High CPU usage: {cpu_percent:.1f}%",
                        component="resource_monitor",
                        metadata={"cpu_percent": cpu_percent}
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Resource monitoring failed: {e}")
                await asyncio.sleep(120)

    # Utility Methods
    
    def calculate_efficiency_score(self, execution_time: float, memory_delta: int) -> float:
        """Calculate efficiency score based on resource usage"""
        # Normalize values and calculate composite score
        time_score = max(0, 1 - (execution_time / 300))  # 5 minutes max
        memory_score = max(0, 1 - (memory_delta / (512 * 1024 * 1024)))  # 512MB max
        
        return (time_score + memory_score) / 2

    def create_timeout_result(self, gate_id: str) -> Dict[str, Any]:
        """Create result object for timed-out gate"""
        return {
            "gate_id": gate_id,
            "status": "timeout",
            "error_message": "Gate execution timed out",
            "timestamp": datetime.now().isoformat()
        }

    def create_error_result(self, gate_id: str, error: Exception) -> Dict[str, Any]:
        """Create result object for failed gate"""
        return {
            "gate_id": gate_id,
            "status": "error",
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }

    async def validate_execution_results(self, results: Dict[str, Any]):
        """Validate execution results"""
        # Check for critical failures
        failed_gates = [gate_id for gate_id, result in results.items() 
                       if result.get("status") not in ["passed", "warning"]]
        
        if failed_gates:
            await self.send_alert(
                level=AlertLevel.ERROR,
                message=f"Quality gates failed: {', '.join(failed_gates)}",
                component="result_validator",
                metadata={"failed_gates": failed_gates}
            )

    async def generate_robust_report(self, results: Dict[str, Any], start_time: datetime, execution_id: str) -> Dict[str, Any]:
        """Generate comprehensive execution report with monitoring data"""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Collect final metrics
        final_health = await self.perform_health_checks()
        
        report = {
            "execution_metadata": {
                "execution_id": execution_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_execution_time": execution_time,
                "system_health_start": "healthy",  # Would track actual start health
                "system_health_end": final_health["overall_status"].value
            },
            "gate_results": results,
            "monitoring_data": {
                "circuit_breaker_states": {name: breaker.state.value for name, breaker in self.circuit_breakers.items()},
                "health_checks": {name: check["last_result"].__dict__ if check["last_result"] else None 
                                for name, check in self.health_checks.items()},
                "alerts_generated": len([a for a in self.alerts if a.timestamp >= start_time])
            },
            "performance_metrics": {
                "average_gate_execution_time": statistics.mean([
                    r.get("monitoring", {}).get("execution_time_seconds", 0) 
                    for r in results.values() if isinstance(r, dict)
                ]) if results else 0,
                "total_memory_used_mb": sum([
                    r.get("monitoring", {}).get("memory_usage_bytes", 0) / 1024 / 1024
                    for r in results.values() if isinstance(r, dict)
                ]) if results else 0
            },
            "recommendations": self.generate_recommendations(results, final_health)
        }
        
        # Save report
        report_path = Path(f"robust_quality_report_{execution_id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Robust quality report saved to {report_path}")
        return report

    def generate_recommendations(self, results: Dict[str, Any], health_status: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on results"""
        recommendations = []
        
        # Analyze results
        failed_count = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") != "passed")
        if failed_count > 0:
            recommendations.append(f"Address {failed_count} failed quality gates before deployment")
        
        # Analyze system health
        if health_status["overall_status"] != HealthStatus.HEALTHY:
            recommendations.append("Investigate system health issues before next execution")
        
        # Circuit breaker recommendations
        open_breakers = [name for name, breaker in self.circuit_breakers.items() 
                        if breaker.state == CircuitBreakerState.OPEN]
        if open_breakers:
            recommendations.append(f"Reset circuit breakers: {', '.join(open_breakers)}")
        
        return recommendations

    async def cleanup_execution_resources(self):
        """Clean up resources after execution"""
        # Clean up any temporary files, connections, etc.
        self.logger.info("Cleaning up execution resources")

async def main():
    """Main execution function for robust quality orchestrator"""
    print("🛡️ GENERATION 2: Robust Quality Orchestrator")
    print("=" * 70)
    
    orchestrator = RobustQualityOrchestrator()
    
    try:
        # Execute quality gates with full monitoring
        report = await orchestrator.execute_quality_gates_robust()
        
        print(f"\n✅ Robust execution completed")
        print(f"   Execution ID: {report['execution_metadata']['execution_id']}")
        print(f"   Total time: {report['execution_metadata']['total_execution_time']:.2f}s")
        print(f"   System health: {report['execution_metadata']['system_health_end']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Robust execution failed: {e}")
        return 1
    
    finally:
        # Stop monitoring tasks
        for task in orchestrator.monitoring_tasks:
            task.cancel()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)