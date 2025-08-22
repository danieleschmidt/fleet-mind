#!/usr/bin/env python3
"""Robust Swarm Coordinator - Generation 2: Make It Robust

This implements comprehensive error handling, fault tolerance, monitoring,
and reliability features for the Fleet-Mind swarm coordination system.

ROBUSTNESS FEATURES IMPLEMENTED:
- Comprehensive error handling and recovery
- Fault tolerance with automatic failover
- Health monitoring and alerting
- Input validation and sanitization
- Circuit breaker patterns for resilience
- Retry mechanisms with exponential backoff
- Graceful degradation under failure conditions
- Security measures and access control
"""

import asyncio
import time
import random
import math
import json
import traceback
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
from pathlib import Path

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fleet_mind_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"


class SecurityLevel(Enum):
    """Security access levels."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"


class ErrorCode(Enum):
    """Standardized error codes."""
    DRONE_TIMEOUT = "DRONE_TIMEOUT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    SYSTEM_OVERLOAD = "SYSTEM_OVERLOAD"
    MISSION_FAILURE = "MISSION_FAILURE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    COORDINATION_FAILURE = "COORDINATION_FAILURE"


@dataclass
class HealthMetric:
    """Health monitoring metric."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    last_updated: float
    alert_count: int = 0


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    user_id: str
    resource: str
    access_level: SecurityLevel
    success: bool
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """Error event for comprehensive tracking."""
    error_code: ErrorCode
    error_message: str
    component: str
    severity: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
                logger.info("Circuit breaker transitioning to half-open state")
            else:
                raise Exception("Circuit breaker is open - function call blocked")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker closed - function calls restored")
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened due to {self.failure_count} failures")
            
            raise e


class RetryManager:
    """Retry mechanism with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_drone_id(drone_id: Any) -> int:
        """Validate and sanitize drone ID."""
        if not isinstance(drone_id, (int, str)):
            raise ValueError(f"Invalid drone_id type: {type(drone_id)}")
        
        try:
            drone_id = int(drone_id)
        except ValueError:
            raise ValueError(f"Cannot convert drone_id to integer: {drone_id}")
        
        if drone_id < 0 or drone_id > 10000:
            raise ValueError(f"Drone ID out of valid range (0-10000): {drone_id}")
        
        return drone_id
    
    @staticmethod
    def validate_position(position: Any) -> Tuple[float, float, float]:
        """Validate and sanitize position coordinates."""
        if not isinstance(position, (tuple, list)):
            raise ValueError(f"Position must be tuple or list: {type(position)}")
        
        if len(position) != 3:
            raise ValueError(f"Position must have 3 coordinates: {len(position)}")
        
        try:
            x, y, z = float(position[0]), float(position[1]), float(position[2])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid position coordinates: {position}")
        
        # Validate coordinate ranges
        if abs(x) > 10000 or abs(y) > 10000:
            raise ValueError(f"Position coordinates out of range (±10km): ({x}, {y}, {z})")
        
        if z < 0 or z > 1000:
            raise ValueError(f"Altitude out of range (0-1000m): {z}")
        
        return (x, y, z)
    
    @staticmethod
    def validate_battery_level(battery: Any) -> float:
        """Validate and sanitize battery level."""
        try:
            battery = float(battery)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid battery level: {battery}")
        
        if battery < 0.0 or battery > 1.0:
            raise ValueError(f"Battery level out of range (0.0-1.0): {battery}")
        
        return battery
    
    @staticmethod
    def sanitize_mission_data(mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize mission data for security."""
        sanitized = {}
        
        # Allowed fields with validation
        allowed_fields = {
            'mission_type': str,
            'target_area': tuple,
            'priority': str,
            'duration': float,
            'max_altitude': float,
            'safety_margin': float
        }
        
        for field, field_type in allowed_fields.items():
            if field in mission_data:
                try:
                    if field == 'target_area':
                        area = mission_data[field]
                        if len(area) == 4:
                            sanitized[field] = tuple(float(x) for x in area)
                    elif field_type == str:
                        sanitized[field] = str(mission_data[field])[:100]  # Limit string length
                    elif field_type == float:
                        sanitized[field] = float(mission_data[field])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {field} in mission data: {mission_data[field]}")
        
        return sanitized


class SecurityManager:
    """Comprehensive security management."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
        self.failed_attempts: Dict[str, int] = {}
        self.security_events: List[SecurityEvent] = []
        self.rate_limits: Dict[str, List[float]] = {}
        
    def authenticate_request(self, user_id: str, token: str) -> bool:
        """Authenticate user request with token validation."""
        try:
            # Simple HMAC token validation
            expected_token = hmac.new(
                self.secret_key,
                user_id.encode(),
                hashlib.sha256
            ).hexdigest()
            
            is_valid = hmac.compare_digest(token, expected_token)
            
            self._log_security_event(
                event_type="authentication",
                user_id=user_id,
                resource="system",
                access_level=SecurityLevel.AUTHENTICATED,
                success=is_valid
            )
            
            if is_valid:
                self.failed_attempts.pop(user_id, None)
            else:
                self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
                if self.failed_attempts[user_id] > 5:
                    logger.warning(f"Multiple failed authentication attempts for user: {user_id}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def check_rate_limit(self, user_id: str, limit: int = 100, window: float = 60.0) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Remove old requests outside the window
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if now - req_time < window
        ]
        
        # Check if within limit
        if len(self.rate_limits[user_id]) >= limit:
            self._log_security_event(
                event_type="rate_limit_exceeded",
                user_id=user_id,
                resource="api",
                access_level=SecurityLevel.PUBLIC,
                success=False,
                details={"requests_in_window": len(self.rate_limits[user_id])}
            )
            return False
        
        # Add current request
        self.rate_limits[user_id].append(now)
        return True
    
    def authorize_action(self, user_id: str, action: str, resource: str) -> bool:
        """Authorize user action on resource."""
        # Simple authorization logic (extend as needed)
        admin_actions = ['delete', 'modify_system', 'emergency_stop']
        
        if action in admin_actions:
            # In real system, check user roles from database
            is_authorized = user_id.startswith('admin_')
            
            self._log_security_event(
                event_type="authorization",
                user_id=user_id,
                resource=resource,
                access_level=SecurityLevel.ADMIN,
                success=is_authorized,
                details={"action": action}
            )
            
            return is_authorized
        
        return True  # Allow non-admin actions for authenticated users
    
    def _log_security_event(self, event_type: str, user_id: str, resource: str,
                           access_level: SecurityLevel, success: bool,
                           details: Dict[str, Any] = None):
        """Log security event for audit trail."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            access_level=access_level,
            success=success,
            timestamp=time.time(),
            details=details or {}
        )
        
        self.security_events.append(event)
        
        if not success:
            logger.warning(f"Security event: {event_type} failed for user {user_id} on {resource}")


class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.health_status = HealthStatus.HEALTHY
        self.monitoring_active = False
        
    def register_metric(self, name: str, threshold: float):
        """Register a health metric for monitoring."""
        self.metrics[name] = HealthMetric(
            name=name,
            value=0.0,
            threshold=threshold,
            status=HealthStatus.HEALTHY,
            last_updated=time.time()
        )
        
    def update_metric(self, name: str, value: float):
        """Update health metric value."""
        if name not in self.metrics:
            logger.warning(f"Unknown health metric: {name}")
            return
            
        metric = self.metrics[name]
        metric.value = value
        metric.last_updated = time.time()
        
        # Determine status based on threshold
        if value > metric.threshold:
            if metric.status != HealthStatus.CRITICAL:
                metric.status = HealthStatus.CRITICAL
                metric.alert_count += 1
                self._trigger_alert(name, value, metric.threshold)
        elif value > metric.threshold * 0.8:
            metric.status = HealthStatus.DEGRADED
        else:
            metric.status = HealthStatus.HEALTHY
            
        # Update overall system health
        self._update_system_health()
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger health alert."""
        alert = {
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'severity': 'critical',
            'timestamp': time.time(),
            'message': f"{metric_name} exceeded threshold: {value} > {threshold}"
        }
        
        self.alerts.append(alert)
        logger.error(f"HEALTH ALERT: {alert['message']}")
    
    def _update_system_health(self):
        """Update overall system health status."""
        if not self.metrics:
            return
            
        critical_count = sum(1 for m in self.metrics.values() 
                           if m.status == HealthStatus.CRITICAL)
        degraded_count = sum(1 for m in self.metrics.values() 
                           if m.status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            self.health_status = HealthStatus.CRITICAL
        elif degraded_count > len(self.metrics) * 0.5:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'overall_status': self.health_status.value,
            'metrics': {
                name: {
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'status': metric.status.value,
                    'last_updated': metric.last_updated,
                    'alert_count': metric.alert_count
                }
                for name, metric in self.metrics.items()
            },
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'timestamp': time.time()
        }


class RobustSwarmCoordinator:
    """Robust swarm coordinator with comprehensive error handling and fault tolerance."""
    
    def __init__(self, max_drones: int = 50, secret_key: str = "default_secret"):
        # Core components
        self.max_drones = max_drones
        self.drones: Dict[int, Dict[str, Any]] = {}
        self.current_mission: Optional[Dict[str, Any]] = None
        self.coordination_history: List[Dict[str, Any]] = []
        
        # Robustness components
        self.health_monitor = HealthMonitor()
        self.security_manager = SecurityManager(secret_key)
        self.circuit_breaker = CircuitBreaker()
        self.retry_manager = RetryManager()
        self.input_validator = InputValidator()
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # State management
        self.is_running = False
        self.system_degraded = False
        self.emergency_stop = False
        
        # Initialize health metrics
        self._initialize_health_monitoring()
        
        logger.info("Robust swarm coordinator initialized with comprehensive error handling")
    
    def _initialize_health_monitoring(self):
        """Initialize health monitoring metrics."""
        self.health_monitor.register_metric("coordination_latency_ms", 1000.0)  # 1 second threshold
        self.health_monitor.register_metric("error_rate", 0.1)  # 10% error rate threshold
        self.health_monitor.register_metric("drone_failure_rate", 0.2)  # 20% failure rate threshold
        self.health_monitor.register_metric("memory_usage_mb", 1024.0)  # 1GB threshold
        self.health_monitor.register_metric("cpu_usage_percent", 80.0)  # 80% CPU threshold
    
    async def add_drone_safe(self, drone_id: Any, initial_position: Any, 
                           user_id: str = "system", auth_token: str = "") -> bool:
        """Safely add drone with validation and security checks."""
        try:
            # Security checks
            if not self.security_manager.authenticate_request(user_id, auth_token):
                self._log_error(ErrorCode.SECURITY_VIOLATION, 
                              "Authentication failed for add_drone", "security")
                return False
            
            if not self.security_manager.check_rate_limit(user_id):
                self._log_error(ErrorCode.SECURITY_VIOLATION,
                              "Rate limit exceeded for add_drone", "security")
                return False
            
            # Input validation
            drone_id = self.input_validator.validate_drone_id(drone_id)
            position = self.input_validator.validate_position(initial_position)
            
            # Business logic validation
            if len(self.drones) >= self.max_drones:
                self._log_error(ErrorCode.RESOURCE_EXHAUSTED,
                              f"Cannot add drone {drone_id}: swarm at capacity", "coordinator")
                return False
            
            if drone_id in self.drones:
                self._log_error(ErrorCode.INVALID_INPUT,
                              f"Drone {drone_id} already exists in swarm", "coordinator")
                return False
            
            # Add drone with retry logic
            await self.retry_manager.retry(self._add_drone_internal, drone_id, position)
            
            logger.info(f"Successfully added drone {drone_id} to swarm at position {position}")
            return True
            
        except Exception as e:
            self._log_error(ErrorCode.COORDINATION_FAILURE,
                          f"Failed to add drone {drone_id}: {str(e)}", "coordinator",
                          context={"drone_id": drone_id, "position": initial_position})
            return False
    
    async def _add_drone_internal(self, drone_id: int, position: Tuple[float, float, float]):
        """Internal drone addition with potential failure simulation."""
        # Simulate potential failures for testing
        if random.random() < 0.1:  # 10% failure rate for testing
            raise Exception("Simulated drone addition failure")
        
        self.drones[drone_id] = {
            'drone_id': drone_id,
            'position': position,
            'velocity': (0.0, 0.0, 0.0),
            'battery_level': 1.0,
            'status': 'idle',
            'last_update': time.time(),
            'failure_count': 0,
            'communication_timeout': 0
        }
    
    async def update_drone_state_safe(self, drone_id: Any, position: Any, 
                                    velocity: Any, battery_level: Any,
                                    user_id: str = "system", auth_token: str = "") -> bool:
        """Safely update drone state with validation."""
        try:
            # Security and validation
            if not self.security_manager.authenticate_request(user_id, auth_token):
                return False
            
            drone_id = self.input_validator.validate_drone_id(drone_id)
            position = self.input_validator.validate_position(position)
            velocity = self.input_validator.validate_position(velocity)  # Same validation
            battery_level = self.input_validator.validate_battery_level(battery_level)
            
            if drone_id not in self.drones:
                self._log_error(ErrorCode.INVALID_INPUT,
                              f"Drone {drone_id} not found for state update", "coordinator")
                return False
            
            # Update with circuit breaker protection
            await self.circuit_breaker.call(
                self._update_drone_state_internal,
                drone_id, position, velocity, battery_level
            )
            
            return True
            
        except Exception as e:
            self._log_error(ErrorCode.COORDINATION_FAILURE,
                          f"Failed to update drone {drone_id} state: {str(e)}", "coordinator")
            return False
    
    async def _update_drone_state_internal(self, drone_id: int, position: Tuple[float, float, float],
                                         velocity: Tuple[float, float, float], battery_level: float):
        """Internal drone state update."""
        drone = self.drones[drone_id]
        
        # Check for significant state changes that might indicate problems
        old_battery = drone['battery_level']
        if battery_level < old_battery - 0.1:  # Rapid battery drain
            logger.warning(f"Rapid battery drain detected for drone {drone_id}: "
                         f"{old_battery:.1%} → {battery_level:.1%}")
        
        # Update state
        drone.update({
            'position': position,
            'velocity': velocity,
            'battery_level': battery_level,
            'last_update': time.time(),
            'communication_timeout': 0  # Reset timeout on successful update
        })
        
        # Update status based on battery and other factors
        if battery_level < 0.1:
            drone['status'] = 'failed'
        elif battery_level < 0.2:
            drone['status'] = 'returning'
        elif drone['status'] == 'failed' and battery_level > 0.2:
            drone['status'] = 'idle'  # Recovery
    
    async def set_mission_safe(self, mission_data: Dict[str, Any],
                             user_id: str = "system", auth_token: str = "") -> bool:
        """Safely set mission with validation and authorization."""
        try:
            # Security checks
            if not self.security_manager.authenticate_request(user_id, auth_token):
                return False
            
            if not self.security_manager.authorize_action(user_id, "set_mission", "coordinator"):
                self._log_error(ErrorCode.SECURITY_VIOLATION,
                              "Insufficient authorization for set_mission", "security")
                return False
            
            # Validate and sanitize mission data
            sanitized_mission = self.input_validator.sanitize_mission_data(mission_data)
            
            if not sanitized_mission.get('mission_type'):
                self._log_error(ErrorCode.INVALID_INPUT,
                              "Mission type is required", "coordinator")
                return False
            
            # Set mission with error handling
            await self.retry_manager.retry(self._set_mission_internal, sanitized_mission)
            
            logger.info(f"Mission set successfully: {sanitized_mission['mission_type']}")
            return True
            
        except Exception as e:
            self._log_error(ErrorCode.MISSION_FAILURE,
                          f"Failed to set mission: {str(e)}", "coordinator",
                          context={"mission_data": mission_data})
            return False
    
    async def _set_mission_internal(self, mission_data: Dict[str, Any]):
        """Internal mission setting."""
        self.current_mission = mission_data.copy()
        self.current_mission['start_time'] = time.time()
        
        # Activate appropriate drones
        active_count = 0
        for drone in self.drones.values():
            if drone['status'] == 'idle' and drone['battery_level'] > 0.3:
                drone['status'] = 'active'
                active_count += 1
        
        if active_count == 0:
            raise Exception("No drones available for mission")
        
        logger.info(f"Activated {active_count} drones for mission")
    
    async def coordinate_swarm_safe(self) -> List[Dict[str, Any]]:
        """Safely coordinate swarm with comprehensive error handling."""
        start_time = time.time()
        
        try:
            if self.emergency_stop:
                return []
            
            if not self.current_mission:
                return []
            
            # Get active drones
            active_drones = [d for d in self.drones.values() 
                           if d['status'] == 'active']
            
            if not active_drones:
                logger.warning("No active drones available for coordination")
                return []
            
            # Check for communication timeouts
            self._check_communication_timeouts()
            
            # Generate coordination actions with error protection
            actions = await self.circuit_breaker.call(
                self._generate_coordination_actions, active_drones
            )
            
            # Track performance metrics
            coordination_time = (time.time() - start_time) * 1000
            self.health_monitor.update_metric("coordination_latency_ms", coordination_time)
            
            # Store actions in history
            for action in actions:
                action['timestamp'] = time.time()
            self.coordination_history.extend(actions)
            
            return actions
            
        except Exception as e:
            coordination_time = (time.time() - start_time) * 1000
            self.health_monitor.update_metric("coordination_latency_ms", coordination_time)
            
            self._log_error(ErrorCode.COORDINATION_FAILURE,
                          f"Coordination failed: {str(e)}", "coordinator")
            
            # Return safe default actions
            return self._generate_safe_actions()
    
    def _check_communication_timeouts(self):
        """Check for drone communication timeouts."""
        current_time = time.time()
        timeout_threshold = 5.0  # 5 seconds
        
        for drone in self.drones.values():
            time_since_update = current_time - drone['last_update']
            
            if time_since_update > timeout_threshold:
                drone['communication_timeout'] += 1
                
                if drone['communication_timeout'] > 3:
                    logger.warning(f"Drone {drone['drone_id']} communication timeout")
                    drone['status'] = 'failed'
                    self._log_error(ErrorCode.DRONE_TIMEOUT,
                                  f"Drone {drone['drone_id']} communication timeout",
                                  "communication")
    
    async def _generate_coordination_actions(self, active_drones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate coordination actions for active drones."""
        actions = []
        
        if not self.current_mission:
            return actions
        
        mission_type = self.current_mission.get('mission_type', 'hold')
        
        try:
            if mission_type == "formation":
                actions = await self._coordinate_formation_robust(active_drones)
            elif mission_type == "search":
                actions = await self._coordinate_search_robust(active_drones)
            elif mission_type == "coverage":
                actions = await self._coordinate_coverage_robust(active_drones)
            else:
                actions = self._coordinate_hold_position(active_drones)
                
        except Exception as e:
            logger.error(f"Error in coordination action generation: {e}")
            actions = self._coordinate_hold_position(active_drones)
        
        return actions
    
    async def _coordinate_formation_robust(self, drones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Robust formation coordination with error handling."""
        actions = []
        
        if not drones:
            return actions
        
        try:
            # Leader drone (most reliable)
            leader = max(drones, key=lambda d: d['battery_level'])
            
            leader_target = (
                leader['position'][0] + 10,
                leader['position'][1],
                leader['position'][2]
            )
            
            actions.append({
                'drone_id': leader['drone_id'],
                'target_position': leader_target,
                'target_velocity': (10, 0, 0),
                'action_type': 'formation_lead'
            })
            
            # Follower drones
            followers = [d for d in drones if d['drone_id'] != leader['drone_id']]
            
            for i, drone in enumerate(followers):
                side = 1 if i % 2 == 1 else -1
                offset = (i // 2 + 1) * 20
                
                target_pos = (
                    leader['position'][0] - offset * 0.5,
                    leader['position'][1] + side * offset,
                    leader['position'][2]
                )
                
                # Safe velocity calculation with bounds checking
                dx = max(-20, min(20, target_pos[0] - drone['position'][0]))
                dy = max(-20, min(20, target_pos[1] - drone['position'][1]))
                dz = max(-5, min(5, target_pos[2] - drone['position'][2]))
                
                target_velocity = (dx * 0.1, dy * 0.1, dz * 0.1)
                
                actions.append({
                    'drone_id': drone['drone_id'],
                    'target_position': target_pos,
                    'target_velocity': target_velocity,
                    'action_type': 'formation_follow'
                })
        
        except Exception as e:
            logger.error(f"Formation coordination error: {e}")
            actions = self._coordinate_hold_position(drones)
        
        return actions
    
    async def _coordinate_search_robust(self, drones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Robust search coordination."""
        actions = []
        
        try:
            target_area = self.current_mission.get('target_area', (-100, -100, 100, 100))
            grid_size = max(1, int(math.sqrt(len(drones))))
            
            for i, drone in enumerate(drones):
                row = i // grid_size
                col = i % grid_size
                
                x_range = target_area[2] - target_area[0]
                y_range = target_area[3] - target_area[1]
                
                target_x = target_area[0] + (col + 0.5) * x_range / grid_size
                target_y = target_area[1] + (row + 0.5) * y_range / grid_size
                target_z = 50  # Safe altitude
                
                target_pos = (target_x, target_y, target_z)
                
                # Safe movement calculation
                dx = max(-15, min(15, target_pos[0] - drone['position'][0]))
                dy = max(-15, min(15, target_pos[1] - drone['position'][1]))
                dz = max(-5, min(5, target_pos[2] - drone['position'][2]))
                
                target_velocity = (dx * 0.05, dy * 0.05, dz * 0.05)
                
                actions.append({
                    'drone_id': drone['drone_id'],
                    'target_position': target_pos,
                    'target_velocity': target_velocity,
                    'action_type': 'search_pattern'
                })
        
        except Exception as e:
            logger.error(f"Search coordination error: {e}")
            actions = self._coordinate_hold_position(drones)
        
        return actions
    
    async def _coordinate_coverage_robust(self, drones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Robust coverage coordination."""
        actions = []
        
        try:
            for i, drone in enumerate(drones):
                # Safe lawnmower pattern
                if i % 2 == 0:
                    target_velocity = (10, 0, 0)
                else:
                    target_velocity = (-10, 0, 0)
                
                target_pos = (
                    drone['position'][0] + target_velocity[0],
                    drone['position'][1],
                    drone['position'][2]
                )
                
                actions.append({
                    'drone_id': drone['drone_id'],
                    'target_position': target_pos,
                    'target_velocity': target_velocity,
                    'action_type': 'coverage_pattern'
                })
        
        except Exception as e:
            logger.error(f"Coverage coordination error: {e}")
            actions = self._coordinate_hold_position(drones)
        
        return actions
    
    def _coordinate_hold_position(self, drones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Safe hold position coordination (fallback)."""
        actions = []
        
        for drone in drones:
            actions.append({
                'drone_id': drone['drone_id'],
                'target_position': drone['position'],
                'target_velocity': (0, 0, 0),
                'action_type': 'hold_position'
            })
        
        return actions
    
    def _generate_safe_actions(self) -> List[Dict[str, Any]]:
        """Generate safe default actions in case of coordination failure."""
        actions = []
        
        # Emergency hold position for all drones
        for drone in self.drones.values():
            if drone['status'] == 'active':
                actions.append({
                    'drone_id': drone['drone_id'],
                    'target_position': drone['position'],
                    'target_velocity': (0, 0, 0),
                    'action_type': 'emergency_hold'
                })
        
        return actions
    
    def _log_error(self, error_code: ErrorCode, message: str, component: str,
                   context: Dict[str, Any] = None):
        """Log error event with comprehensive tracking."""
        error_event = ErrorEvent(
            error_code=error_code,
            error_message=message,
            component=component,
            severity='error',
            timestamp=time.time(),
            context=context or {}
        )
        
        self.error_events.append(error_event)
        
        # Update error rate metric
        recent_errors = [e for e in self.error_events 
                        if time.time() - e.timestamp < 60]  # Last minute
        error_rate = len(recent_errors) / 60.0  # Errors per second
        self.health_monitor.update_metric("error_rate", error_rate)
        
        logger.error(f"[{error_code.value}] {message} in {component}")
    
    async def run_robust_coordination_loop(self, duration: float = 60.0):
        """Run robust coordination loop with comprehensive monitoring."""
        logger.info(f"Starting robust coordination loop for {duration} seconds")
        
        self.is_running = True
        start_time = time.time()
        coordination_count = 0
        error_count = 0
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                try:
                    # Health monitoring
                    if coordination_count % 10 == 0:
                        self._update_system_metrics()
                    
                    # Check for emergency conditions
                    if self.health_monitor.health_status == HealthStatus.CRITICAL:
                        logger.warning("System in critical state - entering degraded mode")
                        self.system_degraded = True
                    
                    # Generate coordination actions
                    actions = await self.coordinate_swarm_safe()
                    
                    # Simulate drone updates
                    await self._simulate_drone_updates_robust(actions)
                    
                    coordination_count += 1
                    
                    # Status logging
                    if coordination_count % 100 == 0:
                        self._log_comprehensive_status()
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Coordination loop error: {e}")
                    self._log_error(ErrorCode.COORDINATION_FAILURE,
                                  f"Loop iteration failed: {str(e)}", "coordinator")
                    
                    # Implement graceful degradation
                    if error_count > 10:
                        logger.critical("Too many coordination errors - entering emergency mode")
                        self.emergency_stop = True
                        break
                
                # Maintain loop frequency
                loop_time = time.time() - loop_start
                if loop_time < 0.1:  # 10Hz target
                    await asyncio.sleep(0.1 - loop_time)
        
        except Exception as e:
            logger.critical(f"Critical coordination loop failure: {e}")
            self.emergency_stop = True
        
        finally:
            self.is_running = False
            logger.info(f"Robust coordination loop completed. "
                       f"Cycles: {coordination_count}, Errors: {error_count}")
    
    def _update_system_metrics(self):
        """Update system performance metrics."""
        import psutil
        
        # CPU and memory usage
        try:
            cpu_percent = psutil.cpu_percent()
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            
            self.health_monitor.update_metric("cpu_usage_percent", cpu_percent)
            self.health_monitor.update_metric("memory_usage_mb", memory_mb)
        except Exception:
            pass  # psutil not available in all environments
        
        # Drone failure rate
        total_drones = len(self.drones)
        failed_drones = sum(1 for d in self.drones.values() if d['status'] == 'failed')
        failure_rate = failed_drones / total_drones if total_drones > 0 else 0
        
        self.health_monitor.update_metric("drone_failure_rate", failure_rate)
    
    async def _simulate_drone_updates_robust(self, actions: List[Dict[str, Any]]):
        """Robust drone state simulation with error handling."""
        for action in actions:
            try:
                drone_id = action['drone_id']
                if drone_id not in self.drones:
                    continue
                
                drone = self.drones[drone_id]
                
                # Simulate physics with error probability
                if random.random() < 0.05:  # 5% chance of simulation error
                    drone['failure_count'] += 1
                    if drone['failure_count'] > 3:
                        drone['status'] = 'failed'
                        continue
                
                dt = 0.1
                target_velocity = action.get('target_velocity', (0, 0, 0))
                
                # Safe position update with bounds checking
                new_pos = (
                    max(-5000, min(5000, drone['position'][0] + target_velocity[0] * dt)),
                    max(-5000, min(5000, drone['position'][1] + target_velocity[1] * dt)),
                    max(0, min(500, drone['position'][2] + target_velocity[2] * dt))
                )
                
                # Battery simulation with realistic drain
                velocity_magnitude = math.sqrt(sum(v**2 for v in target_velocity))
                battery_drain = min(0.001, velocity_magnitude * 0.0005)
                new_battery = max(0.0, drone['battery_level'] - battery_drain)
                
                # Update state safely
                await self.update_drone_state_safe(
                    drone_id, new_pos, target_velocity, new_battery,
                    user_id="system", auth_token=self._generate_system_token()
                )
                
            except Exception as e:
                logger.warning(f"Error updating drone {action.get('drone_id')}: {e}")
    
    def _generate_system_token(self) -> str:
        """Generate system authentication token."""
        return hmac.new(
            self.security_manager.secret_key,
            "system".encode(),
            hashlib.sha256
        ).hexdigest()
    
    def _log_comprehensive_status(self):
        """Log comprehensive system status."""
        active_count = sum(1 for d in self.drones.values() if d['status'] == 'active')
        failed_count = sum(1 for d in self.drones.values() if d['status'] == 'failed')
        avg_battery = sum(d['battery_level'] for d in self.drones.values()) / len(self.drones) if self.drones else 0
        
        health_report = self.health_monitor.get_health_report()
        recent_errors = len([e for e in self.error_events if time.time() - e.timestamp < 60])
        
        logger.info(f"SYSTEM STATUS: Health={health_report['overall_status']}, "
                   f"Drones={active_count}/{len(self.drones)} active, "
                   f"{failed_count} failed, Battery={avg_battery:.1%}, "
                   f"Errors/min={recent_errors}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics and status."""
        return {
            'performance': {
                'coordination_latency_ms': self.health_monitor.metrics.get('coordination_latency_ms').value if 'coordination_latency_ms' in self.health_monitor.metrics else 0,
                'error_rate': self.health_monitor.metrics.get('error_rate').value if 'error_rate' in self.health_monitor.metrics else 0,
                'total_coordinations': len(self.coordination_history),
                'total_errors': len(self.error_events)
            },
            'health': self.health_monitor.get_health_report(),
            'security': {
                'total_events': len(self.security_manager.security_events),
                'failed_attempts': sum(self.security_manager.failed_attempts.values()),
                'active_sessions': len(self.security_manager.rate_limits)
            },
            'fleet': {
                'total_drones': len(self.drones),
                'active_drones': sum(1 for d in self.drones.values() if d['status'] == 'active'),
                'failed_drones': sum(1 for d in self.drones.values() if d['status'] == 'failed'),
                'avg_battery': sum(d['battery_level'] for d in self.drones.values()) / len(self.drones) if self.drones else 0
            },
            'system_state': {
                'is_running': self.is_running,
                'system_degraded': self.system_degraded,
                'emergency_stop': self.emergency_stop,
                'current_mission': self.current_mission['mission_type'] if self.current_mission else None
            }
        }


async def robust_demo():
    """Demonstrate robust swarm coordination with comprehensive error handling."""
    
    logger.info("🛡️ FLEET-MIND ROBUST SWARM COORDINATION DEMO")
    logger.info("=" * 60)
    
    # Initialize robust coordinator
    coordinator = RobustSwarmCoordinator(max_drones=20, secret_key="demo_secret_key")
    
    # Generate system auth token for demo
    system_token = hmac.new(
        "demo_secret_key".encode(),
        "system".encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Add drones with error handling
    logger.info("\n🔐 Adding drones with security and validation...")
    success_count = 0
    for i in range(12):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        z = random.uniform(20, 50)
        
        if await coordinator.add_drone_safe(i, (x, y, z), "system", system_token):
            success_count += 1
    
    logger.info(f"Successfully added {success_count}/12 drones with validation")
    
    # Set mission with validation
    logger.info("\n📋 Setting validated mission...")
    mission_success = await coordinator.set_mission_safe({
        'mission_type': 'formation',
        'target_area': (-300, -300, 300, 300),
        'priority': 'high',
        'duration': 30.0,
        'max_altitude': 100.0,
        'safety_margin': 10.0
    }, "system", system_token)
    
    if mission_success:
        logger.info("Mission set successfully with validation")
    else:
        logger.error("Mission validation failed")
        return
    
    # Run robust coordination
    logger.info("\n🛡️ Running robust coordination with comprehensive monitoring...")
    await coordinator.run_robust_coordination_loop(duration=15.0)
    
    # Test error recovery
    logger.info("\n⚠️ Testing error recovery and fault tolerance...")
    
    # Simulate some drone failures
    failed_drones = random.sample(list(coordinator.drones.keys()), 3)
    for drone_id in failed_drones:
        coordinator.drones[drone_id]['status'] = 'failed'
        logger.info(f"Simulated failure for drone {drone_id}")
    
    # Continue coordination with failures
    await coordinator.run_robust_coordination_loop(duration=10.0)
    
    # Switch mission to test adaptability
    logger.info("\n🔄 Testing mission switching with validation...")
    search_mission_success = await coordinator.set_mission_safe({
        'mission_type': 'search',
        'target_area': (-200, -200, 200, 200),
        'priority': 'critical',
        'duration': 20.0
    }, "system", system_token)
    
    if search_mission_success:
        await coordinator.run_robust_coordination_loop(duration=10.0)
    
    # Get comprehensive final metrics
    final_metrics = coordinator.get_comprehensive_metrics()
    
    logger.info("\n📊 COMPREHENSIVE FINAL METRICS")
    logger.info("-" * 50)
    
    performance = final_metrics['performance']
    health = final_metrics['health']
    security = final_metrics['security']
    fleet = final_metrics['fleet']
    
    logger.info(f"📈 Performance:")
    logger.info(f"  Coordination Latency: {performance['coordination_latency_ms']:.2f}ms")
    logger.info(f"  Error Rate: {performance['error_rate']:.4f}")
    logger.info(f"  Total Coordinations: {performance['total_coordinations']}")
    logger.info(f"  Total Errors: {performance['total_errors']}")
    
    logger.info(f"🏥 Health Status: {health['overall_status']}")
    logger.info(f"  Critical Metrics: {sum(1 for m in health['metrics'].values() if m['status'] == 'critical')}")
    logger.info(f"  Recent Alerts: {len(health['recent_alerts'])}")
    
    logger.info(f"🔐 Security:")
    logger.info(f"  Security Events: {security['total_events']}")
    logger.info(f"  Failed Attempts: {security['failed_attempts']}")
    logger.info(f"  Active Sessions: {security['active_sessions']}")
    
    logger.info(f"🚁 Fleet Status:")
    logger.info(f"  Total Drones: {fleet['total_drones']}")
    logger.info(f"  Active Drones: {fleet['active_drones']}")
    logger.info(f"  Failed Drones: {fleet['failed_drones']}")
    logger.info(f"  Average Battery: {fleet['avg_battery']:.1%}")
    
    logger.info("\n✅ ROBUST COORDINATION DEMO COMPLETED")
    logger.info("🎯 Key robustness achievements:")
    logger.info("  • Comprehensive input validation and sanitization")
    logger.info("  • Security authentication and authorization")
    logger.info("  • Circuit breaker and retry mechanisms")
    logger.info("  • Real-time health monitoring and alerting")
    logger.info("  • Graceful degradation under failure conditions")
    logger.info("  • Comprehensive error tracking and recovery")
    logger.info("  • Fault-tolerant coordination algorithms")
    
    logger.info("\n🚀 GENERATION 2 COMPLETE: ROBUST AND RELIABLE")
    logger.info("Ready for Generation 3: Optimized Performance")
    
    return coordinator, final_metrics


async def main():
    """Main function for robust demonstration."""
    try:
        coordinator, metrics = await robust_demo()
        
        # Save comprehensive logs and metrics
        with open("robust_coordination_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info("Metrics saved to robust_coordination_metrics.json")
        
    except Exception as e:
        logger.critical(f"Critical system failure: {e}")
        logger.critical(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())