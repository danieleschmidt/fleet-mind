"""Central SwarmCoordinator for LLM-powered drone fleet management."""

import asyncio
import time
import json
import hashlib
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import openai
except ImportError:
    openai = None

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for missing pydantic
    BaseModel = object
    def Field(*args, **kwargs):
        return None

from ..communication.latent_encoder import LatentEncoder
from ..communication.webrtc_streamer import WebRTCStreamer
from ..planning.llm_planner import LLMPlanner
from ..fleet.drone_fleet import DroneFleet
from ..security import SecurityManager, SecurityLevel
from ..monitoring import HealthMonitor, HealthStatus, AlertSeverity
from ..utils.performance import cached, async_cached, performance_monitor, get_performance_summary
from ..utils.concurrency import execute_concurrent, get_concurrency_stats
from ..utils.auto_scaling import update_scaling_metric, get_autoscaling_stats
from ..utils.circuit_breaker import circuit_breaker, CircuitBreakerConfig, get_circuit_breaker
from ..utils.retry import retry, RetryConfig, LLM_RETRY_CONFIG, NETWORK_RETRY_CONFIG
from ..utils.input_sanitizer import validate_mission_input, validate_drone_command
from ..optimization.ai_performance_optimizer import get_ai_optimizer, record_performance_metrics
from ..optimization.service_mesh_coordinator import get_service_mesh, ServiceType, route_service_request
from ..optimization.ml_cost_optimizer import get_cost_optimizer, record_cost_metrics
from ..optimization.multi_tier_cache import get_multi_tier_cache, cached_get, cached_put
from ..communication.high_performance_comm import get_hp_communicator, send_high_performance_message, Priority


class MissionStatus(Enum):
    """Mission execution status."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MissionConstraints:
    """Mission execution constraints."""
    max_altitude: float = 120.0  # meters
    battery_time: float = 30.0   # minutes
    safety_distance: float = 5.0  # meters between drones
    geofence: Optional[List[tuple]] = None
    no_fly_zones: Optional[List[tuple]] = None


@dataclass
class SwarmState:
    """Current state of the drone swarm."""
    num_active_drones: int = 0
    num_total_drones: int = 0
    num_failed_drones: int = 0
    average_battery: float = 100.0
    mission_progress: float = 0.0
    safety_status: str = "safe"
    last_update: float = field(default_factory=time.time)


class SwarmCoordinator:
    """Central coordinator for LLM-powered drone swarm management.
    
    Orchestrates high-level mission planning using GPT-4o and distributes
    compressed latent action codes to individual drones via WebRTC.
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o",
        latent_dim: int = 512,
        compression_ratio: float = 100.0,
        max_drones: int = 100,
        update_rate: float = 10.0,
        safety_constraints: Optional[MissionConstraints] = None,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        enable_health_monitoring: bool = True,
    ):
        """Initialize SwarmCoordinator.
        
        Args:
            llm_model: LLM model identifier (default: gpt-4o)
            latent_dim: Latent encoding dimension
            compression_ratio: Target compression ratio
            max_drones: Maximum fleet size
            update_rate: Planning frequency in Hz
            safety_constraints: Mission safety parameters
            security_level: Security level for operations
            enable_health_monitoring: Enable health monitoring
        """
        self.llm_model = llm_model
        self.latent_dim = latent_dim
        self.compression_ratio = compression_ratio
        self.max_drones = max_drones
        self.update_rate = update_rate
        self.safety_constraints = safety_constraints or MissionConstraints()
        self.security_level = security_level
        self.enable_health_monitoring = enable_health_monitoring
        
        # Core components
        self.llm_planner = LLMPlanner(model=llm_model)
        self.latent_encoder = LatentEncoder(
            input_dim=4096,
            latent_dim=latent_dim,
            compression_type="learned_vqvae"
        )
        self.webrtc_streamer = WebRTCStreamer()
        
        # Security and monitoring
        self.security_manager = SecurityManager(
            security_level=security_level,
            key_rotation_interval=3600.0,  # 1 hour
            enable_threat_detection=True
        )
        
        self.health_monitor = HealthMonitor(
            check_interval=30.0,
            alert_cooldown=300.0,
            enable_system_monitoring=True,
            enable_network_monitoring=True
        ) if enable_health_monitoring else None
        
        # State management
        self.fleet: Optional[DroneFleet] = None
        self.mission_status = MissionStatus.IDLE
        self.current_mission: Optional[str] = None
        self.swarm_state = SwarmState()
        self.context_history: List[Dict[str, Any]] = []
        self.start_time = time.time()  # Track initialization time
        
        # Initialize logging
        from ..utils.logging import get_logger
        self.logger = get_logger("swarm_coordinator", component="coordination")
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'mission_start': [],
            'mission_complete': [],
            'drone_failure': [],
            'emergency_stop': [],
            'security_alert': [],
            'health_alert': [],
        }
        
        # Security event tracking
        self.security_events: List[Dict[str, Any]] = []
        self.threat_level = "low"
        
        # Async task management
        self._planning_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._security_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect_fleet(self, fleet: DroneFleet) -> None:
        """Connect and initialize drone fleet.
        
        Args:
            fleet: DroneFleet instance to coordinate
        """
        self.fleet = fleet
        await self.webrtc_streamer.initialize(fleet.drone_ids)
        
        # Generate security credentials for all drones
        for drone_id in fleet.drone_ids:
            credentials = self.security_manager.generate_drone_credentials(
                drone_id, 
                permissions={"basic_flight", "telemetry", "emergency_response", "formation_flight"}
            )
            print(f"Generated security credentials for {drone_id}")
        
        # Register components for health monitoring
        if self.health_monitor:
            self.health_monitor.register_component("coordinator")
            self.health_monitor.register_component("webrtc_streamer") 
            self.health_monitor.register_component("llm_planner")
            self.health_monitor.register_component("latent_encoder")
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Add health alert callback
            self.health_monitor.add_alert_callback(self._on_health_alert)
        
        # Update swarm state
        self.swarm_state.num_total_drones = len(fleet.drone_ids)
        self.swarm_state.num_active_drones = len(fleet.get_active_drones())
        
        print(f"Connected to fleet of {len(fleet.drone_ids)} drones with {self.security_level.value} security level")

    @async_cached(ttl=300, max_size=100)  # Cache plans for 5 minutes
    @performance_monitor
    @retry(max_attempts=3, initial_delay=1.0, exceptions=(ConnectionError, TimeoutError))
    @circuit_breaker("mission_planning", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0))
    async def generate_plan(
        self,
        mission: str,
        constraints: Optional[MissionConstraints] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate mission plan using LLM with robust error handling.
        
        Args:
            mission: Mission description in natural language
            constraints: Mission execution constraints
            context: Additional context (world state, drone capabilities)
            
        Returns:
            Generated mission plan with latent encoding
            
        Raises:
            RuntimeError: If no fleet is connected
            ValueError: If input validation fails
            ConnectionError: If LLM service is unavailable
        """
        # Input validation and sanitization
        try:
            if not mission or not isinstance(mission, str):
                raise ValueError("Mission description must be a non-empty string")
            
            # Sanitize mission input
            sanitized_input = validate_mission_input({
                'mission': mission,
                'context': context or {}
            })
            mission = sanitized_input['mission']
            context = sanitized_input['context']
            
        except Exception as e:
            if self.health_monitor:
                self.health_monitor.record_error("coordination", f"Input validation failed: {e}")
            self.logger.error(f"Mission input validation failed: {e}")
            raise ValueError(f"Mission input validation failed: {e}")
        
        # Fleet connectivity check
        if not self.fleet:
            if self.health_monitor:
                self.health_monitor.record_error("coordination", "No fleet connected for mission planning")
            self.logger.error("No fleet connected - cannot generate mission plan")
            raise RuntimeError("No fleet connected - cannot generate mission plan")
            
        # Use provided constraints or defaults
        active_constraints = constraints or self.safety_constraints
        
        try:
            # Prepare context for LLM
            planning_context = {
                'mission': mission,
                'num_drones': self.swarm_state.num_active_drones,
                'constraints': active_constraints.__dict__,
                'drone_capabilities': self.fleet.get_capabilities(),
                'current_state': self.swarm_state.__dict__,
                'history': self.context_history[-5:],  # Last 5 contexts
            }
            
            if context:
                planning_context.update(context)
            
            # Generate plan with LLM (with circuit breaker protection)
            start_time = time.time()
            plan = await self.llm_planner.generate_plan(planning_context)
            planning_latency = (time.time() - start_time) * 1000  # ms
            
            # Validate plan output
            if not plan or not isinstance(plan, dict):
                raise ValueError("LLM returned invalid plan format")
                
        except Exception as e:
            if self.health_monitor:
                self.health_monitor.record_error("llm_planning", f"Plan generation failed: {e}")
            self.logger.error(f"Plan generation failed: {e}")
            # Try to provide a fallback basic plan
            try:
                fallback_plan = await self._generate_fallback_plan(mission, active_constraints)
                self.logger.warning(f"Using fallback plan due to LLM failure: {e}")
                return fallback_plan
            except Exception as fallback_error:
                self.logger.error(f"Fallback plan generation also failed: {fallback_error}")
                raise ConnectionError(f"Mission planning failed: {e}, fallback also failed: {fallback_error}")
        
        try:
            # Encode to latent representation - handle both old and new plan formats
            actions_to_encode = []
            if 'actions' in plan:
                actions_to_encode = plan['actions']
            elif 'action_sequences' in plan and len(plan['action_sequences']) > 0:
                # Extract actions from action sequences
                for seq in plan['action_sequences']:
                    if 'actions' in seq:
                        actions_to_encode.extend(seq['actions'])
            
            # Encode actions to latent space
            latent_plan = self.latent_encoder.encode(actions_to_encode)
            
            # Package complete plan
            complete_plan = {
                'mission_id': f"mission_{int(time.time())}",
                'description': mission,
                'raw_plan': plan,
                'latent_code': latent_plan,
                'constraints': active_constraints.__dict__,
                'planning_latency_ms': planning_latency,
                'timestamp': time.time(),
            }
            
        except Exception as e:
            if self.health_monitor:
                self.health_monitor.record_error("encoding", f"Latent encoding failed: {e}")
            self.logger.error(f"Failed to encode plan to latent space: {e}")
            raise ValueError(f"Plan encoding failed: {e}")
        
        # Store in context history
        self.context_history.append({
            'timestamp': time.time(),
            'mission': mission,
            'plan_summary': plan.get('summary', ''),
            'latency_ms': planning_latency,
        })
        
        return complete_plan

    @performance_monitor
    async def execute_mission(
        self,
        latent_plan: Dict[str, Any],
        monitor_frequency: float = 10.0,
        replan_threshold: float = 0.7,
    ) -> bool:
        """Execute mission with real-time monitoring.
        
        Args:
            latent_plan: Generated mission plan with latent encoding
            monitor_frequency: Monitoring frequency in Hz
            replan_threshold: Confidence threshold for replanning
            
        Returns:
            True if mission completed successfully
        """
        if not self.fleet:
            raise RuntimeError("No fleet connected")
            
        self.mission_status = MissionStatus.EXECUTING
        self.current_mission = latent_plan['description']
        
        try:
            # Broadcast latent plan to all drones
            await self.webrtc_streamer.broadcast(
                latent_plan['latent_code'],
                priority='real_time',
                reliability='best_effort'
            )
            
            # Trigger mission start callbacks
            await self._trigger_callbacks('mission_start', latent_plan)
            
            # Start monitoring and execution
            self._running = True
            self._planning_task = asyncio.create_task(
                self._planning_loop(monitor_frequency, replan_threshold)
            )
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop(monitor_frequency)
            )
            
            # Wait for mission completion
            await asyncio.gather(self._planning_task, self._monitoring_task)
            
            self.mission_status = MissionStatus.COMPLETED
            await self._trigger_callbacks('mission_complete', latent_plan)
            
            return True
            
        except Exception as e:
            self.mission_status = MissionStatus.FAILED
            print(f"Mission execution failed: {e}")
            return False
        finally:
            self._running = False

    async def emergency_stop(self) -> None:
        """Emergency stop all drone operations."""
        if not self.fleet:
            return
            
        self.mission_status = MissionStatus.PAUSED
        self._running = False
        
        # Send emergency stop to all drones
        emergency_code = self.latent_encoder.encode_emergency_stop()
        await self.webrtc_streamer.broadcast(
            emergency_code,
            priority='critical',
            reliability='guaranteed'
        )
        
        await self._trigger_callbacks('emergency_stop', {})
        print("Emergency stop activated")

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback.
        
        Args:
            event: Event name (mission_start, mission_complete, etc.)
            callback: Callback function to execute
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status and metrics.
        
        Returns:
            Comprehensive swarm status information
        """
        if not self.fleet:
            return {"error": "No fleet connected"}
            
        return {
            'mission_status': self.mission_status.value,
            'current_mission': self.current_mission,
            'swarm_state': self.swarm_state.__dict__,
            'fleet_health': await self.fleet.get_health_status(),
            'communication_status': self.webrtc_streamer.get_status(),
            'recent_latency_ms': self._get_recent_latency(),
            'uptime_seconds': time.time() - self.swarm_state.last_update,
        }

    async def _planning_loop(
        self,
        frequency: float,
        replan_threshold: float
    ) -> None:
        """Continuous planning loop for real-time adaptation."""
        interval = 1.0 / frequency
        
        while self._running:
            try:
                # Check if replanning is needed
                confidence = await self._assess_plan_confidence()
                
                if confidence < replan_threshold and self.current_mission:
                    # Generate new plan
                    new_plan = await self.generate_plan(
                        self.current_mission,
                        context={'replan_reason': 'low_confidence'}
                    )
                    
                    # Broadcast updated plan
                    await self.webrtc_streamer.broadcast(
                        new_plan['latent_code'],
                        priority='real_time'
                    )
                    
                    print(f"Replanned mission (confidence: {confidence:.2f})")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Planning loop error: {e}")
                await asyncio.sleep(interval)

    async def _monitoring_loop(self, frequency: float) -> None:
        """Continuous monitoring loop for swarm state."""
        interval = 1.0 / frequency
        
        while self._running:
            try:
                if self.fleet:
                    # Update swarm state
                    active_drones = self.fleet.get_active_drones()
                    self.swarm_state.num_active_drones = len(active_drones)
                    self.swarm_state.average_battery = self.fleet.get_average_battery()
                    self.swarm_state.last_update = time.time()
                    
                    # Check for drone failures
                    failed_drones = self.fleet.get_failed_drones()
                    if failed_drones:
                        for drone_id in failed_drones:
                            await self._trigger_callbacks('drone_failure', drone_id)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)

    async def _assess_plan_confidence(self) -> float:
        """Assess confidence in current plan execution."""
        if not self.fleet:
            return 0.0
            
        # Simple confidence based on drone health and progress
        health_score = self.fleet.get_average_health()
        progress_score = self.swarm_state.mission_progress
        
        # Weight factors
        confidence = (health_score * 0.6) + (progress_score * 0.4)
        return min(max(confidence, 0.0), 1.0)

    async def _generate_fallback_plan(self, 
                                    mission: str, 
                                    constraints: MissionConstraints) -> Dict[str, Any]:
        """Generate a basic fallback plan when LLM is unavailable."""
        self.logger.info("Generating fallback mission plan")
        
        # Create basic formation-based plan
        num_drones = self.swarm_state.num_active_drones
        formation_pattern = "grid" if num_drones > 4 else "line"
        
        fallback_plan = {
            'summary': f'Fallback {formation_pattern} formation mission',
            'objectives': [
                'Maintain safe formation',
                'Follow basic navigation pattern', 
                'Monitor for obstacles',
                'Report status regularly'
            ],
            'action_sequences': [
                {
                    'phase': 'formation',
                    'duration': 30,
                    'actions': ['form_formation', 'check_spacing', 'stabilize']
                },
                {
                    'phase': 'execution',
                    'duration': 300,
                    'actions': ['navigate_pattern', 'maintain_formation', 'monitor']
                },
                {
                    'phase': 'return',
                    'duration': 60,
                    'actions': ['return_home', 'land_sequence']
                }
            ],
            'estimated_duration_minutes': 7,
            'fallback': True
        }
        
        # Encode to latent representation
        latent_plan = self.latent_encoder.encode(['form_formation', 'navigate_pattern', 'return_home'])
        
        return {
            'mission_id': f"fallback_mission_{int(time.time())}",
            'description': f"Fallback plan: {mission[:100]}...",
            'raw_plan': fallback_plan,
            'latent_code': latent_plan,
            'constraints': constraints.__dict__,
            'planning_latency_ms': 0.5,  # Very fast fallback
            'timestamp': time.time(),
            'fallback': True
        }

    async def _trigger_callbacks(self, event: str, data: Any) -> None:
        """Trigger registered callbacks for an event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    if self.health_monitor:
                        self.health_monitor.record_error("callback", f"Callback error for {event}: {e}")
                    self.logger.error(f"Callback error for {event}: {e}")

    def _get_recent_latency(self) -> float:
        """Get recent average latency from context history."""
        if not self.context_history:
            return 0.0
            
        recent = self.context_history[-3:]  # Last 3 entries
        latencies = [entry.get('latency_ms', 0) for entry in recent]
        return sum(latencies) / len(latencies)
    
    @performance_monitor
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance and operational statistics.
        
        Returns:
            Complete system statistics including performance, scaling, and operational metrics
        """
        # Update scaling metrics
        if self.fleet:
            fleet_status = self.fleet.get_fleet_status()
            update_scaling_metric('cpu', fleet_status.get('average_health', 0.5) * 100)
            update_scaling_metric('queue', len(self.context_history))
            
            # Calculate response time from recent missions
            avg_response_time = self._get_recent_latency()
            update_scaling_metric('response_time', avg_response_time)
        
        # Compile comprehensive statistics
        return {
            'swarm_status': {
                'mission_status': self.mission_status.value,
                'current_mission': self.current_mission,
                'uptime_seconds': time.time() - self.start_time,
                'recent_latency_ms': self._get_recent_latency(),
                'active_drones': self.swarm_state.num_active_drones,
                'failed_drones': self.swarm_state.num_failed_drones,
                'mission_progress': self.swarm_state.mission_progress,
                'safety_status': self.swarm_state.safety_status,
            },
            'fleet_stats': self.fleet.get_fleet_status() if self.fleet else {},
            'performance_stats': get_performance_summary(),
            'concurrency_stats': get_concurrency_stats(),
            'autoscaling_stats': get_autoscaling_stats(),
            'cache_stats': {
                'generate_plan_cache': getattr(self.generate_plan, 'cache_stats', lambda: {})(),
            },
            'system_health': {
                'memory_usage_mb': self._get_memory_usage(),
                'active_tasks': self._get_active_tasks_count(),
                'error_rate': self._calculate_error_rate(),
            },
            'timestamp': time.time(),
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _get_active_tasks_count(self) -> int:
        """Get count of active async tasks."""
        try:
            return len([t for t in asyncio.all_tasks() if not t.done()])
        except RuntimeError:
            # No event loop running
            return 0
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent error rate based on context history."""
        if not self.context_history:
            return 0.0
        
        recent = self.context_history[-20:]  # Last 20 operations
        errors = sum(1 for entry in recent if entry.get('error', False))
        return errors / len(recent) if recent else 0.0
        
    async def optimize_fleet_performance(self) -> Dict[str, Any]:
        """Optimize fleet performance based on current metrics.
        
        Returns:
            Dictionary of optimization results and recommendations
        """
        if not self.fleet:
            return {"error": "No fleet connected"}
        
        # Analyze current performance
        performance_stats = self.get_comprehensive_stats()
        fleet_health = await self.fleet.get_health_status()
        
        optimizations = {
            'recommendations': [],
            'actions_taken': [],
            'performance_gains': {},
        }
        
        # Check for performance issues
        avg_latency = self._get_recent_latency()
        if avg_latency > 80:  # ms
            optimizations['recommendations'].append("Consider reducing LLM context size")
            if avg_latency > 120:
                # Reduce context automatically
                self.context_history = self.context_history[-3:]
                optimizations['actions_taken'].append("Reduced context history for latency")
        
        # Check drone health distribution
        critical_drones = len(fleet_health.get('critical', []))
        if critical_drones > len(self.fleet.drone_ids) * 0.1:  # >10% critical
            optimizations['recommendations'].append("Schedule fleet maintenance")
            
        # Auto-scaling recommendations
        active_drones = self.swarm_state.num_active_drones
        if active_drones < 5 and len(self.fleet.drone_ids) > 10:
            optimizations['recommendations'].append("Many drones offline - check connectivity")
        
        # Update autoscaling metrics
        update_scaling_metric('fleet_health', self.fleet.get_average_health())
        update_scaling_metric('active_ratio', active_drones / len(self.fleet.drone_ids))
        
        return optimizations
    
    async def adaptive_replan(self, context: Dict[str, Any]) -> bool:
        """Perform adaptive replanning based on changing conditions.
        
        Args:
            context: Context information about changed conditions
            
        Returns:
            True if replanning was successful
        """
        if not self.current_mission:
            return False
        
        try:
            # Generate adaptive plan
            adaptive_context = {
                'mission': self.current_mission,
                'adaptation_reason': context.get('reason', 'conditions_changed'),
                'changed_conditions': context,
                'previous_progress': self.swarm_state.mission_progress,
            }
            
            new_plan = await self.generate_plan(
                self.current_mission,
                context=adaptive_context
            )
            
            # Broadcast new plan
            if new_plan:
                await self.webrtc_streamer.broadcast(
                    new_plan['latent_code'],
                    priority='real_time',
                    reliability='reliable'
                )
                
                print(f"Adaptive replanning completed: {context.get('reason')}")
                return True
                
        except Exception as e:
            print(f"Adaptive replanning failed: {e}")
            return False
        
        return False
    
    async def secure_broadcast(self, data: Any, priority: str = "real_time") -> Dict[str, bool]:
        """Securely broadcast data to fleet with encryption and authentication.
        
        Args:
            data: Data to broadcast
            priority: Message priority level
            
        Returns:
            Dictionary mapping drone_id to success status
        """
        if not self.fleet:
            raise RuntimeError("No fleet connected")
        
        results = {}
        
        for drone_id in self.fleet.get_active_drones():
            try:
                # Encrypt message for specific drone
                encrypted_data = self.security_manager.encrypt_message(
                    data, 
                    recipient=drone_id,
                    security_level=self.security_level
                )
                
                # Broadcast encrypted message
                success = await self.webrtc_streamer.send_to_drone(
                    drone_id,
                    encrypted_data,
                    priority=priority,
                    reliability="reliable"
                )
                
                results[drone_id] = success
                
                # Update health metrics
                if self.health_monitor and success:
                    self.health_monitor.update_metric("coordinator", "successful_broadcasts", 1.0)
                
            except Exception as e:
                print(f"Secure broadcast failed for {drone_id}: {e}")
                results[drone_id] = False
                
                # Log security event
                self.security_events.append({
                    "timestamp": time.time(),
                    "event": "broadcast_failure",
                    "drone_id": drone_id,
                    "error": str(e)
                })
        
        return results
    
    def _on_health_alert(self, alert) -> None:
        """Handle health monitoring alerts."""
        print(f"Health Alert [{alert.severity.value.upper()}]: {alert.message}")
        
        # Store alert in context
        self.context_history.append({
            "timestamp": time.time(),
            "type": "health_alert",
            "severity": alert.severity.value,
            "component": alert.component,
            "metric": alert.metric_name,
            "message": alert.message
        })
        
        # Take automatic action for critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            asyncio.create_task(self._handle_critical_health_alert(alert))
        
        # Trigger callbacks
        asyncio.create_task(self._trigger_callbacks('health_alert', alert))
    
    async def _handle_critical_health_alert(self, alert) -> None:
        """Handle critical health alerts automatically."""
        if alert.component == "coordinator" and alert.metric_name == "memory_usage":
            # Reduce context history to free memory
            self.context_history = self.context_history[-10:]
            print("Emergency: Reduced context history due to high memory usage")
        
        elif alert.component == "webrtc_streamer" and "latency" in alert.metric_name:
            # Reduce update rate to improve performance
            original_rate = self.update_rate
            self.update_rate = max(1.0, self.update_rate * 0.5)
            print(f"Emergency: Reduced update rate from {original_rate} to {self.update_rate} Hz")
        
        elif alert.metric_name == "error_rate":
            # Switch to safe mode
            self.threat_level = "high"
            print("Emergency: Switched to high threat level due to high error rate")
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status.
        
        Returns:
            Security status and threat information
        """
        security_status = self.security_manager.get_security_status()
        
        return {
            "security_level": self.security_level.value,
            "threat_level": self.threat_level,
            "recent_security_events": len(self.security_events),
            "security_manager_status": security_status,
            "key_rotation_due": security_status["key_rotation"]["time_until_next"] < 300,  # Within 5 minutes
            "blocked_sources": security_status.get("blocked_sources", 0),
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status.
        
        Returns:
            Health status and monitoring information
        """
        if not self.health_monitor:
            return {"health_monitoring": "disabled"}
        
        system_health = self.health_monitor.get_system_health()
        active_alerts = self.health_monitor.get_alerts(active_only=True)
        
        return {
            "overall_status": system_health["overall_status"],
            "monitored_components": system_health["total_components"],
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "uptime_seconds": system_health["uptime_seconds"],
            "last_check": system_health["last_check"],
            "alert_details": [
                {
                    "severity": alert.severity.value,
                    "component": alert.component,
                    "metric": alert.metric_name,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in active_alerts[:5]  # Show first 5 alerts
            ]
        }
    
    async def update_health_metrics(self) -> None:
        """Update health metrics for coordinator components."""
        if not self.health_monitor:
            return
        
        try:
            # Coordinator metrics
            self.health_monitor.update_metric(
                "coordinator", 
                "missions_completed", 
                self.swarm_state.mission_progress,
                "%",
                "Mission completion progress"
            )
            
            self.health_monitor.update_metric(
                "coordinator",
                "active_drones_ratio",
                self.swarm_state.num_active_drones / max(1, self.swarm_state.num_total_drones) * 100,
                "%", 
                "Percentage of active drones"
            )
            
            # LLM planner metrics
            planner_stats = self.llm_planner.get_performance_stats()
            self.health_monitor.update_metric(
                "llm_planner",
                "average_planning_time",
                planner_stats.get("average_planning_time_ms", 0),
                "ms",
                "Average LLM planning time"
            )
            
            self.health_monitor.update_metric(
                "llm_planner",
                "success_rate",
                planner_stats.get("success_rate", 1.0) * 100,
                "%",
                "LLM planning success rate"
            )
            
            # WebRTC metrics
            webrtc_status = self.webrtc_streamer.get_status()
            self.health_monitor.update_metric(
                "webrtc_streamer",
                "active_connections",
                webrtc_status.get("active_connections", 0),
                "",
                "Number of active WebRTC connections"
            )
            
            self.health_monitor.update_metric(
                "webrtc_streamer", 
                "average_latency",
                webrtc_status.get("average_latency_ms", 0),
                "ms",
                "Average WebRTC communication latency"
            )
            
            # Latent encoder metrics
            encoder_stats = self.latent_encoder.get_compression_stats()
            self.health_monitor.update_metric(
                "latent_encoder",
                "compression_ratio",
                encoder_stats.get("current_metrics", {}).get("compression_ratio", 0),
                "",
                "Current compression ratio"
            )
            
            self.health_monitor.update_metric(
                "latent_encoder",
                "encoding_time", 
                encoder_stats.get("average_encoding_time_ms", 0),
                "ms",
                "Average encoding time"
            )
            
        except Exception as e:
            print(f"Health metrics update error: {e}")
    
    async def perform_security_audit(self) -> Dict[str, Any]:
        """Perform comprehensive security audit.
        
        Returns:
            Security audit results and recommendations
        """
        audit_results = {
            "timestamp": time.time(),
            "security_level": self.security_level.value,
            "findings": [],
            "recommendations": [],
            "risk_score": 0.0,
        }
        
        try:
            # Check key rotation status
            security_status = await self.get_security_status()
            if security_status.get("key_rotation_due"):
                audit_results["findings"].append("Encryption key rotation due")
                audit_results["recommendations"].append("Rotate encryption keys immediately")
                audit_results["risk_score"] += 0.2
            
            # Check threat detection
            if len(self.security_events) > 0:
                recent_events = [e for e in self.security_events if time.time() - e["timestamp"] < 3600]
                if len(recent_events) > 10:
                    audit_results["findings"].append(f"High security event frequency: {len(recent_events)} events in last hour")
                    audit_results["recommendations"].append("Investigate security event patterns")
                    audit_results["risk_score"] += 0.3
            
            # Check drone authentication status
            if self.fleet:
                unauthenticated_drones = []
                for drone_id in self.fleet.drone_ids:
                    if drone_id not in self.security_manager.drone_credentials:
                        unauthenticated_drones.append(drone_id)
                
                if unauthenticated_drones:
                    audit_results["findings"].append(f"Unauthenticated drones: {unauthenticated_drones}")
                    audit_results["recommendations"].append("Generate credentials for all drones")
                    audit_results["risk_score"] += 0.4
            
            # Check communication security
            if self.security_level == SecurityLevel.LOW:
                audit_results["findings"].append("Low security level in use")
                audit_results["recommendations"].append("Consider upgrading to HIGH security level")
                audit_results["risk_score"] += 0.1
            
            # Determine overall risk level
            if audit_results["risk_score"] >= 0.7:
                audit_results["risk_level"] = "CRITICAL"
            elif audit_results["risk_score"] >= 0.4:
                audit_results["risk_level"] = "HIGH"
            elif audit_results["risk_score"] >= 0.2:
                audit_results["risk_level"] = "MEDIUM"
            else:
                audit_results["risk_level"] = "LOW"
            
            return audit_results
            
        except Exception as e:
            audit_results["error"] = str(e)
            audit_results["risk_level"] = "UNKNOWN"
            return audit_results
    
    # ============== GENERATION 3 ENHANCEMENTS ==============
    
    async def initialize_generation3_systems(self):
        """Initialize Generation 3 performance optimization systems."""
        try:
            # Initialize AI performance optimizer
            self.ai_optimizer = get_ai_optimizer()
            self.logger.info("Initialized AI performance optimizer")
            
            # Initialize service mesh
            self.service_mesh = await get_service_mesh()
            self.logger.info("Initialized service mesh coordinator")
            
            # Initialize cost optimizer
            self.cost_optimizer = get_cost_optimizer()
            self.logger.info("Initialized ML cost optimizer")
            
            # Initialize multi-tier cache
            self.multi_cache = await get_multi_tier_cache()
            self.logger.info("Initialized multi-tier cache system")
            
            # Initialize high-performance communicator
            self.hp_communicator = await get_hp_communicator()
            self.logger.info("Initialized high-performance communication layer")
            
            # Start Generation 3 background tasks
            self._gen3_tasks = [
                asyncio.create_task(self._gen3_performance_optimization_loop()),
                asyncio.create_task(self._gen3_cost_monitoring_loop()),
                asyncio.create_task(self._gen3_service_mesh_management()),
            ]
            
            self.logger.info("Generation 3 systems fully initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Generation 3 systems: {e}")
    
    async def _gen3_performance_optimization_loop(self):
        """Generation 3 AI-powered performance optimization loop."""
        while self._running:
            try:
                # Collect current performance metrics
                current_metrics = await self._collect_generation3_metrics()
                
                # Record metrics for AI optimization
                record_performance_metrics(
                    latency_ms=current_metrics.get("avg_latency_ms", 0),
                    throughput_rps=current_metrics.get("throughput_rps", 0),
                    cpu_usage_percent=current_metrics.get("cpu_usage", 0),
                    memory_usage_mb=current_metrics.get("memory_usage_mb", 0),
                    error_rate=current_metrics.get("error_rate", 0),
                    cache_hit_rate=current_metrics.get("cache_hit_rate", 0),
                )
                
                # Get AI optimization suggestions
                ai_optimizer = get_ai_optimizer()
                optimization_action = await ai_optimizer.suggest_optimization()
                
                if optimization_action and optimization_action.confidence > 0.7:
                    await ai_optimizer._apply_optimization(optimization_action)
                    self.logger.info(f"Applied AI optimization: {optimization_action.action_type}")
                
                await asyncio.sleep(120)  # Run every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Generation 3 performance optimization error: {e}")
                await asyncio.sleep(120)
    
    async def _gen3_cost_monitoring_loop(self):
        """Generation 3 cost optimization monitoring loop."""
        while self._running:
            try:
                # Collect cost metrics
                current_load = {
                    "cpu_usage": await self._get_cpu_usage(),
                    "memory_usage": await self._get_memory_usage(),
                    "response_time": self._get_recent_latency(),
                    "throughput": self.processed_tasks,
                }
                
                # Record cost metrics
                record_cost_metrics(
                    current_cost=self._calculate_current_operational_cost(),
                    instance_counts={"on_demand": 1},  # Simplified
                    performance_metrics=current_load,
                )
                
                # Get cost optimization recommendation
                cost_optimizer = get_cost_optimizer()
                recommendation = await cost_optimizer.predict_optimal_scaling(current_load, current_load)
                
                if recommendation.predicted_cost_savings > 0.1:  # >$0.10/hour savings
                    await cost_optimizer.execute_cost_optimization(recommendation)
                    self.logger.info(f"Applied cost optimization: {recommendation.action}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Generation 3 cost monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _gen3_service_mesh_management(self):
        """Generation 3 service mesh management loop."""
        while self._running:
            try:
                # Register coordinator as a service
                service_mesh = await get_service_mesh()
                
                # Auto-scale services based on load
                mesh_stats = service_mesh.get_mesh_status()
                
                if mesh_stats["active_requests"] > mesh_stats["max_concurrent_requests"] * 0.8:
                    # Scale up planning services
                    await service_mesh.scale_service_type(ServiceType.PLANNER, 3)
                elif mesh_stats["active_requests"] < mesh_stats["max_concurrent_requests"] * 0.3:
                    # Scale down planning services
                    await service_mesh.scale_service_type(ServiceType.PLANNER, 1)
                
                await asyncio.sleep(180)  # Run every 3 minutes
                
            except Exception as e:
                self.logger.error(f"Generation 3 service mesh management error: {e}")
                await asyncio.sleep(180)
    
    async def generate_plan_generation3(
        self,
        mission: str,
        constraints: Optional[MissionConstraints] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Enhanced plan generation with Generation 3 optimizations."""
        start_time = time.time()
        
        try:
            # Try cache first
            cache_key = f"plan:{hashlib.md5(f'{mission}:{json.dumps(context or {}, sort_keys=True)}'.encode()).hexdigest()}"
            cached_plan = await cached_get(cache_key)
            
            if cached_plan:
                self.logger.info("Retrieved plan from cache")
                return cached_plan
            
            # Use service mesh for distributed planning
            service_mesh = await get_service_mesh()
            
            planning_request = {
                "mission": mission,
                "constraints": constraints.__dict__ if constraints else None,
                "context": context or {},
                "fleet_size": self.swarm_state.num_active_drones,
            }
            
            # Route request through service mesh
            response = await route_service_request(
                source_service="swarm_coordinator",
                target_service_type=ServiceType.PLANNER,
                method="generate_plan",
                payload=planning_request,
                timeout_seconds=30.0,
            )
            
            if response.status_code == 200:
                plan = response.payload
                
                # Cache successful plan
                await cached_put(cache_key, plan, ttl_l1=300.0, ttl_l2=1800.0)
                
                # Record performance
                planning_latency = (time.time() - start_time) * 1000
                record_performance_metrics(latency_ms=planning_latency)
                
                return plan
            else:
                # Fallback to original method
                return await self.generate_plan(mission, constraints, context)
                
        except Exception as e:
            self.logger.error(f"Generation 3 plan generation error: {e}")
            # Fallback to original method
            return await self.generate_plan(mission, constraints, context)
    
    async def execute_mission_generation3(
        self,
        latent_plan: Dict[str, Any],
        monitor_frequency: float = 10.0,
        replan_threshold: float = 0.7,
    ) -> bool:
        """Enhanced mission execution with Generation 3 optimizations."""
        if not self.fleet:
            raise RuntimeError("No fleet connected")
        
        try:
            # Use high-performance communication for drone coordination
            hp_comm = await get_hp_communicator()
            
            # Prepare drone destinations
            drone_destinations = [f"ws://drone_{drone_id}:8080/command" 
                                for drone_id in self.fleet.get_active_drones()]
            
            # Broadcast mission using high-performance communication
            broadcast_result = await hp_comm.broadcast_message(
                destinations=drone_destinations,
                payload={
                    "mission_plan": latent_plan,
                    "execution_mode": "generation3",
                    "performance_optimized": True,
                },
                priority=Priority.HIGH,
                max_concurrent=min(100, len(drone_destinations)),
            )
            
            if broadcast_result["success_rate"] < 0.8:
                self.logger.warning(f"Low broadcast success rate: {broadcast_result['success_rate']:.2f}")
            
            # Enhanced monitoring with AI optimization
            self.mission_status = MissionStatus.EXECUTING
            self.current_mission = latent_plan['description']
            
            return True
            
        except Exception as e:
            self.logger.error(f"Generation 3 mission execution error: {e}")
            # Fallback to original method
            return await self.execute_mission(latent_plan, monitor_frequency, replan_threshold)
    
    async def _collect_generation3_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive Generation 3 performance metrics."""
        try:
            metrics = {}
            
            # Basic performance metrics
            if self.performance_history:
                recent_latencies = [entry.get('latency_ms', 0) for entry in self.context_history[-10:]]
                metrics["avg_latency_ms"] = statistics.mean(recent_latencies) if recent_latencies else 0
            else:
                metrics["avg_latency_ms"] = 0
            
            # System metrics
            try:
                import psutil
                metrics["cpu_usage"] = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                metrics["memory_usage_mb"] = memory.used / (1024 * 1024)
            except ImportError:
                metrics["cpu_usage"] = 50.0  # Default
                metrics["memory_usage_mb"] = 1024.0  # Default
            
            # Fleet metrics
            if self.fleet:
                metrics["active_drones"] = len(self.fleet.get_active_drones())
                metrics["total_drones"] = len(self.fleet.drone_ids)
                metrics["fleet_utilization"] = metrics["active_drones"] / max(1, metrics["total_drones"])
            
            # Communication metrics
            if hasattr(self, 'hp_communicator') and self.hp_communicator:
                comm_stats = self.hp_communicator.get_comprehensive_stats()
                metrics["throughput_rps"] = comm_stats.get("messages_per_second", 0)
                metrics["connection_utilization"] = comm_stats.get("connection_utilization", 0)
            
            # Cache metrics
            if hasattr(self, 'multi_cache') and self.multi_cache:
                cache_stats = self.multi_cache.get_comprehensive_stats()
                metrics["cache_hit_rate"] = cache_stats.get("overall_hit_rate", 0)
            
            # Error rate
            total_operations = len(self.context_history)
            errors = sum(1 for entry in self.context_history if entry.get("error", False))
            metrics["error_rate"] = errors / max(1, total_operations)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting Generation 3 metrics: {e}")
            return {}
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 50.0  # Default
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.virtual_memory().used / (1024 * 1024)
        except ImportError:
            return 1024.0  # Default
    
    def _calculate_current_operational_cost(self) -> float:
        """Calculate current operational cost per hour."""
        # Simplified cost calculation
        base_cost = 0.10  # Base infrastructure cost
        
        if self.fleet:
            # Scale cost with fleet size
            drone_count = len(self.fleet.drone_ids)
            drone_cost = drone_count * 0.01  # $0.01 per drone per hour
            base_cost += drone_cost
        
        # Add compute cost based on performance
        if hasattr(self, '_current_cpu_usage'):
            compute_cost = (self._current_cpu_usage / 100.0) * 0.05
            base_cost += compute_cost
        
        return base_cost
    
    def get_generation3_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive Generation 3 statistics."""
        base_stats = self.get_comprehensive_stats()
        
        # Add Generation 3 specific stats
        gen3_stats = {
            "generation": 3,
            "optimizations_enabled": {
                "ai_performance_optimizer": hasattr(self, 'ai_optimizer'),
                "service_mesh": hasattr(self, 'service_mesh'),
                "cost_optimizer": hasattr(self, 'cost_optimizer'),
                "multi_tier_cache": hasattr(self, 'multi_cache'),
                "hp_communication": hasattr(self, 'hp_communicator'),
            },
        }
        
        # AI optimizer stats
        if hasattr(self, 'ai_optimizer'):
            ai_stats = self.ai_optimizer.get_optimization_stats()
            gen3_stats["ai_optimization"] = ai_stats
        
        # Service mesh stats
        if hasattr(self, 'service_mesh'):
            mesh_stats = self.service_mesh.get_mesh_status()
            gen3_stats["service_mesh"] = mesh_stats
        
        # Cost optimizer stats
        if hasattr(self, 'cost_optimizer'):
            cost_stats = self.cost_optimizer.get_optimization_stats()
            gen3_stats["cost_optimization"] = cost_stats
        
        # Cache stats
        if hasattr(self, 'multi_cache'):
            cache_stats = self.multi_cache.get_comprehensive_stats()
            gen3_stats["multi_tier_cache"] = cache_stats
        
        # Communication stats
        if hasattr(self, 'hp_communicator'):
            comm_stats = self.hp_communicator.get_comprehensive_stats()
            gen3_stats["high_performance_communication"] = comm_stats
        
        # Combine with base stats
        return {**base_stats, "generation3": gen3_stats}
    
    async def benchmark_generation3_performance(
        self,
        test_scenarios: List[Dict[str, Any]],
        duration_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Benchmark Generation 3 performance improvements."""
        self.logger.info(f"Starting Generation 3 performance benchmark for {duration_seconds}s")
        
        benchmark_results = {
            "test_duration": duration_seconds,
            "scenarios_tested": len(test_scenarios),
            "results": [],
            "summary": {},
        }
        
        try:
            start_time = time.time()
            
            for i, scenario in enumerate(test_scenarios):
                scenario_start = time.time()
                scenario_result = {
                    "scenario_id": i,
                    "scenario_type": scenario.get("type", "unknown"),
                    "start_time": scenario_start,
                }
                
                try:
                    # Execute scenario based on type
                    if scenario["type"] == "mission_planning":
                        result = await self.generate_plan_generation3(
                            scenario["mission"],
                            context=scenario.get("context", {})
                        )
                        scenario_result["success"] = "error" not in result
                        scenario_result["latency_ms"] = (time.time() - scenario_start) * 1000
                        
                    elif scenario["type"] == "fleet_coordination":
                        # Simulate fleet coordination
                        if self.fleet:
                            result = await self.execute_mission_generation3({
                                "mission_id": f"benchmark_{i}",
                                "description": scenario["mission"],
                                "latent_code": [1, 2, 3],  # Mock
                            })
                            scenario_result["success"] = result
                            scenario_result["latency_ms"] = (time.time() - scenario_start) * 1000
                        else:
                            scenario_result["success"] = False
                            scenario_result["error"] = "No fleet connected"
                    
                    elif scenario["type"] == "cache_performance":
                        # Test cache performance
                        cache_key = f"benchmark_key_{i}"
                        test_data = {"benchmark": True, "data": list(range(100))}
                        
                        # Put and get from cache
                        await cached_put(cache_key, test_data)
                        retrieved = await cached_get(cache_key)
                        
                        scenario_result["success"] = retrieved is not None
                        scenario_result["latency_ms"] = (time.time() - scenario_start) * 1000
                    
                    else:
                        scenario_result["success"] = False
                        scenario_result["error"] = f"Unknown scenario type: {scenario['type']}"
                    
                except Exception as e:
                    scenario_result["success"] = False
                    scenario_result["error"] = str(e)
                    scenario_result["latency_ms"] = (time.time() - scenario_start) * 1000
                
                benchmark_results["results"].append(scenario_result)
                
                # Check if we've exceeded duration
                if time.time() - start_time > duration_seconds:
                    break
            
            # Calculate summary statistics
            successful_tests = sum(1 for r in benchmark_results["results"] if r.get("success"))
            total_tests = len(benchmark_results["results"])
            
            latencies = [r.get("latency_ms", 0) for r in benchmark_results["results"] 
                        if r.get("success") and "latency_ms" in r]
            
            benchmark_results["summary"] = {
                "success_rate": successful_tests / max(1, total_tests),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
                "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
                "throughput_rps": successful_tests / max(1, duration_seconds),
            }
            
            self.logger.info(f"Benchmark completed: {benchmark_results['summary']['success_rate']:.2%} success rate, "
                           f"{benchmark_results['summary']['avg_latency_ms']:.1f}ms avg latency")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            benchmark_results["error"] = str(e)
            return benchmark_results
    
    async def shutdown_generation3_systems(self):
        """Shutdown Generation 3 systems gracefully."""
        try:
            # Cancel Generation 3 background tasks
            if hasattr(self, '_gen3_tasks'):
                for task in self._gen3_tasks:
                    task.cancel()
                await asyncio.gather(*self._gen3_tasks, return_exceptions=True)
            
            # Shutdown components
            if hasattr(self, 'hp_communicator') and self.hp_communicator:
                await self.hp_communicator.stop()
            
            if hasattr(self, 'service_mesh') and self.service_mesh:
                await self.service_mesh.stop()
            
            if hasattr(self, 'multi_cache') and self.multi_cache:
                await self.multi_cache.stop()
            
            self.logger.info("Generation 3 systems shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error shutting down Generation 3 systems: {e}")