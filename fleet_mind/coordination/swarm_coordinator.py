"""Central SwarmCoordinator for LLM-powered drone fleet management."""

import asyncio
import time
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
    async def generate_plan(
        self,
        mission: str,
        constraints: Optional[MissionConstraints] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate mission plan using LLM.
        
        Args:
            mission: Mission description in natural language
            constraints: Mission execution constraints
            context: Additional context (world state, drone capabilities)
            
        Returns:
            Generated mission plan with latent encoding
        """
        if not self.fleet:
            raise RuntimeError("No fleet connected")
            
        # Use provided constraints or defaults
        active_constraints = constraints or self.safety_constraints
        
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
        
        # Generate plan with LLM
        start_time = time.time()
        plan = await self.llm_planner.generate_plan(planning_context)
        planning_latency = (time.time() - start_time) * 1000  # ms
        
        # Encode to latent representation - handle both old and new plan formats
        actions_to_encode = []
        if 'actions' in plan:
            actions_to_encode = plan['actions']
        elif 'action_sequences' in plan and len(plan['action_sequences']) > 0:
            # Extract actions from action sequences
            for seq in plan['action_sequences']:
                if 'actions' in seq:
                    actions_to_encode.extend(seq['actions'])
        
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
                    print(f"Callback error for {event}: {e}")

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