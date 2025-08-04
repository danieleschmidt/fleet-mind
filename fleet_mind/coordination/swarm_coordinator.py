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
    average_battery: float = 100.0
    mission_progress: float = 0.0
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
    ):
        """Initialize SwarmCoordinator.
        
        Args:
            llm_model: LLM model identifier (default: gpt-4o)
            latent_dim: Latent encoding dimension
            compression_ratio: Target compression ratio
            max_drones: Maximum fleet size
            update_rate: Planning frequency in Hz
            safety_constraints: Mission safety parameters
        """
        self.llm_model = llm_model
        self.latent_dim = latent_dim
        self.compression_ratio = compression_ratio
        self.max_drones = max_drones
        self.update_rate = update_rate
        self.safety_constraints = safety_constraints or MissionConstraints()
        
        # Core components
        self.llm_planner = LLMPlanner(model=llm_model)
        self.latent_encoder = LatentEncoder(
            input_dim=4096,
            latent_dim=latent_dim,
            compression_type="learned_vqvae"
        )
        self.webrtc_streamer = WebRTCStreamer()
        
        # State management
        self.fleet: Optional[DroneFleet] = None
        self.mission_status = MissionStatus.IDLE
        self.current_mission: Optional[str] = None
        self.swarm_state = SwarmState()
        self.context_history: List[Dict[str, Any]] = []
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'mission_start': [],
            'mission_complete': [],
            'drone_failure': [],
            'emergency_stop': [],
        }
        
        # Async task management
        self._planning_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect_fleet(self, fleet: DroneFleet) -> None:
        """Connect and initialize drone fleet.
        
        Args:
            fleet: DroneFleet instance to coordinate
        """
        self.fleet = fleet
        await self.webrtc_streamer.initialize(fleet.drone_ids)
        
        # Update swarm state
        self.swarm_state.num_total_drones = len(fleet.drone_ids)
        self.swarm_state.num_active_drones = len(fleet.get_active_drones())
        
        print(f"Connected to fleet of {len(fleet.drone_ids)} drones")

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
        
        # Encode to latent representation
        latent_plan = self.latent_encoder.encode(plan['actions'])
        
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
        return sum(latencies) / len(latencies) if latencies else 0.0