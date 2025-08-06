"""Drone fleet management and coordination."""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# NumPy import with fallback handling
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    # Fallback implementation for numpy functions
    class np:
        @staticmethod
        def array(data): return data
        class random:
            @staticmethod
            def uniform(low, high, size=None): return low + (high-low) * 0.5
            @staticmethod
            def randn(*shape): return 0.5
        class linalg:
            @staticmethod
            def norm(x): return sum(abs(i) for i in x) if hasattr(x, '__iter__') else abs(x)
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available in drone_fleet, using fallback")


class DroneStatus(Enum):
    """Drone operational status."""
    IDLE = "idle"
    ACTIVE = "active"
    MISSION = "mission"
    RETURNING = "returning"
    LANDING = "landing"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    OFFLINE = "offline"


class DroneCapability(Enum):
    """Drone capability types."""
    BASIC_FLIGHT = "basic_flight"
    PRECISION_HOVER = "precision_hover"
    FORMATION_FLIGHT = "formation_flight"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    CAMERA = "camera"
    LIDAR = "lidar"
    THERMAL = "thermal"
    GPS = "gps"
    COMMUNICATION = "communication"
    CARGO_DELIVERY = "cargo_delivery"


@dataclass
class DroneState:
    """Current state of a drone."""
    drone_id: str
    status: DroneStatus = DroneStatus.IDLE
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # vx, vy, vz
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    battery_percent: float = 100.0
    health_score: float = 1.0
    last_update: float = field(default_factory=time.time)
    mission_progress: float = 0.0
    capabilities: Set[DroneCapability] = field(default_factory=set)
    communication_quality: float = 1.0
    sensor_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FleetConfiguration:
    """Fleet-wide configuration settings."""
    max_altitude: float = 120.0  # meters
    min_separation: float = 5.0   # meters
    max_velocity: float = 10.0    # m/s
    communication_range: float = 1000.0  # meters
    battery_warning_threshold: float = 20.0  # percent
    battery_critical_threshold: float = 10.0  # percent
    health_warning_threshold: float = 0.7
    health_critical_threshold: float = 0.5


class DroneFleet:
    """Management system for coordinating multiple drones.
    
    Handles drone state tracking, health monitoring, capability management,
    and provides high-level fleet coordination interfaces.
    """
    
    def __init__(
        self,
        drone_ids: List[str],
        communication_protocol: str = "webrtc",
        topology: str = "mesh",
        config: Optional[FleetConfiguration] = None
    ):
        """Initialize drone fleet.
        
        Args:
            drone_ids: List of unique drone identifiers
            communication_protocol: Communication protocol (webrtc, mqtt, etc.)
            topology: Network topology (mesh, star, hierarchical)
            config: Fleet configuration parameters
        """
        self.drone_ids = drone_ids
        self.communication_protocol = communication_protocol
        self.topology = topology
        self.config = config or FleetConfiguration()
        
        # Drone state management
        self.drone_states: Dict[str, DroneState] = {}
        self.drone_capabilities: Dict[str, Set[DroneCapability]] = {}
        
        # Fleet status tracking
        self.active_drones: Set[str] = set()
        self.failed_drones: Set[str] = set()
        self.maintenance_drones: Set[str] = set()
        
        # Performance metrics
        self.fleet_metrics = {
            'total_flight_time': 0.0,
            'total_distance': 0.0,
            'missions_completed': 0,
            'average_battery': 100.0,
            'average_health': 1.0,
            'communication_quality': 1.0,
        }
        
        # Initialize drone states
        self._initialize_drone_states()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

    def _initialize_drone_states(self) -> None:
        """Initialize states for all drones."""
        default_capabilities = {
            DroneCapability.BASIC_FLIGHT,
            DroneCapability.GPS,
            DroneCapability.COMMUNICATION,
            DroneCapability.OBSTACLE_AVOIDANCE,
        }
        
        for drone_id in self.drone_ids:
            self.drone_states[drone_id] = DroneState(
                drone_id=drone_id,
                capabilities=default_capabilities.copy()
            )
            self.drone_capabilities[drone_id] = default_capabilities.copy()
            self.active_drones.add(drone_id)

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        print(f"Started monitoring for {len(self.drone_ids)} drones")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        print("Stopped fleet monitoring")

    def get_active_drones(self) -> List[str]:
        """Get list of currently active drone IDs."""
        return list(self.active_drones)

    def get_failed_drones(self) -> List[str]:
        """Get list of failed drone IDs."""
        return list(self.failed_drones)

    def get_drone_state(self, drone_id: str) -> Optional[DroneState]:
        """Get current state of specific drone."""
        return self.drone_states.get(drone_id)

    def get_fleet_status(self) -> Dict[str, Any]:
        """Get comprehensive fleet status."""
        return {
            'total_drones': len(self.drone_ids),
            'active_drones': len(self.active_drones),
            'failed_drones': len(self.failed_drones),
            'maintenance_drones': len(self.maintenance_drones),
            'average_battery': self.get_average_battery(),
            'average_health': self.get_average_health(),
            'communication_quality': self.fleet_metrics['communication_quality'],
            'missions_completed': self.fleet_metrics['missions_completed'],
            'total_flight_time_hours': self.fleet_metrics['total_flight_time'] / 3600,
        }

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities overview for all drones."""
        return {
            drone_id: [cap.value for cap in capabilities]
            for drone_id, capabilities in self.drone_capabilities.items()
        }

    def get_drones_with_capability(self, capability: DroneCapability) -> List[str]:
        """Get drones that have specific capability."""
        return [
            drone_id for drone_id, capabilities in self.drone_capabilities.items()
            if capability in capabilities and drone_id in self.active_drones
        ]

    def get_average_battery(self) -> float:
        """Get average battery level across active drones."""
        if not self.active_drones:
            return 0.0
        
        total_battery = sum(
            self.drone_states[drone_id].battery_percent
            for drone_id in self.active_drones
            if drone_id in self.drone_states
        )
        
        return total_battery / len(self.active_drones)

    def get_average_health(self) -> float:
        """Get average health score across active drones."""
        if not self.active_drones:
            return 0.0
        
        total_health = sum(
            self.drone_states[drone_id].health_score
            for drone_id in self.active_drones
            if drone_id in self.drone_states
        )
        
        return total_health / len(self.active_drones)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status for all drones."""
        health_status = {
            'healthy': [],
            'warning': [],
            'critical': [],
            'failed': [],
        }
        
        for drone_id in self.drone_ids:
            if drone_id not in self.drone_states:
                health_status['failed'].append(drone_id)
                continue
            
            state = self.drone_states[drone_id]
            
            if drone_id in self.failed_drones:
                health_status['failed'].append(drone_id)
            elif (state.health_score < self.config.health_critical_threshold or
                  state.battery_percent < self.config.battery_critical_threshold):
                health_status['critical'].append(drone_id)
            elif (state.health_score < self.config.health_warning_threshold or
                  state.battery_percent < self.config.battery_warning_threshold):
                health_status['warning'].append(drone_id)
            else:
                health_status['healthy'].append(drone_id)
        
        return health_status

    def update_drone_state(
        self,
        drone_id: str,
        position: Optional[Tuple[float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        battery_percent: Optional[float] = None,
        health_score: Optional[float] = None,
        status: Optional[DroneStatus] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update state information for specific drone.
        
        Args:
            drone_id: Drone identifier
            position: New position (x, y, z)
            velocity: New velocity (vx, vy, vz)
            battery_percent: Battery level (0-100)
            health_score: Health score (0-1)
            status: Operational status
            sensor_data: Sensor readings
            
        Returns:
            True if update was successful
        """
        if drone_id not in self.drone_states:
            return False
        
        state = self.drone_states[drone_id]
        
        # Update provided fields
        if position is not None:
            state.position = position
        if velocity is not None:
            state.velocity = velocity
        if battery_percent is not None:
            state.battery_percent = max(0.0, min(100.0, battery_percent))
        if health_score is not None:
            state.health_score = max(0.0, min(1.0, health_score))
        if status is not None:
            state.status = status
        if sensor_data is not None:
            state.sensor_data.update(sensor_data)
        
        state.last_update = time.time()
        
        # Update fleet-level tracking based on status
        self._update_drone_tracking(drone_id, state)
        
        return True

    def add_drone_capability(self, drone_id: str, capability: DroneCapability) -> bool:
        """Add capability to specific drone."""
        if drone_id in self.drone_capabilities:
            self.drone_capabilities[drone_id].add(capability)
            if drone_id in self.drone_states:
                self.drone_states[drone_id].capabilities.add(capability)
            return True
        return False

    def remove_drone_capability(self, drone_id: str, capability: DroneCapability) -> bool:
        """Remove capability from specific drone."""
        if drone_id in self.drone_capabilities:
            self.drone_capabilities[drone_id].discard(capability)
            if drone_id in self.drone_states:
                self.drone_states[drone_id].capabilities.discard(capability)
            return True
        return False

    def get_formation_candidates(
        self,
        formation_type: str,
        min_drones: int = 3,
        required_capabilities: Optional[List[DroneCapability]] = None
    ) -> List[str]:
        """Get drones suitable for formation flying.
        
        Args:
            formation_type: Type of formation
            min_drones: Minimum number of drones needed
            required_capabilities: Required capabilities for formation
            
        Returns:
            List of suitable drone IDs
        """
        candidates = []
        required_caps = required_capabilities or [DroneCapability.FORMATION_FLIGHT]
        
        for drone_id in self.active_drones:
            if drone_id not in self.drone_states:
                continue
            
            state = self.drone_states[drone_id]
            
            # Check basic health requirements
            if (state.battery_percent < self.config.battery_warning_threshold or
                state.health_score < self.config.health_warning_threshold):
                continue
            
            # Check required capabilities
            if all(cap in state.capabilities for cap in required_caps):
                candidates.append(drone_id)
        
        return candidates[:min(len(candidates), min_drones * 2)]  # Return up to 2x needed

    async def execute_emergency_procedure(self, procedure: str) -> Dict[str, bool]:
        """Execute emergency procedure across fleet.
        
        Args:
            procedure: Emergency procedure name
            
        Returns:
            Success status for each drone
        """
        results = {}
        
        for drone_id in self.active_drones:
            try:
                if procedure == "emergency_land":
                    results[drone_id] = await self._emergency_land_drone(drone_id)
                elif procedure == "return_home":
                    results[drone_id] = await self._return_home_drone(drone_id)
                elif procedure == "hold_position":
                    results[drone_id] = await self._hold_position_drone(drone_id)
                else:
                    results[drone_id] = False
                    
            except Exception as e:
                print(f"Emergency procedure failed for drone {drone_id}: {e}")
                results[drone_id] = False
        
        return results

    async def _monitoring_loop(self) -> None:
        """Background loop for fleet monitoring."""
        try:
            while self._running:
                # Update fleet metrics
                self._update_fleet_metrics()
                
                # Check for stale data
                self._check_stale_data()
                
                # Simulate telemetry updates (in real implementation, this comes from drones)
                self._simulate_telemetry_updates()
                
                await asyncio.sleep(1.0)  # Update every second
                
        except asyncio.CancelledError:
            print("Fleet monitoring loop cancelled")

    async def _health_check_loop(self) -> None:
        """Background loop for health monitoring."""
        try:
            while self._running:
                # Check drone health and take action if needed
                await self._perform_health_checks()
                
                await asyncio.sleep(5.0)  # Health checks every 5 seconds
                
        except asyncio.CancelledError:
            print("Health check loop cancelled")

    def _update_drone_tracking(self, drone_id: str, state: DroneState) -> None:
        """Update fleet-level tracking based on drone state."""
        # Update active/failed/maintenance sets
        if state.status == DroneStatus.FAILED:
            self.active_drones.discard(drone_id)
            self.failed_drones.add(drone_id)
            self.maintenance_drones.discard(drone_id)
        elif state.status == DroneStatus.MAINTENANCE:
            self.active_drones.discard(drone_id)
            self.failed_drones.discard(drone_id)
            self.maintenance_drones.add(drone_id)
        elif state.status in [DroneStatus.IDLE, DroneStatus.ACTIVE, DroneStatus.MISSION]:
            self.active_drones.add(drone_id)
            self.failed_drones.discard(drone_id)
            self.maintenance_drones.discard(drone_id)

    def _update_fleet_metrics(self) -> None:
        """Update fleet-wide performance metrics."""
        if not self.active_drones:
            return
        
        # Update averages
        self.fleet_metrics['average_battery'] = self.get_average_battery()
        self.fleet_metrics['average_health'] = self.get_average_health()
        
        # Calculate communication quality (simplified)
        comm_quality = sum(
            self.drone_states[drone_id].communication_quality
            for drone_id in self.active_drones
            if drone_id in self.drone_states
        ) / len(self.active_drones)
        
        self.fleet_metrics['communication_quality'] = comm_quality

    def _check_stale_data(self) -> None:
        """Check for drones with stale telemetry data."""
        current_time = time.time()
        stale_threshold = 30.0  # seconds
        
        for drone_id in list(self.active_drones):
            if drone_id in self.drone_states:
                state = self.drone_states[drone_id]
                if current_time - state.last_update > stale_threshold:
                    print(f"Warning: Stale data from drone {drone_id}")
                    # Could mark as communication issue or offline

    def _simulate_telemetry_updates(self) -> None:
        """Simulate telemetry updates for testing (remove in production)."""
        for drone_id in self.active_drones:
            if drone_id in self.drone_states:
                state = self.drone_states[drone_id]
                
                # Simulate battery drain
                state.battery_percent = max(0, state.battery_percent - 0.01)
                
                # Simulate minor position changes
                noise = np.random.normal(0, 0.1, 3)
                state.position = (
                    state.position[0] + noise[0],
                    state.position[1] + noise[1],
                    state.position[2] + noise[2]
                )
                
                # Simulate health fluctuations
                state.health_score = max(0.5, min(1.0, 
                    state.health_score + np.random.normal(0, 0.01)
                ))
                
                state.last_update = time.time()

    async def _perform_health_checks(self) -> None:
        """Perform health checks and take corrective actions."""
        for drone_id in list(self.active_drones):
            if drone_id not in self.drone_states:
                continue
            
            state = self.drone_states[drone_id]
            
            # Check critical battery
            if state.battery_percent < self.config.battery_critical_threshold:
                print(f"CRITICAL: Drone {drone_id} battery at {state.battery_percent:.1f}%")
                await self._emergency_land_drone(drone_id)
            
            # Check critical health
            elif state.health_score < self.config.health_critical_threshold:
                print(f"CRITICAL: Drone {drone_id} health at {state.health_score:.2f}")
                await self._return_home_drone(drone_id)
            
            # Check warning levels
            elif (state.battery_percent < self.config.battery_warning_threshold or
                  state.health_score < self.config.health_warning_threshold):
                print(f"WARNING: Drone {drone_id} needs attention")

    async def _emergency_land_drone(self, drone_id: str) -> bool:
        """Execute emergency landing for specific drone."""
        if drone_id in self.drone_states:
            self.drone_states[drone_id].status = DroneStatus.LANDING
            print(f"Emergency landing initiated for drone {drone_id}")
            # In real implementation, send landing command
            return True
        return False

    async def _return_home_drone(self, drone_id: str) -> bool:
        """Execute return-to-home for specific drone."""
        if drone_id in self.drone_states:
            self.drone_states[drone_id].status = DroneStatus.RETURNING
            print(f"Return-to-home initiated for drone {drone_id}")
            # In real implementation, send return command
            return True
        return False

    async def _hold_position_drone(self, drone_id: str) -> bool:
        """Execute hold position for specific drone."""
        if drone_id in self.drone_states:
            self.drone_states[drone_id].velocity = (0.0, 0.0, 0.0)
            print(f"Hold position initiated for drone {drone_id}")
            # In real implementation, send hold command
            return True
        return False
    
    def get_formation_quality_score(self, target_formation: Dict[str, Any]) -> float:
        """Calculate formation quality score based on current positions.
        
        Args:
            target_formation: Target formation specification
            
        Returns:
            Quality score between 0-1 (1 = perfect formation)
        """
        formation_type = target_formation.get('formation_type', 'grid')
        spacing = target_formation.get('spacing_meters', 10.0)
        
        if len(self.active_drones) < 2:
            return 1.0  # Perfect for single drone
        
        # Calculate formation error
        positions = []
        for drone_id in self.active_drones:
            if drone_id in self.drone_states:
                pos = self.drone_states[drone_id].position
                positions.append(pos)
        
        if not positions:
            return 0.0
        
        # Simple formation quality based on spacing consistency
        if formation_type == 'grid':
            return self._calculate_grid_quality(positions, spacing)
        elif formation_type == 'line':
            return self._calculate_line_quality(positions, spacing)
        elif formation_type == 'v_formation':
            return self._calculate_v_formation_quality(positions, spacing)
        else:
            # Default: measure spacing consistency
            return self._calculate_spacing_consistency(positions, spacing)
    
    def _calculate_grid_quality(self, positions: List[Tuple[float, float, float]], spacing: float) -> float:
        """Calculate grid formation quality."""
        if len(positions) < 4:
            return self._calculate_spacing_consistency(positions, spacing)
        
        # For grid, check if positions form regular grid pattern
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Check spacing consistency in both directions
        x_sorted = sorted(set(x_coords))
        y_sorted = sorted(set(y_coords))
        
        x_spacing_error = 0.0
        if len(x_sorted) > 1:
            x_diffs = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
            target_x_spacing = spacing
            x_spacing_error = sum(abs(diff - target_x_spacing) for diff in x_diffs) / len(x_diffs)
        
        y_spacing_error = 0.0
        if len(y_sorted) > 1:
            y_diffs = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]
            target_y_spacing = spacing
            y_spacing_error = sum(abs(diff - target_y_spacing) for diff in y_diffs) / len(y_diffs)
        
        avg_error = (x_spacing_error + y_spacing_error) / 2
        return max(0.0, 1.0 - (avg_error / spacing))
    
    def _calculate_line_quality(self, positions: List[Tuple[float, float, float]], spacing: float) -> float:
        """Calculate line formation quality."""
        if len(positions) < 2:
            return 1.0
        
        # Sort positions by distance from first position
        ref_pos = positions[0]
        sorted_positions = sorted(positions, key=lambda p: 
                                (p[0] - ref_pos[0])**2 + (p[1] - ref_pos[1])**2)
        
        # Check spacing between consecutive drones
        spacing_errors = []
        for i in range(len(sorted_positions) - 1):
            pos1 = sorted_positions[i]
            pos2 = sorted_positions[i + 1]
            actual_dist = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
            error = abs(actual_dist - spacing)
            spacing_errors.append(error)
        
        if spacing_errors:
            avg_error = sum(spacing_errors) / len(spacing_errors)
            return max(0.0, 1.0 - (avg_error / spacing))
        
        return 1.0
    
    def _calculate_v_formation_quality(self, positions: List[Tuple[float, float, float]], spacing: float) -> float:
        """Calculate V-formation quality."""
        if len(positions) < 3:
            return self._calculate_line_quality(positions, spacing)
        
        # For V-formation, find leader (front-most) and check if others form V behind
        # This is a simplified check - real implementation would be more sophisticated
        return self._calculate_spacing_consistency(positions, spacing)
    
    def _calculate_spacing_consistency(self, positions: List[Tuple[float, float, float]], target_spacing: float) -> float:
        """Calculate general spacing consistency quality."""
        if len(positions) < 2:
            return 1.0
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1 = positions[i]
                pos2 = positions[j]
                dist = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)**0.5
                distances.append(dist)
        
        # Find distances closest to target spacing (neighboring drones)
        distances.sort()
        min_expected = len(positions) - 1  # Minimum edges in formation
        neighbor_distances = distances[:min_expected]
        
        # Calculate error from target spacing
        spacing_errors = [abs(dist - target_spacing) for dist in neighbor_distances]
        avg_error = sum(spacing_errors) / len(spacing_errors) if spacing_errors else 0
        
        return max(0.0, 1.0 - (avg_error / target_spacing))
    
    async def auto_heal_fleet(self) -> Dict[str, Any]:
        """Automatically attempt to heal fleet issues.
        
        Returns:
            Dictionary of healing actions taken
        """
        healing_actions = {
            'drones_recovered': [],
            'issues_resolved': [],
            'recommendations': [],
        }
        
        # Try to recover failed drones
        for drone_id in list(self.failed_drones):
            if drone_id in self.drone_states:
                state = self.drone_states[drone_id]
                
                # Check if it's recoverable (maybe just communication issue)
                if time.time() - state.last_update < 60:  # Within last minute
                    # Attempt recovery by resetting status
                    state.status = DroneStatus.IDLE
                    state.health_score = max(0.5, state.health_score)
                    
                    self.failed_drones.discard(drone_id)
                    self.active_drones.add(drone_id)
                    healing_actions['drones_recovered'].append(drone_id)
                    healing_actions['issues_resolved'].append(f"Recovered communication with {drone_id}")
        
        # Check for maintenance drones that could return to service
        for drone_id in list(self.maintenance_drones):
            if drone_id in self.drone_states:
                state = self.drone_states[drone_id]
                
                # If health improved and battery sufficient, return to service
                if (state.health_score > self.config.health_warning_threshold and
                    state.battery_percent > self.config.battery_warning_threshold):
                    
                    state.status = DroneStatus.IDLE
                    self.maintenance_drones.discard(drone_id)
                    self.active_drones.add(drone_id)
                    healing_actions['drones_recovered'].append(drone_id)
                    healing_actions['issues_resolved'].append(f"Returned {drone_id} to service after maintenance")
        
        # Generate recommendations for unrecoverable issues
        if len(self.failed_drones) > 0:
            healing_actions['recommendations'].append(f"Manual intervention needed for {len(self.failed_drones)} failed drones")
        
        if len(healing_actions['drones_recovered']) > 0:
            print(f"Fleet auto-healing recovered {len(healing_actions['drones_recovered'])} drones")
        
        return healing_actions