"""Bio-Inspired Swarm Intelligence for Drone Coordination.

Advanced biomimetic algorithms inspired by natural swarm behaviors:
- Ant colony optimization for path planning
- Bee colony algorithms for resource allocation  
- Bird flocking for formation control
- Fish schooling for obstacle avoidance
"""

import math
import random
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class SwarmBehaviorType(Enum):
    FLOCKING = "flocking"
    FORAGING = "foraging"
    SWARMING = "swarming"
    SCHOOLING = "schooling"
    HERDING = "herding"

class FlockingBehavior(Enum):
    SEPARATION = "separation"
    ALIGNMENT = "alignment"
    COHESION = "cohesion"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    LEADER_FOLLOWING = "leader_following"

@dataclass
class EmergentPattern:
    """Emergent behavior pattern detected in swarm."""
    pattern_type: str
    confidence: float
    participants: List[str]
    centroid: Tuple[float, float, float]
    formation_quality: float
    stability_score: float
    detected_at: float

@dataclass
class BioDrone:
    """Biological-inspired drone representation."""
    id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    energy_level: float
    pheromone_trail: List[Tuple[float, float, float]]
    social_connections: List[str]
    behavioral_state: SwarmBehaviorType
    fitness_score: float = 0.0
    
class BioSwarmIntelligence:
    """Bio-inspired swarm intelligence system."""
    
    def __init__(self, 
                 swarm_size: int = 100,
                 communication_range: float = 50.0,
                 pheromone_decay_rate: float = 0.1):
        self.swarm_size = swarm_size
        self.communication_range = communication_range
        self.pheromone_decay_rate = pheromone_decay_rate
        
        self.bio_drones: Dict[str, BioDrone] = {}
        self.pheromone_map: Dict[Tuple[int, int, int], float] = defaultdict(float)
        self.emergent_patterns: List[EmergentPattern] = []
        self.social_network: Dict[str, List[str]] = defaultdict(list)
        
        # Behavioral parameters
        self.flocking_weights = {
            FlockingBehavior.SEPARATION: 2.0,
            FlockingBehavior.ALIGNMENT: 1.0,
            FlockingBehavior.COHESION: 1.0,
            FlockingBehavior.OBSTACLE_AVOIDANCE: 3.0,
            FlockingBehavior.LEADER_FOLLOWING: 1.5
        }
        
        self.behavior_stats = {
            'collective_intelligence_score': 0.0,
            'swarm_cohesion': 0.0,
            'adaptation_rate': 0.0,
            'emergence_frequency': 0.0
        }
    
    async def initialize_bio_swarm(self, drone_positions: Dict[str, Tuple[float, float, float]]):
        """Initialize bio-inspired swarm from drone positions."""
        
        for drone_id, position in drone_positions.items():
            bio_drone = BioDrone(
                id=drone_id,
                position=position,
                velocity=(0.0, 0.0, 0.0),
                energy_level=100.0,
                pheromone_trail=[],
                social_connections=[],
                behavioral_state=SwarmBehaviorType.FLOCKING,
                fitness_score=random.uniform(0.5, 1.0)
            )
            self.bio_drones[drone_id] = bio_drone
        
        # Establish initial social connections
        await self._establish_social_network()
    
    async def execute_flocking_behavior(self, 
                                      target_formation: str = "v_formation") -> Dict[str, Tuple[float, float, float]]:
        """Execute bio-inspired flocking behavior."""
        
        new_positions = {}
        
        for drone_id, bio_drone in self.bio_drones.items():
            # Calculate flocking forces
            separation_force = await self._calculate_separation_force(bio_drone)
            alignment_force = await self._calculate_alignment_force(bio_drone)
            cohesion_force = await self._calculate_cohesion_force(bio_drone)
            obstacle_avoidance_force = await self._calculate_obstacle_avoidance_force(bio_drone)
            
            # Combine forces with weights
            total_force = (
                separation_force[0] * self.flocking_weights[FlockingBehavior.SEPARATION] +
                alignment_force[0] * self.flocking_weights[FlockingBehavior.ALIGNMENT] +
                cohesion_force[0] * self.flocking_weights[FlockingBehavior.COHESION] +
                obstacle_avoidance_force[0] * self.flocking_weights[FlockingBehavior.OBSTACLE_AVOIDANCE],
                
                separation_force[1] * self.flocking_weights[FlockingBehavior.SEPARATION] +
                alignment_force[1] * self.flocking_weights[FlockingBehavior.ALIGNMENT] +
                cohesion_force[1] * self.flocking_weights[FlockingBehavior.COHESION] +
                obstacle_avoidance_force[1] * self.flocking_weights[FlockingBehavior.OBSTACLE_AVOIDANCE],
                
                separation_force[2] * self.flocking_weights[FlockingBehavior.SEPARATION] +
                alignment_force[2] * self.flocking_weights[FlockingBehavior.ALIGNMENT] +
                cohesion_force[2] * self.flocking_weights[FlockingBehavior.COHESION] +
                obstacle_avoidance_force[2] * self.flocking_weights[FlockingBehavior.OBSTACLE_AVOIDANCE]
            )
            
            # Update velocity and position
            max_force = 2.0
            total_force = self._limit_force(total_force, max_force)
            
            new_velocity = (
                bio_drone.velocity[0] + total_force[0] * 0.1,
                bio_drone.velocity[1] + total_force[1] * 0.1,
                bio_drone.velocity[2] + total_force[2] * 0.1
            )
            
            max_speed = 10.0
            new_velocity = self._limit_velocity(new_velocity, max_speed)
            
            new_position = (
                bio_drone.position[0] + new_velocity[0] * 0.1,
                bio_drone.position[1] + new_velocity[1] * 0.1,
                bio_drone.position[2] + new_velocity[2] * 0.1
            )
            
            # Update bio drone
            bio_drone.velocity = new_velocity
            bio_drone.position = new_position
            new_positions[drone_id] = new_position
            
            # Update pheromone trail
            bio_drone.pheromone_trail.append(new_position)
            if len(bio_drone.pheromone_trail) > 10:
                bio_drone.pheromone_trail = bio_drone.pheromone_trail[-10:]
        
        # Detect emergent patterns
        await self._detect_emergent_patterns()
        
        return new_positions
    
    async def execute_ant_colony_optimization(self, 
                                            start_points: List[Tuple[float, float, float]],
                                            target_points: List[Tuple[float, float, float]],
                                            obstacles: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
        """Execute ant colony optimization for path planning."""
        
        paths = []
        
        for i, (start, target) in enumerate(zip(start_points, target_points)):
            # Initialize ant colony for this path
            ant_paths = await self._ant_colony_path_search(start, target, obstacles)
            
            # Select best path based on pheromone concentration
            best_path = self._select_best_ant_path(ant_paths)
            paths.append(best_path)
            
            # Update global pheromone map
            await self._update_pheromone_trail(best_path, 1.0)
        
        return paths
    
    async def execute_bee_colony_algorithm(self, 
                                         resource_locations: List[Tuple[float, float, float]],
                                         drone_capacities: Dict[str, float]) -> Dict[str, str]:
        """Execute bee colony algorithm for resource allocation."""
        
        allocations = {}
        
        # Scout bees explore resource sites
        site_qualities = await self._scout_resource_sites(resource_locations)
        
        # Worker bees evaluate and communicate quality
        for drone_id, capacity in drone_capacities.items():
            if drone_id in self.bio_drones:
                best_site = await self._bee_site_selection(
                    self.bio_drones[drone_id], 
                    resource_locations, 
                    site_qualities,
                    capacity
                )
                allocations[drone_id] = f"resource_{best_site}"
                
                # Perform waggle dance to communicate findings
                await self._perform_waggle_dance(drone_id, best_site, site_qualities[best_site])
        
        return allocations
    
    async def execute_fish_schooling(self, 
                                   predator_locations: List[Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """Execute fish schooling behavior for predator avoidance."""
        
        new_positions = {}
        
        for drone_id, bio_drone in self.bio_drones.items():
            # Calculate predator avoidance force
            predator_avoidance = await self._calculate_predator_avoidance(bio_drone, predator_locations)
            
            # Calculate schooling forces
            neighbor_alignment = await self._calculate_neighbor_alignment(bio_drone)
            group_cohesion = await self._calculate_group_cohesion(bio_drone)
            
            # Combine forces for schooling behavior
            schooling_force = (
                predator_avoidance[0] * 3.0 + neighbor_alignment[0] + group_cohesion[0],
                predator_avoidance[1] * 3.0 + neighbor_alignment[1] + group_cohesion[1],
                predator_avoidance[2] * 3.0 + neighbor_alignment[2] + group_cohesion[2]
            )
            
            # Update position with schooling behavior
            new_velocity = (
                bio_drone.velocity[0] + schooling_force[0] * 0.1,
                bio_drone.velocity[1] + schooling_force[1] * 0.1,
                bio_drone.velocity[2] + schooling_force[2] * 0.1
            )
            
            new_velocity = self._limit_velocity(new_velocity, 15.0)  # Higher speed for escape
            
            new_position = (
                bio_drone.position[0] + new_velocity[0] * 0.1,
                bio_drone.position[1] + new_velocity[1] * 0.1,
                bio_drone.position[2] + new_velocity[2] * 0.1
            )
            
            bio_drone.velocity = new_velocity
            bio_drone.position = new_position
            new_positions[drone_id] = new_position
        
        return new_positions
    
    async def _establish_social_network(self):
        """Establish social connections between bio drones."""
        
        for drone_id, bio_drone in self.bio_drones.items():
            # Find neighbors within communication range
            neighbors = []
            
            for other_id, other_drone in self.bio_drones.items():
                if other_id != drone_id:
                    distance = self._calculate_distance(bio_drone.position, other_drone.position)
                    
                    if distance <= self.communication_range:
                        neighbors.append(other_id)
            
            bio_drone.social_connections = neighbors
            self.social_network[drone_id] = neighbors
    
    async def _calculate_separation_force(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate separation force to avoid crowding."""
        
        force = (0.0, 0.0, 0.0)
        neighbor_count = 0
        separation_radius = 20.0
        
        for neighbor_id in bio_drone.social_connections:
            if neighbor_id in self.bio_drones:
                neighbor = self.bio_drones[neighbor_id]
                distance = self._calculate_distance(bio_drone.position, neighbor.position)
                
                if 0 < distance < separation_radius:
                    # Calculate repulsion force
                    diff = (
                        bio_drone.position[0] - neighbor.position[0],
                        bio_drone.position[1] - neighbor.position[1],
                        bio_drone.position[2] - neighbor.position[2]
                    )
                    
                    # Normalize and scale by inverse distance
                    magnitude = max(distance, 0.1)
                    normalized_diff = (diff[0] / magnitude, diff[1] / magnitude, diff[2] / magnitude)
                    
                    force = (
                        force[0] + normalized_diff[0] / distance,
                        force[1] + normalized_diff[1] / distance,
                        force[2] + normalized_diff[2] / distance
                    )
                    neighbor_count += 1
        
        if neighbor_count > 0:
            force = (force[0] / neighbor_count, force[1] / neighbor_count, force[2] / neighbor_count)
        
        return force
    
    async def _calculate_alignment_force(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate alignment force to match neighbor velocities."""
        
        avg_velocity = (0.0, 0.0, 0.0)
        neighbor_count = 0
        alignment_radius = 30.0
        
        for neighbor_id in bio_drone.social_connections:
            if neighbor_id in self.bio_drones:
                neighbor = self.bio_drones[neighbor_id]
                distance = self._calculate_distance(bio_drone.position, neighbor.position)
                
                if distance < alignment_radius:
                    avg_velocity = (
                        avg_velocity[0] + neighbor.velocity[0],
                        avg_velocity[1] + neighbor.velocity[1],
                        avg_velocity[2] + neighbor.velocity[2]
                    )
                    neighbor_count += 1
        
        if neighbor_count > 0:
            avg_velocity = (
                avg_velocity[0] / neighbor_count,
                avg_velocity[1] / neighbor_count,
                avg_velocity[2] / neighbor_count
            )
            
            # Calculate desired velocity change
            force = (
                avg_velocity[0] - bio_drone.velocity[0],
                avg_velocity[1] - bio_drone.velocity[1],
                avg_velocity[2] - bio_drone.velocity[2]
            )
        else:
            force = (0.0, 0.0, 0.0)
        
        return force
    
    async def _calculate_cohesion_force(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate cohesion force to move toward neighbor center."""
        
        center_of_mass = (0.0, 0.0, 0.0)
        neighbor_count = 0
        cohesion_radius = 40.0
        
        for neighbor_id in bio_drone.social_connections:
            if neighbor_id in self.bio_drones:
                neighbor = self.bio_drones[neighbor_id]
                distance = self._calculate_distance(bio_drone.position, neighbor.position)
                
                if distance < cohesion_radius:
                    center_of_mass = (
                        center_of_mass[0] + neighbor.position[0],
                        center_of_mass[1] + neighbor.position[1],
                        center_of_mass[2] + neighbor.position[2]
                    )
                    neighbor_count += 1
        
        if neighbor_count > 0:
            center_of_mass = (
                center_of_mass[0] / neighbor_count,
                center_of_mass[1] / neighbor_count,
                center_of_mass[2] / neighbor_count
            )
            
            # Calculate force toward center
            force = (
                center_of_mass[0] - bio_drone.position[0],
                center_of_mass[1] - bio_drone.position[1],
                center_of_mass[2] - bio_drone.position[2]
            )
        else:
            force = (0.0, 0.0, 0.0)
        
        return force
    
    async def _calculate_obstacle_avoidance_force(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate obstacle avoidance force."""
        
        # Simplified obstacle avoidance
        # In real implementation, this would use sensor data
        force = (0.0, 0.0, 0.0)
        
        # Avoid ground (altitude constraint)
        if bio_drone.position[2] < 10.0:
            force = (force[0], force[1], force[2] + 2.0)
        
        # Avoid ceiling
        if bio_drone.position[2] > 100.0:
            force = (force[0], force[1], force[2] - 2.0)
        
        return force
    
    async def _detect_emergent_patterns(self):
        """Detect emergent patterns in swarm behavior."""
        
        current_time = time.time()
        
        # Detect flocking patterns
        flocking_groups = await self._detect_flocking_groups()
        
        for group in flocking_groups:
            if len(group) >= 5:  # Minimum group size for pattern
                centroid = self._calculate_group_centroid(group)
                formation_quality = self._assess_formation_quality(group)
                
                pattern = EmergentPattern(
                    pattern_type="flocking_formation",
                    confidence=formation_quality,
                    participants=group,
                    centroid=centroid,
                    formation_quality=formation_quality,
                    stability_score=0.8,  # Would be calculated from historical data
                    detected_at=current_time
                )
                
                self.emergent_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.emergent_patterns) > 50:
            self.emergent_patterns = self.emergent_patterns[-50:]
    
    async def _detect_flocking_groups(self) -> List[List[str]]:
        """Detect groups of drones exhibiting flocking behavior."""
        
        groups = []
        processed_drones = set()
        
        for drone_id, bio_drone in self.bio_drones.items():
            if drone_id in processed_drones:
                continue
                
            group = [drone_id]
            to_process = [drone_id]
            processed_drones.add(drone_id)
            
            while to_process:
                current_id = to_process.pop(0)
                current_drone = self.bio_drones[current_id]
                
                for neighbor_id in current_drone.social_connections:
                    if neighbor_id not in processed_drones and neighbor_id in self.bio_drones:
                        neighbor = self.bio_drones[neighbor_id]
                        
                        # Check if neighbor exhibits similar velocity (flocking)
                        velocity_similarity = self._calculate_velocity_similarity(
                            current_drone.velocity, neighbor.velocity
                        )
                        
                        if velocity_similarity > 0.7:  # Threshold for flocking
                            group.append(neighbor_id)
                            to_process.append(neighbor_id)
                            processed_drones.add(neighbor_id)
            
            if len(group) >= 3:  # Minimum flock size
                groups.append(group)
        
        return groups
    
    def _calculate_group_centroid(self, group: List[str]) -> Tuple[float, float, float]:
        """Calculate centroid of drone group."""
        
        if not group:
            return (0.0, 0.0, 0.0)
        
        total_x = sum(self.bio_drones[drone_id].position[0] for drone_id in group if drone_id in self.bio_drones)
        total_y = sum(self.bio_drones[drone_id].position[1] for drone_id in group if drone_id in self.bio_drones)
        total_z = sum(self.bio_drones[drone_id].position[2] for drone_id in group if drone_id in self.bio_drones)
        
        group_size = len([drone_id for drone_id in group if drone_id in self.bio_drones])
        
        if group_size > 0:
            return (total_x / group_size, total_y / group_size, total_z / group_size)
        else:
            return (0.0, 0.0, 0.0)
    
    def _assess_formation_quality(self, group: List[str]) -> float:
        """Assess quality of formation for drone group."""
        
        if len(group) < 3:
            return 0.0
        
        # Calculate spacing uniformity
        distances = []
        positions = [self.bio_drones[drone_id].position for drone_id in group if drone_id in self.bio_drones]
        
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distance = self._calculate_distance(pos1, pos2)
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Calculate coefficient of variation (lower = more uniform)
        mean_distance = sum(distances) / len(distances)
        variance = sum((d - mean_distance) ** 2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)
        
        if mean_distance > 0:
            cv = std_dev / mean_distance
            uniformity = max(0.0, 1.0 - cv)  # Higher uniformity = better formation
        else:
            uniformity = 0.0
        
        return min(1.0, uniformity)
    
    def _calculate_velocity_similarity(self, vel1: Tuple[float, float, float], vel2: Tuple[float, float, float]) -> float:
        """Calculate similarity between two velocity vectors."""
        
        # Calculate dot product and magnitudes
        dot_product = vel1[0] * vel2[0] + vel1[1] * vel2[1] + vel1[2] * vel2[2]
        
        mag1 = math.sqrt(vel1[0]**2 + vel1[1]**2 + vel1[2]**2)
        mag2 = math.sqrt(vel2[0]**2 + vel2[1]**2 + vel2[2]**2)
        
        if mag1 > 0 and mag2 > 0:
            cosine_similarity = dot_product / (mag1 * mag2)
            return max(0.0, cosine_similarity)  # Clamp to [0, 1]
        else:
            return 0.0
    
    async def _ant_colony_path_search(self, 
                                    start: Tuple[float, float, float],
                                    target: Tuple[float, float, float],
                                    obstacles: List[Tuple[float, float, float]]) -> List[List[Tuple[float, float, float]]]:
        """Execute ant colony search for optimal path."""
        
        paths = []
        num_ants = 20
        
        for ant in range(num_ants):
            path = await self._ant_path_construction(start, target, obstacles)
            paths.append(path)
        
        return paths
    
    async def _ant_path_construction(self,
                                   start: Tuple[float, float, float],
                                   target: Tuple[float, float, float],
                                   obstacles: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Construct path for single ant."""
        
        path = [start]
        current_pos = start
        max_steps = 50
        step_size = 5.0
        
        for step in range(max_steps):
            # Calculate direction to target
            target_direction = (
                target[0] - current_pos[0],
                target[1] - current_pos[1],
                target[2] - current_pos[2]
            )
            
            # Normalize direction
            magnitude = math.sqrt(sum(d**2 for d in target_direction))
            if magnitude > 0:
                target_direction = tuple(d / magnitude for d in target_direction)
            
            # Add pheromone influence
            pheromone_direction = self._get_pheromone_gradient(current_pos)
            
            # Combine directions with random exploration
            exploration_factor = 0.3
            combined_direction = (
                target_direction[0] * 0.7 + pheromone_direction[0] * 0.2 + random.uniform(-exploration_factor, exploration_factor),
                target_direction[1] * 0.7 + pheromone_direction[1] * 0.2 + random.uniform(-exploration_factor, exploration_factor),
                target_direction[2] * 0.7 + pheromone_direction[2] * 0.2 + random.uniform(-exploration_factor, exploration_factor)
            )
            
            # Normalize combined direction
            magnitude = math.sqrt(sum(d**2 for d in combined_direction))
            if magnitude > 0:
                combined_direction = tuple(d / magnitude for d in combined_direction)
            
            # Take step
            next_pos = (
                current_pos[0] + combined_direction[0] * step_size,
                current_pos[1] + combined_direction[1] * step_size,
                current_pos[2] + combined_direction[2] * step_size
            )
            
            # Check for obstacles
            if not self._path_intersects_obstacles(current_pos, next_pos, obstacles):
                path.append(next_pos)
                current_pos = next_pos
                
                # Check if reached target
                distance_to_target = self._calculate_distance(current_pos, target)
                if distance_to_target < step_size:
                    path.append(target)
                    break
            else:
                # Try alternative direction
                alternative_direction = self._find_alternative_direction(current_pos, obstacles)
                next_pos = (
                    current_pos[0] + alternative_direction[0] * step_size,
                    current_pos[1] + alternative_direction[1] * step_size,
                    current_pos[2] + alternative_direction[2] * step_size
                )
                path.append(next_pos)
                current_pos = next_pos
        
        return path
    
    def _select_best_ant_path(self, ant_paths: List[List[Tuple[float, float, float]]]) -> List[Tuple[float, float, float]]:
        """Select best path from ant colony results."""
        
        if not ant_paths:
            return []
        
        best_path = None
        best_score = float('inf')
        
        for path in ant_paths:
            # Calculate path quality (shorter + less obstacles = better)
            path_length = self._calculate_path_length(path)
            smoothness = self._calculate_path_smoothness(path)
            
            score = path_length + (1.0 - smoothness) * 50.0  # Penalty for rough paths
            
            if score < best_score:
                best_score = score
                best_path = path
        
        return best_path or []
    
    async def _update_pheromone_trail(self, path: List[Tuple[float, float, float]], strength: float):
        """Update pheromone trail along path."""
        
        for point in path:
            # Discretize position for pheromone map
            discrete_pos = (
                int(point[0] / 5.0),  # 5-meter grid
                int(point[1] / 5.0),
                int(point[2] / 5.0)
            )
            
            self.pheromone_map[discrete_pos] += strength
    
    def _get_pheromone_gradient(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Get pheromone gradient at position."""
        
        discrete_pos = (
            int(position[0] / 5.0),
            int(position[1] / 5.0),
            int(position[2] / 5.0)
        )
        
        # Calculate gradient by checking neighboring cells
        gradient = (0.0, 0.0, 0.0)
        current_pheromone = self.pheromone_map.get(discrete_pos, 0.0)
        
        # Check 6 neighboring cells
        neighbors = [
            (discrete_pos[0] + 1, discrete_pos[1], discrete_pos[2]),
            (discrete_pos[0] - 1, discrete_pos[1], discrete_pos[2]),
            (discrete_pos[0], discrete_pos[1] + 1, discrete_pos[2]),
            (discrete_pos[0], discrete_pos[1] - 1, discrete_pos[2]),
            (discrete_pos[0], discrete_pos[1], discrete_pos[2] + 1),
            (discrete_pos[0], discrete_pos[1], discrete_pos[2] - 1)
        ]
        
        directions = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]
        
        for neighbor, direction in zip(neighbors, directions):
            neighbor_pheromone = self.pheromone_map.get(neighbor, 0.0)
            pheromone_diff = neighbor_pheromone - current_pheromone
            
            gradient = (
                gradient[0] + direction[0] * pheromone_diff,
                gradient[1] + direction[1] * pheromone_diff,
                gradient[2] + direction[2] * pheromone_diff
            )
        
        return gradient
    
    async def _scout_resource_sites(self, resource_locations: List[Tuple[float, float, float]]) -> Dict[int, float]:
        """Scout resource sites and evaluate quality."""
        
        site_qualities = {}
        
        for i, location in enumerate(resource_locations):
            # Evaluate site quality based on various factors
            accessibility = self._evaluate_site_accessibility(location)
            resource_density = random.uniform(0.3, 1.0)  # Simulated resource density
            safety_score = self._evaluate_site_safety(location)
            
            # Combine factors for overall quality
            quality = (accessibility * 0.4 + resource_density * 0.4 + safety_score * 0.2)
            site_qualities[i] = quality
        
        return site_qualities
    
    async def _bee_site_selection(self,
                                bio_drone: BioDrone,
                                resource_locations: List[Tuple[float, float, float]],
                                site_qualities: Dict[int, float],
                                capacity: float) -> int:
        """Select best resource site for bee drone."""
        
        best_site = 0
        best_score = 0.0
        
        for site_idx, quality in site_qualities.items():
            if site_idx < len(resource_locations):
                location = resource_locations[site_idx]
                distance = self._calculate_distance(bio_drone.position, location)
                
                # Score based on quality, distance, and capacity match
                distance_penalty = distance / 100.0  # Normalize distance
                capacity_bonus = min(1.0, capacity / 10.0)  # Favor higher capacity drones for better sites
                
                score = quality * capacity_bonus - distance_penalty * 0.3
                
                if score > best_score:
                    best_score = score
                    best_site = site_idx
        
        return best_site
    
    async def _perform_waggle_dance(self, dancer_id: str, site_index: int, quality: float):
        """Perform waggle dance to communicate site information."""
        
        if dancer_id not in self.bio_drones:
            return
        
        dancer = self.bio_drones[dancer_id]
        
        # Communicate to nearby drones
        for neighbor_id in dancer.social_connections:
            if neighbor_id in self.bio_drones:
                neighbor = self.bio_drones[neighbor_id]
                
                # Probability of following dance based on quality
                follow_probability = quality * 0.8
                
                if random.random() < follow_probability:
                    # Neighbor influenced by waggle dance
                    neighbor.fitness_score += quality * 0.1
    
    async def _calculate_predator_avoidance(self,
                                          bio_drone: BioDrone,
                                          predator_locations: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Calculate force to avoid predators."""
        
        avoidance_force = (0.0, 0.0, 0.0)
        danger_radius = 30.0
        
        for predator_pos in predator_locations:
            distance = self._calculate_distance(bio_drone.position, predator_pos)
            
            if distance < danger_radius:
                # Calculate escape direction
                escape_direction = (
                    bio_drone.position[0] - predator_pos[0],
                    bio_drone.position[1] - predator_pos[1],
                    bio_drone.position[2] - predator_pos[2]
                )
                
                # Normalize and scale by inverse distance
                magnitude = max(distance, 0.1)
                normalized_escape = tuple(d / magnitude for d in escape_direction)
                
                # Stronger force for closer predators
                force_strength = (danger_radius - distance) / danger_radius
                
                avoidance_force = (
                    avoidance_force[0] + normalized_escape[0] * force_strength,
                    avoidance_force[1] + normalized_escape[1] * force_strength,
                    avoidance_force[2] + normalized_escape[2] * force_strength
                )
        
        return avoidance_force
    
    async def _calculate_neighbor_alignment(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate schooling neighbor alignment force."""
        return await self._calculate_alignment_force(bio_drone)
    
    async def _calculate_group_cohesion(self, bio_drone: BioDrone) -> Tuple[float, float, float]:
        """Calculate schooling group cohesion force."""
        return await self._calculate_cohesion_force(bio_drone)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def _limit_force(self, force: Tuple[float, float, float], max_force: float) -> Tuple[float, float, float]:
        """Limit force magnitude to maximum value."""
        magnitude = math.sqrt(sum(f**2 for f in force))
        
        if magnitude > max_force:
            scale = max_force / magnitude
            return tuple(f * scale for f in force)
        
        return force
    
    def _limit_velocity(self, velocity: Tuple[float, float, float], max_speed: float) -> Tuple[float, float, float]:
        """Limit velocity magnitude to maximum speed."""
        magnitude = math.sqrt(sum(v**2 for v in velocity))
        
        if magnitude > max_speed:
            scale = max_speed / magnitude
            return tuple(v * scale for v in velocity)
        
        return velocity
    
    def _path_intersects_obstacles(self,
                                 start: Tuple[float, float, float],
                                 end: Tuple[float, float, float],
                                 obstacles: List[Tuple[float, float, float]]) -> bool:
        """Check if path segment intersects any obstacles."""
        
        obstacle_radius = 5.0  # Obstacle avoidance radius
        
        for obstacle in obstacles:
            # Check distance from line segment to obstacle
            distance = self._point_to_line_distance(obstacle, start, end)
            
            if distance < obstacle_radius:
                return True
        
        return False
    
    def _point_to_line_distance(self,
                              point: Tuple[float, float, float],
                              line_start: Tuple[float, float, float],
                              line_end: Tuple[float, float, float]) -> float:
        """Calculate distance from point to line segment."""
        
        # Vector from line_start to line_end
        line_vec = tuple(e - s for s, e in zip(line_start, line_end))
        
        # Vector from line_start to point
        point_vec = tuple(p - s for p, s in zip(point, line_start))
        
        # Project point onto line
        line_length_sq = sum(v**2 for v in line_vec)
        
        if line_length_sq == 0:
            return self._calculate_distance(point, line_start)
        
        projection = sum(p * l for p, l in zip(point_vec, line_vec)) / line_length_sq
        projection = max(0, min(1, projection))  # Clamp to line segment
        
        # Find closest point on line segment
        closest_point = tuple(s + projection * l for s, l in zip(line_start, line_vec))
        
        return self._calculate_distance(point, closest_point)
    
    def _find_alternative_direction(self,
                                  position: Tuple[float, float, float],
                                  obstacles: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Find alternative direction to avoid obstacles."""
        
        # Try several random directions and pick the one with least obstacle interference
        best_direction = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        best_score = 0.0
        
        for _ in range(10):
            direction = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            magnitude = math.sqrt(sum(d**2 for d in direction))
            
            if magnitude > 0:
                direction = tuple(d / magnitude for d in direction)
                
                # Score direction based on obstacle avoidance
                score = self._score_direction(position, direction, obstacles)
                
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        return best_direction
    
    def _score_direction(self,
                       position: Tuple[float, float, float],
                       direction: Tuple[float, float, float],
                       obstacles: List[Tuple[float, float, float]]) -> float:
        """Score a direction based on obstacle avoidance."""
        
        test_distance = 10.0
        test_point = (
            position[0] + direction[0] * test_distance,
            position[1] + direction[1] * test_distance,
            position[2] + direction[2] * test_distance
        )
        
        # Calculate minimum distance to obstacles
        min_distance = float('inf')
        
        for obstacle in obstacles:
            distance = self._calculate_distance(test_point, obstacle)
            min_distance = min(min_distance, distance)
        
        # Higher score for directions that avoid obstacles
        return min_distance
    
    def _calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate total length of path."""
        
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        
        for i in range(len(path) - 1):
            segment_length = self._calculate_distance(path[i], path[i + 1])
            total_length += segment_length
        
        return total_length
    
    def _calculate_path_smoothness(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate smoothness of path (0 = rough, 1 = smooth)."""
        
        if len(path) < 3:
            return 1.0
        
        total_angle_change = 0.0
        
        for i in range(len(path) - 2):
            # Calculate angle between consecutive segments
            vec1 = tuple(path[i+1][j] - path[i][j] for j in range(3))
            vec2 = tuple(path[i+2][j] - path[i+1][j] for j in range(3))
            
            # Calculate angle between vectors
            dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
            
            mag1 = math.sqrt(sum(v**2 for v in vec1))
            mag2 = math.sqrt(sum(v**2 for v in vec2))
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)
                total_angle_change += angle
        
        # Normalize by path length and convert to smoothness score
        if len(path) > 2:
            avg_angle_change = total_angle_change / (len(path) - 2)
            smoothness = max(0.0, 1.0 - avg_angle_change / math.pi)
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _evaluate_site_accessibility(self, location: Tuple[float, float, float]) -> float:
        """Evaluate accessibility of resource site."""
        
        # Simple accessibility based on altitude and position
        altitude_score = 1.0 - abs(location[2] - 50.0) / 100.0  # Prefer ~50m altitude
        
        # Prefer locations not too far from origin
        distance_from_origin = math.sqrt(location[0]**2 + location[1]**2)
        distance_score = max(0.0, 1.0 - distance_from_origin / 500.0)
        
        return max(0.0, (altitude_score + distance_score) / 2.0)
    
    def _evaluate_site_safety(self, location: Tuple[float, float, float]) -> float:
        """Evaluate safety of resource site."""
        
        # Simple safety evaluation (avoid extremes)
        safety_score = 1.0
        
        # Penalize very high or very low altitudes
        if location[2] < 5.0 or location[2] > 150.0:
            safety_score *= 0.5
        
        # Penalize positions near "danger zones" (simplified)
        danger_zones = [(0, 0, 0), (100, 100, 50)]  # Example danger zones
        
        for danger_zone in danger_zones:
            distance = self._calculate_distance(location, danger_zone)
            if distance < 30.0:
                safety_score *= (distance / 30.0)
        
        return max(0.0, safety_score)
    
    def get_bio_swarm_statistics(self) -> Dict[str, Any]:
        """Get bio-inspired swarm intelligence statistics."""
        
        # Calculate collective intelligence metrics
        avg_fitness = sum(drone.fitness_score for drone in self.bio_drones.values()) / max(1, len(self.bio_drones))
        
        # Calculate swarm cohesion
        if len(self.bio_drones) > 1:
            positions = [drone.position for drone in self.bio_drones.values()]
            centroid = self._calculate_group_centroid(list(self.bio_drones.keys()))
            
            distances = [self._calculate_distance(pos, centroid) for pos in positions]
            avg_distance = sum(distances) / len(distances)
            cohesion = max(0.0, 1.0 - avg_distance / 100.0)  # Normalize to 100m
        else:
            cohesion = 1.0
        
        # Update behavior stats
        self.behavior_stats.update({
            'collective_intelligence_score': avg_fitness,
            'swarm_cohesion': cohesion,
            'adaptation_rate': 0.8,  # Would be calculated from behavioral changes
            'emergence_frequency': len(self.emergent_patterns) / max(1, time.time())
        })
        
        return {
            'active_bio_drones': len(self.bio_drones),
            'social_connections': sum(len(connections) for connections in self.social_network.values()),
            'emergent_patterns_detected': len(self.emergent_patterns),
            'pheromone_map_size': len(self.pheromone_map),
            'behavior_statistics': self.behavior_stats.copy(),
            'flocking_weights': self.flocking_weights.copy()
        }