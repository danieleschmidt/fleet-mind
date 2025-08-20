"""Hyperscale Coordination Engine for Ultra-Large Drone Swarms.

This module provides advanced scalability capabilities:
- Hierarchical coordination for 10,000+ drones
- Distributed computing and edge processing
- Advanced load balancing and resource optimization
- Dynamic scaling and auto-provisioning
- Performance monitoring and optimization
"""

import asyncio
import time
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for swarm coordination."""
    HIERARCHICAL = "hierarchical"
    DISTRIBUTED = "distributed"
    FEDERATED = "federated"
    MESH_NETWORK = "mesh_network"
    HYBRID_ADAPTIVE = "hybrid_adaptive"


class LoadBalancingMethod(Enum):
    """Load balancing methods."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_FAIR = "weighted_fair"
    LEAST_LOADED = "least_loaded"
    GEOGRAPHIC = "geographic"
    CAPABILITY_BASED = "capability_based"
    AI_OPTIMIZED = "ai_optimized"


class PerformanceMetric(Enum):
    """Performance metrics for optimization."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_UTILIZATION = "resource_utilization"
    NETWORK_EFFICIENCY = "network_efficiency"
    ENERGY_CONSUMPTION = "energy_consumption"
    COORDINATION_ACCURACY = "coordination_accuracy"


@dataclass
class CoordinationNode:
    """Represents a coordination node in the hierarchy."""
    node_id: str
    node_type: str  # "primary", "regional", "local", "drone"
    level: int  # Hierarchy level (0=primary, 1=regional, etc.)
    
    # Capacity and performance
    max_children: int = 50
    current_children: int = 0
    processing_capacity: float = 1.0
    current_load: float = 0.0
    
    # Network topology
    parent_node: Optional[str] = None
    child_nodes: Set[str] = field(default_factory=set)
    peer_nodes: Set[str] = field(default_factory=set)
    
    # Geographic information
    position: Optional[Tuple[float, float, float]] = None
    coverage_radius: float = 1000.0  # meters
    
    # Performance metrics
    latency_to_parent: float = 0.0
    bandwidth_to_parent: float = 1000.0  # Mbps
    last_heartbeat: float = field(default_factory=time.time)
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    energy_consumption: float = 0.0


@dataclass
class ScalingPolicy:
    """Defines scaling policies and thresholds."""
    policy_id: str
    scaling_strategy: ScalingStrategy
    
    # Scaling thresholds
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Scaling parameters
    min_nodes: int = 1
    max_nodes: int = 10000
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    
    # Performance targets
    target_latency: float = 100.0  # ms
    target_throughput: float = 1000.0  # ops/sec
    target_resource_utilization: float = 0.7
    
    # Constraints
    geographic_constraints: Dict[str, Any] = field(default_factory=dict)
    resource_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceOptimization:
    """Performance optimization configuration."""
    optimization_id: str
    target_metrics: List[PerformanceMetric]
    
    # Optimization algorithms
    load_balancing_method: LoadBalancingMethod = LoadBalancingMethod.AI_OPTIMIZED
    caching_strategy: str = "adaptive"
    prefetching_enabled: bool = True
    
    # Performance parameters
    optimization_interval: float = 10.0  # seconds
    learning_rate: float = 0.1
    exploration_factor: float = 0.2
    
    # Resource management
    dynamic_resource_allocation: bool = True
    auto_scaling_enabled: bool = True
    predictive_scaling: bool = True


class HyperscaleCoordinator:
    """Ultra-large scale coordination engine."""
    
    def __init__(self):
        # Coordination topology
        self.coordination_nodes: Dict[str, CoordinationNode] = {}
        self.hierarchy_levels: Dict[int, List[str]] = defaultdict(list)
        
        # Scaling management
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.performance_optimizations: Dict[str, PerformanceOptimization] = {}
        
        # Load balancing
        self.load_balancers: Dict[str, Callable] = {}
        self.routing_tables: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance monitoring
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_targets: Dict[str, float] = {}
        
        # Optimization state
        self.optimization_history: deque = deque(maxlen=10000)
        self.ml_models: Dict[str, Any] = {}
        self.prediction_cache: Dict[str, Any] = {}
        
        # Scalability metrics
        self.scalability_metrics: Dict[str, Any] = {
            "total_nodes": 0,
            "max_hierarchy_depth": 0,
            "coordination_efficiency": 1.0,
            "average_latency": 0.0,
            "throughput": 0.0,
            "resource_utilization": 0.0,
            "scaling_events": 0,
            "optimization_iterations": 0
        }
        
        # Initialize default configurations
        self._initialize_default_policies()
        self._initialize_load_balancers()
        
        # Start optimization loops
        self._start_optimization_loops()
        
        logger.info("Hyperscale coordinator initialized")
    
    def _initialize_default_policies(self) -> None:
        """Initialize default scaling policies."""
        
        # Hierarchical scaling policy
        hierarchical_policy = ScalingPolicy(
            policy_id="default_hierarchical",
            scaling_strategy=ScalingStrategy.HIERARCHICAL,
            scale_up_threshold=0.75,
            scale_down_threshold=0.25,
            min_nodes=1,
            max_nodes=10000,
            target_latency=50.0,
            target_throughput=5000.0
        )
        
        self.scaling_policies["default"] = hierarchical_policy
        
        # Performance optimization
        performance_opt = PerformanceOptimization(
            optimization_id="default_optimization",
            target_metrics=[
                PerformanceMetric.LATENCY,
                PerformanceMetric.THROUGHPUT,
                PerformanceMetric.RESOURCE_UTILIZATION
            ],
            load_balancing_method=LoadBalancingMethod.AI_OPTIMIZED,
            optimization_interval=5.0
        )
        
        self.performance_optimizations["default"] = performance_opt
    
    def _initialize_load_balancers(self) -> None:
        """Initialize load balancing algorithms."""
        
        self.load_balancers = {
            LoadBalancingMethod.ROUND_ROBIN: self._round_robin_balancer,
            LoadBalancingMethod.WEIGHTED_FAIR: self._weighted_fair_balancer,
            LoadBalancingMethod.LEAST_LOADED: self._least_loaded_balancer,
            LoadBalancingMethod.GEOGRAPHIC: self._geographic_balancer,
            LoadBalancingMethod.CAPABILITY_BASED: self._capability_based_balancer,
            LoadBalancingMethod.AI_OPTIMIZED: self._ai_optimized_balancer
        }
    
    def _start_optimization_loops(self) -> None:
        """Start background optimization loops."""
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._scaling_optimization_loop())
        asyncio.create_task(self._load_balancing_loop())
        asyncio.create_task(self._predictive_scaling_loop())
    
    async def create_coordination_hierarchy(self, 
                                          max_drones: int,
                                          hierarchy_config: Dict[str, Any]) -> str:
        """Create hierarchical coordination structure."""
        
        hierarchy_id = f"hierarchy_{int(time.time())}"
        
        # Calculate optimal hierarchy structure
        structure = self._calculate_optimal_hierarchy(max_drones, hierarchy_config)
        
        # Create primary coordinator
        primary_node = await self._create_coordination_node(
            node_id=f"{hierarchy_id}_primary",
            node_type="primary",
            level=0,
            max_children=structure["primary_children"],
            position=hierarchy_config.get("primary_position", (0.0, 0.0, 100.0))
        )
        
        # Create regional coordinators
        regional_nodes = []
        for i in range(structure["regional_count"]):
            regional_node = await self._create_coordination_node(
                node_id=f"{hierarchy_id}_regional_{i}",
                node_type="regional",
                level=1,
                max_children=structure["regional_children"],
                parent_node=primary_node,
                position=self._calculate_regional_position(i, structure["regional_count"])
            )
            regional_nodes.append(regional_node)
        
        # Create local coordinators
        local_nodes = []
        for regional_node in regional_nodes:
            for i in range(structure["local_per_regional"]):
                local_node = await self._create_coordination_node(
                    node_id=f"{regional_node}_local_{i}",
                    node_type="local",
                    level=2,
                    max_children=structure["local_children"],
                    parent_node=regional_node,
                    position=self._calculate_local_position(regional_node, i)
                )
                local_nodes.append(local_node)
        
        # Update hierarchy tracking
        self.hierarchy_levels[0].append(primary_node)
        self.hierarchy_levels[1].extend(regional_nodes)
        self.hierarchy_levels[2].extend(local_nodes)
        
        self.scalability_metrics["total_nodes"] = len(self.coordination_nodes)
        self.scalability_metrics["max_hierarchy_depth"] = max(self.hierarchy_levels.keys())
        
        logger.info(f"Created coordination hierarchy: {hierarchy_id}")
        return hierarchy_id
    
    def _calculate_optimal_hierarchy(self, 
                                   max_drones: int,
                                   config: Dict[str, Any]) -> Dict[str, int]:
        """Calculate optimal hierarchy structure for given drone count."""
        
        # Use logarithmic scaling for hierarchy depth
        optimal_fanout = config.get("optimal_fanout", 10)
        max_local_drones = config.get("max_local_drones", 50)
        
        # Calculate hierarchy levels
        if max_drones <= max_local_drones:
            # Single level hierarchy
            return {
                "primary_children": 1,
                "regional_count": 1,
                "regional_children": 1,
                "local_per_regional": 1,
                "local_children": max_drones
            }
        
        # Multi-level hierarchy
        total_local_coordinators = math.ceil(max_drones / max_local_drones)
        regional_count = max(1, math.ceil(total_local_coordinators / optimal_fanout))
        local_per_regional = math.ceil(total_local_coordinators / regional_count)
        
        return {
            "primary_children": regional_count,
            "regional_count": regional_count,
            "regional_children": local_per_regional,
            "local_per_regional": local_per_regional,
            "local_children": max_local_drones
        }
    
    async def _create_coordination_node(self,
                                      node_id: str,
                                      node_type: str,
                                      level: int,
                                      max_children: int,
                                      parent_node: Optional[str] = None,
                                      position: Optional[Tuple[float, float, float]] = None) -> str:
        """Create a coordination node."""
        
        # Calculate processing capacity based on node type
        capacity_map = {
            "primary": 10.0,
            "regional": 5.0,
            "local": 2.0,
            "drone": 1.0
        }
        
        processing_capacity = capacity_map.get(node_type, 1.0)
        
        # Calculate coverage radius based on level
        coverage_radius = 10000.0 / (level + 1)  # Larger radius for higher levels
        
        node = CoordinationNode(
            node_id=node_id,
            node_type=node_type,
            level=level,
            max_children=max_children,
            processing_capacity=processing_capacity,
            parent_node=parent_node,
            position=position,
            coverage_radius=coverage_radius
        )
        
        # Add to parent's children if specified
        if parent_node and parent_node in self.coordination_nodes:
            parent = self.coordination_nodes[parent_node]
            parent.child_nodes.add(node_id)
            parent.current_children += 1
            
            # Calculate latency to parent based on distance
            if position and parent.position:
                distance = self._calculate_distance(position, parent.position)
                node.latency_to_parent = self._estimate_latency(distance)
        
        self.coordination_nodes[node_id] = node
        return node_id
    
    def _calculate_regional_position(self, index: int, total_count: int) -> Tuple[float, float, float]:
        """Calculate position for regional coordinator."""
        angle = (2 * math.pi * index) / total_count
        radius = 50000.0  # 50km radius
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 200.0  # 200m altitude
        
        return (x, y, z)
    
    def _calculate_local_position(self, regional_node_id: str, index: int) -> Tuple[float, float, float]:
        """Calculate position for local coordinator."""
        regional_node = self.coordination_nodes.get(regional_node_id)
        if not regional_node or not regional_node.position:
            return (0.0, 0.0, 50.0)
        
        # Position local coordinators around regional coordinator
        angle = (2 * math.pi * index) / 8  # Assume max 8 local per regional
        radius = 10000.0  # 10km radius
        
        regional_x, regional_y, regional_z = regional_node.position
        
        x = regional_x + radius * math.cos(angle)
        y = regional_y + radius * math.sin(angle)
        z = 100.0  # 100m altitude
        
        return (x, y, z)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between positions."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _estimate_latency(self, distance: float) -> float:
        """Estimate communication latency based on distance."""
        # Base latency + propagation delay
        base_latency = 5.0  # ms
        speed_of_light = 299792458.0  # m/s
        propagation_delay = (distance / speed_of_light) * 1000  # Convert to ms
        
        # Add processing and queuing delays
        processing_delay = random.uniform(1.0, 5.0)
        
        return base_latency + propagation_delay + processing_delay
    
    async def register_drone(self, 
                           drone_id: str,
                           position: Tuple[float, float, float],
                           capabilities: List[str]) -> str:
        """Register drone with optimal coordinator."""
        
        # Find best coordinator for this drone
        best_coordinator = await self._find_optimal_coordinator(position, capabilities)
        
        if not best_coordinator:
            raise ValueError("No suitable coordinator found")
        
        # Create drone node
        drone_node = await self._create_coordination_node(
            node_id=drone_id,
            node_type="drone",
            level=3,  # Drone level
            max_children=0,
            parent_node=best_coordinator,
            position=position
        )
        
        # Update load balancing
        await self._update_load_distribution(best_coordinator)
        
        logger.info(f"Registered drone {drone_id} with coordinator {best_coordinator}")
        return best_coordinator
    
    async def _find_optimal_coordinator(self, 
                                      position: Tuple[float, float, float],
                                      capabilities: List[str]) -> Optional[str]:
        """Find optimal coordinator for drone based on position and capabilities."""
        
        # Find local coordinators within range
        candidate_coordinators = []
        
        for node_id, node in self.coordination_nodes.items():
            if (node.node_type == "local" and 
                node.current_children < node.max_children and
                node.position):
                
                distance = self._calculate_distance(position, node.position)
                if distance <= node.coverage_radius:
                    # Calculate score based on distance, load, and capability match
                    load_factor = 1.0 - (node.current_load / node.processing_capacity)
                    distance_factor = 1.0 - (distance / node.coverage_radius)
                    
                    score = (load_factor * 0.6) + (distance_factor * 0.4)
                    candidate_coordinators.append((score, node_id))
        
        if candidate_coordinators:
            # Return coordinator with highest score
            candidate_coordinators.sort(reverse=True)
            return candidate_coordinators[0][1]
        
        return None
    
    async def _performance_monitoring_loop(self) -> None:
        """Continuous performance monitoring loop."""
        while True:
            try:
                await self._collect_performance_metrics()
                await self._analyze_performance_trends()
                await asyncio.sleep(1.0)  # Monitor every second
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics from all nodes."""
        current_time = time.time()
        
        total_latency = 0.0
        total_throughput = 0.0
        total_resource_utilization = 0.0
        active_nodes = 0
        
        for node_id, node in self.coordination_nodes.items():
            # Simulate metric collection
            node.cpu_usage = random.uniform(0.1, 0.9)
            node.memory_usage = random.uniform(0.2, 0.8)
            node.network_usage = random.uniform(0.1, 0.7)
            
            # Calculate derived metrics
            node.current_load = (node.cpu_usage + node.memory_usage + node.network_usage) / 3.0
            
            # Aggregate metrics
            total_latency += node.latency_to_parent
            total_throughput += node.processing_capacity * (1.0 - node.current_load)
            total_resource_utilization += node.current_load
            active_nodes += 1
            
            # Store historical data
            self.performance_metrics[f"{node_id}_latency"].append(node.latency_to_parent)
            self.performance_metrics[f"{node_id}_load"].append(node.current_load)
            self.performance_metrics[f"{node_id}_throughput"].append(node.processing_capacity)
        
        # Update global metrics
        if active_nodes > 0:
            self.scalability_metrics["average_latency"] = total_latency / active_nodes
            self.scalability_metrics["throughput"] = total_throughput
            self.scalability_metrics["resource_utilization"] = total_resource_utilization / active_nodes
        
        # Calculate coordination efficiency
        self.scalability_metrics["coordination_efficiency"] = self._calculate_coordination_efficiency()
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate overall coordination efficiency."""
        if not self.coordination_nodes:
            return 1.0
        
        # Factors affecting efficiency
        factors = []
        
        # Load distribution factor
        loads = [node.current_load for node in self.coordination_nodes.values()]
        if loads:
            load_variance = sum((load - sum(loads)/len(loads))**2 for load in loads) / len(loads)
            load_factor = max(0.0, 1.0 - load_variance)
            factors.append(load_factor)
        
        # Latency factor
        latencies = [node.latency_to_parent for node in self.coordination_nodes.values() if node.latency_to_parent > 0]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            target_latency = self.scaling_policies["default"].target_latency
            latency_factor = max(0.0, min(1.0, target_latency / max(avg_latency, 1.0)))
            factors.append(latency_factor)
        
        # Resource utilization factor (optimal around 0.7)
        avg_utilization = self.scalability_metrics["resource_utilization"]
        optimal_utilization = 0.7
        utilization_factor = 1.0 - abs(avg_utilization - optimal_utilization) / optimal_utilization
        factors.append(max(0.0, utilization_factor))
        
        return sum(factors) / len(factors) if factors else 1.0
    
    async def _scaling_optimization_loop(self) -> None:
        """Continuous scaling optimization loop."""
        while True:
            try:
                await self._evaluate_scaling_needs()
                await self._execute_scaling_decisions()
                await asyncio.sleep(10.0)  # Evaluate every 10 seconds
            except Exception as e:
                logger.error(f"Scaling optimization error: {e}")
                await asyncio.sleep(5.0)
    
    async def _evaluate_scaling_needs(self) -> None:
        """Evaluate if scaling is needed."""
        policy = self.scaling_policies["default"]
        
        # Check each hierarchy level for scaling needs
        for level, node_ids in self.hierarchy_levels.items():
            if level == 0:  # Skip primary level
                continue
            
            overloaded_nodes = []
            underloaded_nodes = []
            
            for node_id in node_ids:
                node = self.coordination_nodes.get(node_id)
                if not node:
                    continue
                
                # Check load thresholds
                if node.current_load > policy.scale_up_threshold:
                    overloaded_nodes.append(node_id)
                elif node.current_load < policy.scale_down_threshold:
                    underloaded_nodes.append(node_id)
            
            # Plan scaling actions
            if overloaded_nodes:
                await self._plan_scale_up(level, overloaded_nodes)
            
            if len(underloaded_nodes) > 1:  # Keep at least one node
                await self._plan_scale_down(level, underloaded_nodes[:-1])
    
    async def _plan_scale_up(self, level: int, overloaded_nodes: List[str]) -> None:
        """Plan scale-up actions for overloaded nodes."""
        for node_id in overloaded_nodes:
            node = self.coordination_nodes[node_id]
            
            # Create additional node at same level
            new_node_id = f"{node_id}_scale_{int(time.time())}"
            
            # Position new node near overloaded node
            if node.position:
                new_position = (
                    node.position[0] + random.uniform(-1000, 1000),
                    node.position[1] + random.uniform(-1000, 1000),
                    node.position[2]
                )
            else:
                new_position = None
            
            new_node = await self._create_coordination_node(
                node_id=new_node_id,
                node_type=node.node_type,
                level=level,
                max_children=node.max_children,
                parent_node=node.parent_node,
                position=new_position
            )
            
            # Add to hierarchy tracking
            self.hierarchy_levels[level].append(new_node_id)
            
            # Redistribute load
            await self._redistribute_load(node_id, new_node_id)
            
            self.scalability_metrics["scaling_events"] += 1
            logger.info(f"Scaled up: created {new_node_id} to help {node_id}")
    
    async def _plan_scale_down(self, level: int, underloaded_nodes: List[str]) -> None:
        """Plan scale-down actions for underloaded nodes."""
        for node_id in underloaded_nodes:
            # Redistribute children to other nodes
            await self._redistribute_children(node_id)
            
            # Remove node
            node = self.coordination_nodes.pop(node_id, None)
            if node and node.parent_node and node.parent_node in self.coordination_nodes:
                parent = self.coordination_nodes[node.parent_node]
                parent.child_nodes.discard(node_id)
                parent.current_children = max(0, parent.current_children - 1)
            
            # Remove from hierarchy tracking
            if level in self.hierarchy_levels:
                self.hierarchy_levels[level] = [
                    n for n in self.hierarchy_levels[level] if n != node_id
                ]
            
            self.scalability_metrics["scaling_events"] += 1
            logger.info(f"Scaled down: removed underloaded node {node_id}")
    
    async def _redistribute_load(self, source_node_id: str, target_node_id: str) -> None:
        """Redistribute load between nodes."""
        source_node = self.coordination_nodes.get(source_node_id)
        target_node = self.coordination_nodes.get(target_node_id)
        
        if not source_node or not target_node:
            return
        
        # Move half of the children from source to target
        children_to_move = list(source_node.child_nodes)[:len(source_node.child_nodes)//2]
        
        for child_id in children_to_move:
            if child_id in self.coordination_nodes:
                child_node = self.coordination_nodes[child_id]
                
                # Update parent references
                child_node.parent_node = target_node_id
                source_node.child_nodes.discard(child_id)
                target_node.child_nodes.add(child_id)
                
                # Update child counts
                source_node.current_children = max(0, source_node.current_children - 1)
                target_node.current_children += 1
    
    async def _redistribute_children(self, node_id: str) -> None:
        """Redistribute children of a node to siblings."""
        node = self.coordination_nodes.get(node_id)
        if not node or not node.parent_node:
            return
        
        parent = self.coordination_nodes.get(node.parent_node)
        if not parent:
            return
        
        # Find sibling nodes
        siblings = [
            sibling_id for sibling_id in parent.child_nodes
            if sibling_id != node_id and sibling_id in self.coordination_nodes
        ]
        
        if not siblings:
            return
        
        # Distribute children among siblings
        children = list(node.child_nodes)
        for i, child_id in enumerate(children):
            if child_id in self.coordination_nodes:
                target_sibling = siblings[i % len(siblings)]
                child_node = self.coordination_nodes[child_id]
                
                # Update parent reference
                child_node.parent_node = target_sibling
                
                # Update child sets
                target_sibling_node = self.coordination_nodes[target_sibling]
                target_sibling_node.child_nodes.add(child_id)
                target_sibling_node.current_children += 1
        
        # Clear node's children
        node.child_nodes.clear()
        node.current_children = 0
    
    async def _execute_scaling_decisions(self) -> None:
        """Execute planned scaling decisions."""
        # Update total node count
        self.scalability_metrics["total_nodes"] = len(self.coordination_nodes)
        
        # Update load balancing after scaling
        await self._rebalance_global_load()
    
    async def _load_balancing_loop(self) -> None:
        """Continuous load balancing loop."""
        while True:
            try:
                await self._perform_load_balancing()
                await asyncio.sleep(5.0)  # Balance every 5 seconds
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(2.0)
    
    async def _perform_load_balancing(self) -> None:
        """Perform intelligent load balancing."""
        optimization = self.performance_optimizations["default"]
        balancer = self.load_balancers[optimization.load_balancing_method]
        
        # Apply load balancing to each hierarchy level
        for level, node_ids in self.hierarchy_levels.items():
            if len(node_ids) > 1:
                await balancer(node_ids)
    
    async def _round_robin_balancer(self, node_ids: List[str]) -> None:
        """Round-robin load balancing."""
        # Simple round-robin redistribution
        total_load = sum(self.coordination_nodes[node_id].current_load 
                        for node_id in node_ids if node_id in self.coordination_nodes)
        
        if total_load == 0:
            return
        
        target_load_per_node = total_load / len(node_ids)
        
        for node_id in node_ids:
            if node_id in self.coordination_nodes:
                node = self.coordination_nodes[node_id]
                # Adjust load towards target (simplified)
                adjustment = (target_load_per_node - node.current_load) * 0.1
                node.current_load = max(0.0, min(1.0, node.current_load + adjustment))
    
    async def _weighted_fair_balancer(self, node_ids: List[str]) -> None:
        """Weighted fair queuing load balancing."""
        total_capacity = sum(self.coordination_nodes[node_id].processing_capacity 
                           for node_id in node_ids if node_id in self.coordination_nodes)
        
        if total_capacity == 0:
            return
        
        for node_id in node_ids:
            if node_id in self.coordination_nodes:
                node = self.coordination_nodes[node_id]
                # Weight based on processing capacity
                weight = node.processing_capacity / total_capacity
                target_load = weight * 0.7  # Target 70% utilization
                
                # Gradual adjustment
                adjustment = (target_load - node.current_load) * 0.2
                node.current_load = max(0.0, min(1.0, node.current_load + adjustment))
    
    async def _least_loaded_balancer(self, node_ids: List[str]) -> None:
        """Least loaded node balancing."""
        # Find least and most loaded nodes
        nodes_with_load = [
            (self.coordination_nodes[node_id].current_load, node_id)
            for node_id in node_ids if node_id in self.coordination_nodes
        ]
        
        if len(nodes_with_load) < 2:
            return
        
        nodes_with_load.sort()
        least_loaded = nodes_with_load[0]
        most_loaded = nodes_with_load[-1]
        
        # Transfer load from most to least loaded
        if most_loaded[0] - least_loaded[0] > 0.2:  # Significant difference
            transfer_amount = (most_loaded[0] - least_loaded[0]) * 0.1
            
            most_loaded_node = self.coordination_nodes[most_loaded[1]]
            least_loaded_node = self.coordination_nodes[least_loaded[1]]
            
            most_loaded_node.current_load -= transfer_amount
            least_loaded_node.current_load += transfer_amount
    
    async def _geographic_balancer(self, node_ids: List[str]) -> None:
        """Geographic proximity-based load balancing."""
        # Group nodes by geographic proximity
        geographic_groups = defaultdict(list)
        
        for node_id in node_ids:
            if node_id in self.coordination_nodes:
                node = self.coordination_nodes[node_id]
                if node.position:
                    # Simple geographic grouping by grid
                    grid_x = int(node.position[0] // 10000)  # 10km grid
                    grid_y = int(node.position[1] // 10000)
                    geographic_groups[(grid_x, grid_y)].append(node_id)
        
        # Balance within each geographic group
        for group_nodes in geographic_groups.values():
            if len(group_nodes) > 1:
                await self._least_loaded_balancer(group_nodes)
    
    async def _capability_based_balancer(self, node_ids: List[str]) -> None:
        """Capability-based load balancing."""
        # Group nodes by processing capacity
        high_capacity_nodes = []
        medium_capacity_nodes = []
        low_capacity_nodes = []
        
        for node_id in node_ids:
            if node_id in self.coordination_nodes:
                node = self.coordination_nodes[node_id]
                if node.processing_capacity >= 5.0:
                    high_capacity_nodes.append(node_id)
                elif node.processing_capacity >= 2.0:
                    medium_capacity_nodes.append(node_id)
                else:
                    low_capacity_nodes.append(node_id)
        
        # Apply different strategies to different capacity groups
        for group in [high_capacity_nodes, medium_capacity_nodes, low_capacity_nodes]:
            if len(group) > 1:
                await self._weighted_fair_balancer(group)
    
    async def _ai_optimized_balancer(self, node_ids: List[str]) -> None:
        """AI-optimized load balancing using machine learning."""
        # Simplified AI optimization using performance prediction
        
        # Collect current state features
        features = []
        for node_id in node_ids:
            if node_id in self.coordination_nodes:
                node = self.coordination_nodes[node_id]
                node_features = [
                    node.current_load,
                    node.processing_capacity,
                    node.current_children / max(1, node.max_children),
                    node.latency_to_parent / 100.0,  # Normalize latency
                    len(node.child_nodes) / max(1, node.max_children)
                ]
                features.append((node_id, node_features))
        
        if len(features) < 2:
            return
        
        # Simple optimization: minimize load variance
        loads = [self.coordination_nodes[node_id].current_load for node_id, _ in features]
        mean_load = sum(loads) / len(loads)
        
        for node_id, node_features in features:
            node = self.coordination_nodes[node_id]
            
            # Predict optimal load based on capacity and current state
            capacity_factor = node.processing_capacity / 10.0  # Normalize
            optimal_load = mean_load * capacity_factor
            
            # Gradual adjustment with learning rate
            learning_rate = self.performance_optimizations["default"].learning_rate
            adjustment = (optimal_load - node.current_load) * learning_rate
            node.current_load = max(0.0, min(1.0, node.current_load + adjustment))
    
    async def _predictive_scaling_loop(self) -> None:
        """Predictive scaling based on trends and patterns."""
        while True:
            try:
                await self._analyze_scaling_patterns()
                await self._predict_future_load()
                await self._proactive_scaling()
                await asyncio.sleep(30.0)  # Predict every 30 seconds
            except Exception as e:
                logger.error(f"Predictive scaling error: {e}")
                await asyncio.sleep(10.0)
    
    async def _analyze_scaling_patterns(self) -> None:
        """Analyze historical patterns for predictive scaling."""
        # Analyze load patterns over time
        load_history = []
        for node_id in self.coordination_nodes:
            node_load_history = list(self.performance_metrics[f"{node_id}_load"])
            if node_load_history:
                load_history.extend(node_load_history)
        
        if len(load_history) < 10:
            return
        
        # Simple trend analysis
        recent_loads = load_history[-10:]
        trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        
        # Store trend for prediction
        self.prediction_cache["load_trend"] = trend
        self.prediction_cache["recent_average"] = sum(recent_loads) / len(recent_loads)
    
    async def _predict_future_load(self) -> None:
        """Predict future load based on historical data."""
        current_avg_load = self.scalability_metrics["resource_utilization"]
        trend = self.prediction_cache.get("load_trend", 0.0)
        
        # Simple linear prediction for next 5 minutes
        prediction_horizon = 5.0  # minutes
        predicted_load = current_avg_load + (trend * prediction_horizon)
        
        self.prediction_cache["predicted_load"] = max(0.0, min(1.0, predicted_load))
        self.prediction_cache["prediction_confidence"] = min(1.0, abs(trend) * 10)
    
    async def _proactive_scaling(self) -> None:
        """Perform proactive scaling based on predictions."""
        predicted_load = self.prediction_cache.get("predicted_load", 0.0)
        confidence = self.prediction_cache.get("prediction_confidence", 0.0)
        
        if confidence < 0.5:  # Low confidence, skip proactive scaling
            return
        
        policy = self.scaling_policies["default"]
        
        # Proactive scale-up if high load predicted
        if predicted_load > policy.scale_up_threshold * 0.9:  # Scale up early
            await self._proactive_scale_up()
        
        # Proactive scale-down if low load predicted
        elif predicted_load < policy.scale_down_threshold * 1.1:  # Scale down early
            await self._proactive_scale_down()
    
    async def _proactive_scale_up(self) -> None:
        """Proactive scale-up before load increases."""
        # Find nodes approaching capacity
        candidates = []
        for node_id, node in self.coordination_nodes.items():
            if (node.node_type in ["regional", "local"] and
                node.current_load > 0.6 and
                node.current_children < node.max_children):
                candidates.append(node_id)
        
        # Scale up top candidates
        for node_id in candidates[:3]:  # Limit to 3 scale-ups
            await self._plan_scale_up(self.coordination_nodes[node_id].level, [node_id])
    
    async def _proactive_scale_down(self) -> None:
        """Proactive scale-down before load decreases."""
        # Find underutilized nodes that can be consolidated
        for level in self.hierarchy_levels:
            if level == 0:  # Skip primary
                continue
            
            underutilized = [
                node_id for node_id in self.hierarchy_levels[level]
                if (node_id in self.coordination_nodes and
                    self.coordination_nodes[node_id].current_load < 0.2)
            ]
            
            # Scale down excess underutilized nodes
            if len(underutilized) > 2:  # Keep at least 2 nodes
                await self._plan_scale_down(level, underutilized[2:])
    
    async def _rebalance_global_load(self) -> None:
        """Rebalance load across the entire hierarchy."""
        # Update routing tables based on current topology
        await self._update_routing_tables()
        
        # Optimize inter-node communication
        await self._optimize_communication_paths()
    
    async def _update_routing_tables(self) -> None:
        """Update routing tables for optimal communication."""
        for node_id, node in self.coordination_nodes.items():
            routing_table = {}
            
            # Calculate routes to all other nodes
            for target_id, target_node in self.coordination_nodes.items():
                if target_id != node_id:
                    # Simple routing: use hierarchy path
                    route_cost = self._calculate_route_cost(node, target_node)
                    routing_table[target_id] = route_cost
            
            self.routing_tables[node_id] = routing_table
    
    def _calculate_route_cost(self, source: CoordinationNode, target: CoordinationNode) -> float:
        """Calculate communication cost between nodes."""
        # Base cost on hierarchy distance and geographic distance
        hierarchy_distance = abs(source.level - target.level)
        
        if source.position and target.position:
            geographic_distance = self._calculate_distance(source.position, target.position)
            normalized_geo_distance = geographic_distance / 100000.0  # Normalize to 100km
        else:
            normalized_geo_distance = 1.0
        
        return hierarchy_distance + normalized_geo_distance
    
    async def _optimize_communication_paths(self) -> None:
        """Optimize communication paths for efficiency."""
        # Identify communication bottlenecks
        bottlenecks = []
        
        for node_id, node in self.coordination_nodes.items():
            if node.current_children > node.max_children * 0.8:  # Near capacity
                communication_load = node.current_children * node.current_load
                bottlenecks.append((communication_load, node_id))
        
        # Sort by communication load
        bottlenecks.sort(reverse=True)
        
        # Address top bottlenecks
        for _, node_id in bottlenecks[:5]:  # Top 5 bottlenecks
            await self._relieve_communication_bottleneck(node_id)
    
    async def _relieve_communication_bottleneck(self, node_id: str) -> None:
        """Relieve communication bottleneck at specific node."""
        node = self.coordination_nodes.get(node_id)
        if not node:
            return
        
        # Strategy 1: Create peer connections to distribute load
        peer_candidates = [
            other_id for other_id, other_node in self.coordination_nodes.items()
            if (other_id != node_id and
                other_node.level == node.level and
                other_node.current_load < 0.6)
        ]
        
        # Add peer connections
        for peer_id in peer_candidates[:3]:  # Add up to 3 peers
            node.peer_nodes.add(peer_id)
            if peer_id in self.coordination_nodes:
                self.coordination_nodes[peer_id].peer_nodes.add(node_id)
        
        logger.info(f"Relieved communication bottleneck at {node_id}")
    
    async def _update_load_distribution(self, coordinator_id: str) -> None:
        """Update load distribution after drone registration."""
        coordinator = self.coordination_nodes.get(coordinator_id)
        if not coordinator:
            return
        
        # Recalculate load
        coordinator.current_load = coordinator.current_children / max(1, coordinator.max_children)
        
        # Trigger rebalancing if needed
        if coordinator.current_load > 0.8:
            await self._perform_load_balancing()
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        current_time = time.time()
        
        # Calculate hierarchy statistics
        hierarchy_stats = {}
        for level, node_ids in self.hierarchy_levels.items():
            level_nodes = [self.coordination_nodes[nid] for nid in node_ids if nid in self.coordination_nodes]
            
            if level_nodes:
                avg_load = sum(node.current_load for node in level_nodes) / len(level_nodes)
                avg_children = sum(node.current_children for node in level_nodes) / len(level_nodes)
                
                hierarchy_stats[f"level_{level}"] = {
                    "node_count": len(level_nodes),
                    "average_load": avg_load,
                    "average_children": avg_children,
                    "total_capacity": sum(node.processing_capacity for node in level_nodes)
                }
        
        # Performance optimization status
        optimization_status = {}
        for opt_id, optimization in self.performance_optimizations.items():
            optimization_status[opt_id] = {
                "target_metrics": [metric.value for metric in optimization.target_metrics],
                "load_balancing_method": optimization.load_balancing_method.value,
                "optimization_interval": optimization.optimization_interval,
                "auto_scaling_enabled": optimization.auto_scaling_enabled
            }
        
        return {
            "scaling_overview": {
                "total_coordination_nodes": len(self.coordination_nodes),
                "hierarchy_depth": max(self.hierarchy_levels.keys()) if self.hierarchy_levels else 0,
                "active_drones": sum(1 for node in self.coordination_nodes.values() if node.node_type == "drone"),
                "scaling_strategy": self.scaling_policies["default"].scaling_strategy.value
            },
            "performance_metrics": self.scalability_metrics.copy(),
            "hierarchy_statistics": hierarchy_stats,
            "optimization_status": optimization_status,
            "load_balancing": {
                "routing_tables_count": len(self.routing_tables),
                "communication_optimization": "active",
                "bottleneck_detection": "enabled"
            },
            "predictive_scaling": {
                "prediction_confidence": self.prediction_cache.get("prediction_confidence", 0.0),
                "predicted_load": self.prediction_cache.get("predicted_load", 0.0),
                "load_trend": self.prediction_cache.get("load_trend", 0.0)
            }
        }