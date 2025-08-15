"""Edge Computing Coordinator for Ultra-Low Latency Processing.

Advanced edge computing orchestration:
- Dynamic workload placement across edge nodes
- Real-time latency optimization
- Intelligent load balancing with predictive scaling
- Edge-cloud hybrid processing
"""

import asyncio
import time
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import concurrent.futures

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes."""
    DRONE_ONBOARD = "drone_onboard"      # On-drone computing
    GROUND_STATION = "ground_station"    # Local ground stations
    MOBILE_TOWER = "mobile_tower"        # 5G base stations
    EDGE_DATACENTER = "edge_datacenter"  # Regional edge DCs
    CLOUD_INSTANCE = "cloud_instance"    # Cloud fallback


class WorkloadType(Enum):
    """Types of computational workloads."""
    PATH_PLANNING = "path_planning"
    COLLISION_AVOIDANCE = "collision_avoidance"
    COMPUTER_VISION = "computer_vision"
    SENSOR_FUSION = "sensor_fusion"
    COORDINATION = "coordination"
    COMMUNICATION = "communication"


class ProcessingPriority(Enum):
    """Processing priority levels."""
    CRITICAL = "critical"      # <1ms latency
    HIGH = "high"             # <10ms latency
    MEDIUM = "medium"         # <100ms latency
    LOW = "low"              # <1s latency


@dataclass
class ComputeWorkload:
    """Computational workload definition."""
    workload_id: str
    workload_type: WorkloadType
    priority: ProcessingPriority
    
    # Resource requirements
    cpu_cores: float = 1.0
    memory_mb: int = 512
    gpu_memory_mb: int = 0
    storage_mb: int = 100
    
    # Timing constraints
    deadline_ms: float = 100.0
    arrival_time: float = 0.0
    estimated_runtime_ms: float = 50.0
    
    # Data characteristics
    input_size_kb: int = 10
    output_size_kb: int = 1
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependent_nodes: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize workload timestamps."""
        if self.arrival_time == 0.0:
            self.arrival_time = time.time() * 1000  # ms


@dataclass
class EdgeNode:
    """Edge computing node representation."""
    node_id: str
    node_type: EdgeNodeType
    location: Tuple[float, float, float]  # lat, lon, alt
    
    # Compute capabilities
    cpu_cores: int = 4
    memory_mb: int = 8192
    gpu_memory_mb: int = 0
    storage_gb: int = 100
    
    # Current utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    
    # Network characteristics
    network_latency_ms: float = 5.0
    bandwidth_mbps: float = 1000.0
    packet_loss_rate: float = 0.001
    
    # Availability and reliability
    availability: float = 0.99
    processing_queue: deque = field(default_factory=deque)
    active_workloads: Dict[str, ComputeWorkload] = field(default_factory=dict)
    
    # Performance history
    completion_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 0.95
    
    def __post_init__(self):
        """Initialize edge node state."""
        if not hasattr(self, 'processing_queue'):
            self.processing_queue = deque()
        if not hasattr(self, 'active_workloads'):
            self.active_workloads = {}
        if not hasattr(self, 'completion_times'):
            self.completion_times = deque(maxlen=100)
    
    def get_available_resources(self) -> Dict[str, float]:
        """Get currently available resources."""
        return {
            "cpu_cores": self.cpu_cores * (1.0 - self.cpu_utilization),
            "memory_mb": self.memory_mb * (1.0 - self.memory_utilization),
            "gpu_memory_mb": self.gpu_memory_mb * (1.0 - self.gpu_utilization)
        }
    
    def can_accept_workload(self, workload: ComputeWorkload) -> bool:
        """Check if node can accept workload."""
        available = self.get_available_resources()
        
        return (available["cpu_cores"] >= workload.cpu_cores and
                available["memory_mb"] >= workload.memory_mb and
                available["gpu_memory_mb"] >= workload.gpu_memory_mb and
                len(self.processing_queue) < 10)  # Queue limit
    
    def estimate_completion_time(self, workload: ComputeWorkload) -> float:
        """Estimate workload completion time."""
        # Base processing time
        base_time = workload.estimated_runtime_ms
        
        # Adjust for current load
        load_factor = 1.0 + self.cpu_utilization
        
        # Queue delay
        queue_delay = len(self.processing_queue) * 10.0  # 10ms per queued job
        
        return base_time * load_factor + queue_delay


class EdgeCoordinator:
    """Coordinates workload placement across edge computing infrastructure."""
    
    def __init__(self, optimization_frequency: float = 10.0):
        self.optimization_frequency = optimization_frequency
        
        # Edge infrastructure
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.workload_history: Dict[str, ComputeWorkload] = {}
        
        # Workload queues
        self.pending_workloads: deque = deque()
        self.completed_workloads: deque = deque(maxlen=1000)
        self.failed_workloads: deque = deque(maxlen=100)
        
        # Performance metrics
        self.average_latency_ms: float = 0.0
        self.workload_success_rate: float = 0.0
        self.resource_utilization: float = 0.0
        self.cost_efficiency: float = 0.0
        
        # Optimization state
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Predictive models (simplified)
        self.workload_predictor = WorkloadPredictor()
        self.latency_predictor = LatencyPredictor()
        
        logger.info("Edge coordinator initialized")
    
    async def start(self) -> None:
        """Start edge coordination."""
        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Edge coordination started")
    
    async def stop(self) -> None:
        """Stop edge coordination."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Edge coordination stopped")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while self._running:
            try:
                await asyncio.sleep(1.0 / self.optimization_frequency)
                
                await self._process_pending_workloads()
                await self._update_node_status()
                await self._optimize_placement()
                await self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Edge optimization error: {e}")
    
    async def _process_pending_workloads(self) -> None:
        """Process pending workloads for placement."""
        processed = []
        
        while self.pending_workloads:
            workload = self.pending_workloads.popleft()
            
            # Check deadline
            current_time = time.time() * 1000
            if current_time > workload.arrival_time + workload.deadline_ms:
                self.failed_workloads.append(workload)
                logger.warning(f"Workload {workload.workload_id} missed deadline")
                continue
            
            # Find optimal placement
            best_node = await self._find_optimal_placement(workload)
            
            if best_node:
                await self._assign_workload(best_node, workload)
                processed.append(workload)
            else:
                # Return to queue for retry
                self.pending_workloads.appendleft(workload)
                break
    
    async def _find_optimal_placement(self, workload: ComputeWorkload) -> Optional[EdgeNode]:
        """Find optimal edge node for workload placement."""
        eligible_nodes = []
        
        # Filter nodes that can accept workload
        for node in self.edge_nodes.values():
            if node.can_accept_workload(workload):
                eligible_nodes.append(node)
        
        if not eligible_nodes:
            return None
        
        # Multi-criteria optimization
        best_node = None
        best_score = float('inf')
        
        for node in eligible_nodes:
            score = await self._calculate_placement_score(node, workload)
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    async def _calculate_placement_score(self, node: EdgeNode, workload: ComputeWorkload) -> float:
        """Calculate placement score (lower is better)."""
        # Estimated completion time
        completion_time = node.estimate_completion_time(workload)
        
        # Network latency
        network_latency = node.network_latency_ms
        
        # Resource utilization penalty
        utilization_penalty = (node.cpu_utilization + node.memory_utilization) * 50
        
        # Priority weighting
        priority_weight = {
            ProcessingPriority.CRITICAL: 0.1,
            ProcessingPriority.HIGH: 0.5,
            ProcessingPriority.MEDIUM: 1.0,
            ProcessingPriority.LOW: 2.0
        }[workload.priority]
        
        # Distance penalty for drone workloads
        distance_penalty = 0.0
        if workload.workload_type in [WorkloadType.COLLISION_AVOIDANCE, WorkloadType.PATH_PLANNING]:
            # Prefer closer nodes for real-time workloads
            distance_penalty = node.network_latency_ms * 2
        
        # Combined score
        score = (completion_time + network_latency + utilization_penalty + distance_penalty) * priority_weight
        
        return score
    
    async def _assign_workload(self, node: EdgeNode, workload: ComputeWorkload) -> None:
        """Assign workload to edge node."""
        node.processing_queue.append(workload)
        node.active_workloads[workload.workload_id] = workload
        
        # Update resource utilization
        node.cpu_utilization = min(1.0, node.cpu_utilization + workload.cpu_cores / node.cpu_cores)
        node.memory_utilization = min(1.0, node.memory_utilization + workload.memory_mb / node.memory_mb)
        
        # Start processing
        asyncio.create_task(self._process_workload(node, workload))
        
        logger.debug(f"Assigned workload {workload.workload_id} to node {node.node_id}")
    
    async def _process_workload(self, node: EdgeNode, workload: ComputeWorkload) -> None:
        """Process workload on edge node."""
        start_time = time.time() * 1000
        
        try:
            # Simulate processing time
            processing_time = node.estimate_completion_time(workload)
            await asyncio.sleep(processing_time / 1000.0)
            
            # Mark as completed
            completion_time = time.time() * 1000
            total_time = completion_time - start_time
            
            node.completion_times.append(total_time)
            self.completed_workloads.append(workload)
            
            # Update node state
            if workload.workload_id in node.active_workloads:
                del node.active_workloads[workload.workload_id]
            
            # Release resources
            node.cpu_utilization = max(0.0, node.cpu_utilization - workload.cpu_cores / node.cpu_cores)
            node.memory_utilization = max(0.0, node.memory_utilization - workload.memory_mb / node.memory_mb)
            
            logger.debug(f"Completed workload {workload.workload_id} in {total_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Workload processing failed: {e}")
            self.failed_workloads.append(workload)
            
            # Clean up
            if workload.workload_id in node.active_workloads:
                del node.active_workloads[workload.workload_id]
    
    async def _update_node_status(self) -> None:
        """Update status of all edge nodes."""
        for node in self.edge_nodes.values():
            # Update availability based on recent performance
            if node.completion_times:
                avg_completion = sum(node.completion_times) / len(node.completion_times)
                node.success_rate = min(1.0, 100.0 / max(1.0, avg_completion))
            
            # Simulate network conditions
            if random.random() < 0.01:  # 1% chance of network change
                node.network_latency_ms *= random.uniform(0.8, 1.2)
                node.bandwidth_mbps *= random.uniform(0.9, 1.1)
    
    async def _optimize_placement(self) -> None:
        """Optimize current workload placement."""
        # Migration strategies for overloaded nodes
        overloaded_nodes = [
            node for node in self.edge_nodes.values()
            if node.cpu_utilization > 0.8 or node.memory_utilization > 0.8
        ]
        
        for node in overloaded_nodes:
            # Consider migrating low-priority workloads
            for workload_id, workload in list(node.active_workloads.items()):
                if workload.priority in [ProcessingPriority.LOW, ProcessingPriority.MEDIUM]:
                    # Find alternative node
                    alternative = await self._find_optimal_placement(workload)
                    if alternative and alternative != node:
                        await self._migrate_workload(node, alternative, workload)
                        break
    
    async def _migrate_workload(self, source_node: EdgeNode, 
                               target_node: EdgeNode, workload: ComputeWorkload) -> None:
        """Migrate workload between edge nodes."""
        try:
            # Remove from source
            if workload.workload_id in source_node.active_workloads:
                del source_node.active_workloads[workload.workload_id]
            
            # Assign to target
            await self._assign_workload(target_node, workload)
            
            logger.info(f"Migrated workload {workload.workload_id} from {source_node.node_id} to {target_node.node_id}")
            
        except Exception as e:
            logger.error(f"Workload migration failed: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """Update system performance metrics."""
        if self.completed_workloads:
            # Average latency
            recent_completions = list(self.completed_workloads)[-100:]  # Last 100
            completion_times = []
            
            for node in self.edge_nodes.values():
                completion_times.extend(list(node.completion_times))
            
            if completion_times:
                self.average_latency_ms = sum(completion_times) / len(completion_times)
        
        # Success rate
        total_processed = len(self.completed_workloads) + len(self.failed_workloads)
        if total_processed > 0:
            self.workload_success_rate = len(self.completed_workloads) / total_processed
        
        # Resource utilization
        if self.edge_nodes:
            total_utilization = sum(
                (node.cpu_utilization + node.memory_utilization) / 2
                for node in self.edge_nodes.values()
            )
            self.resource_utilization = total_utilization / len(self.edge_nodes)
        
        # Cost efficiency (simplified metric)
        if self.average_latency_ms > 0:
            self.cost_efficiency = self.workload_success_rate / (self.average_latency_ms / 100.0)
    
    async def register_edge_node(self, node: EdgeNode) -> None:
        """Register new edge node."""
        self.edge_nodes[node.node_id] = node
        logger.info(f"Registered edge node: {node.node_id} ({node.node_type.value})")
    
    async def submit_workload(self, workload: ComputeWorkload) -> None:
        """Submit workload for processing."""
        self.pending_workloads.append(workload)
        self.workload_history[workload.workload_id] = workload
        logger.debug(f"Submitted workload: {workload.workload_id}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        node_summary = {}
        for node_type in EdgeNodeType:
            nodes = [n for n in self.edge_nodes.values() if n.node_type == node_type]
            if nodes:
                avg_utilization = sum((n.cpu_utilization + n.memory_utilization) / 2 for n in nodes) / len(nodes)
                node_summary[node_type.value] = {
                    "count": len(nodes),
                    "avg_utilization": avg_utilization
                }
        
        return {
            "edge_nodes": len(self.edge_nodes),
            "pending_workloads": len(self.pending_workloads),
            "active_workloads": sum(len(n.active_workloads) for n in self.edge_nodes.values()),
            "completed_workloads": len(self.completed_workloads),
            "failed_workloads": len(self.failed_workloads),
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.workload_success_rate,
            "resource_utilization": self.resource_utilization,
            "cost_efficiency": self.cost_efficiency,
            "node_summary": node_summary,
            "running": self._running
        }


class WorkloadPredictor:
    """Predicts future workload patterns."""
    
    def __init__(self):
        self.workload_history: deque = deque(maxlen=1000)
        self.prediction_accuracy = 0.75
    
    def predict_workload_demand(self, time_horizon_minutes: int = 10) -> Dict[WorkloadType, int]:
        """Predict workload demand for next time horizon."""
        # Simplified prediction based on historical patterns
        base_demand = {
            WorkloadType.PATH_PLANNING: 20,
            WorkloadType.COLLISION_AVOIDANCE: 50,
            WorkloadType.COMPUTER_VISION: 30,
            WorkloadType.SENSOR_FUSION: 40,
            WorkloadType.COORDINATION: 15,
            WorkloadType.COMMUNICATION: 25
        }
        
        # Add some randomness and time-of-day effects
        current_hour = time.localtime().tm_hour
        time_factor = 1.0 + 0.3 * math.sin(2 * math.pi * current_hour / 24)
        
        predicted_demand = {}
        for workload_type, base_count in base_demand.items():
            variance = random.uniform(0.8, 1.2)
            predicted_demand[workload_type] = int(base_count * time_factor * variance)
        
        return predicted_demand


class LatencyPredictor:
    """Predicts network latency and processing times."""
    
    def __init__(self):
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prediction_accuracy = 0.80
    
    def predict_latency(self, source: str, destination: str) -> float:
        """Predict network latency between nodes."""
        historical_latencies = self.latency_history[f"{source}-{destination}"]
        
        if len(historical_latencies) >= 3:
            # Use exponential smoothing
            alpha = 0.3
            prediction = historical_latencies[-1]
            for i in range(len(historical_latencies) - 2, -1, -1):
                prediction = alpha * historical_latencies[i] + (1 - alpha) * prediction
            return max(1.0, prediction)
        else:
            # Default estimate
            return 10.0  # ms