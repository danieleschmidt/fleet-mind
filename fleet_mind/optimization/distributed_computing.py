"""Distributed computing and load balancing for Fleet-Mind scaling."""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.logging import get_logger
from ..utils.circuit_breaker import circuit_breaker, CircuitBreakerConfig
from ..utils.retry import retry


class NodeStatus(Enum):
    """Distributed node status."""
    ACTIVE = "active"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    host: str
    port: int
    cpu_cores: int
    memory_gb: float
    current_load: float = 0.0
    status: NodeStatus = NodeStatus.ACTIVE
    last_heartbeat: float = 0
    capabilities: Set[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {"general"}


@dataclass
class DistributedTask:
    """Task for distributed execution."""
    task_id: str
    task_type: str
    payload: Any
    priority: int = 0
    max_retries: int = 3
    timeout_seconds: float = 60.0
    required_capabilities: Set[str] = None
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = {"general"}


class LoadBalancer:
    """Intelligent load balancer for distributed tasks."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        self.logger = get_logger("load_balancer")
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin_select,
            "least_loaded": self._least_loaded_select,
            "capability_based": self._capability_based_select,
            "geographic": self._geographic_select
        }
        self.current_strategy = "least_loaded"
        self._round_robin_index = 0
        
        # Metrics
        self.task_count = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
    
    def register_node(self, node: ComputeNode):
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        self.logger.info(f"Registered compute node: {node.node_id} ({node.host}:{node.port})")
    
    def update_node_status(self, node_id: str, load: float, status: NodeStatus):
        """Update node status and load."""
        if node_id in self.nodes:
            self.nodes[node_id].current_load = load
            self.nodes[node_id].status = status
            self.nodes[node_id].last_heartbeat = time.time()
    
    def get_available_nodes(self, required_capabilities: Set[str] = None) -> List[ComputeNode]:
        """Get available nodes that meet capability requirements."""
        available = []
        current_time = time.time()
        
        for node in self.nodes.values():
            # Check if node is online (heartbeat within 30 seconds)
            if current_time - node.last_heartbeat > 30:
                node.status = NodeStatus.OFFLINE
                continue
                
            if node.status == NodeStatus.OFFLINE:
                continue
                
            # Check capability requirements
            if required_capabilities and not required_capabilities.issubset(node.capabilities):
                continue
                
            available.append(node)
        
        return available
    
    def _round_robin_select(self, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Round-robin node selection."""
        if not nodes:
            return None
        
        node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return node
    
    def _least_loaded_select(self, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Select node with least load."""
        if not nodes:
            return None
        
        # Filter out busy nodes
        available = [n for n in nodes if n.status == NodeStatus.ACTIVE and n.current_load < 0.8]
        if not available:
            # If all nodes busy, take least loaded
            available = [n for n in nodes if n.status != NodeStatus.OFFLINE]
        
        if not available:
            return None
        
        return min(available, key=lambda n: n.current_load)
    
    def _capability_based_select(self, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Select node based on specialized capabilities."""
        # This could be enhanced with ML model routing, GPU vs CPU selection, etc.
        return self._least_loaded_select(nodes)
    
    def _geographic_select(self, nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Select node based on geographic proximity (mock implementation)."""
        # In real implementation, this would consider latency, region, etc.
        return self._least_loaded_select(nodes)
    
    def select_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select optimal node for task execution."""
        available_nodes = self.get_available_nodes(task.required_capabilities)
        
        if not available_nodes:
            self.logger.warning(f"No available nodes for task {task.task_id}")
            return None
        
        strategy_func = self.strategies.get(self.current_strategy, self._least_loaded_select)
        selected = strategy_func(available_nodes)
        
        if selected:
            self.logger.debug(f"Selected node {selected.node_id} for task {task.task_id}")
        
        return selected
    
    async def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution."""
        node = self.select_node(task)
        if not node:
            raise RuntimeError(f"No available nodes for task {task.task_id}")
        
        await self.task_queue.put((task, node))
        self.task_count += 1
        self.logger.info(f"Submitted task {task.task_id} to node {node.node_id}")
        return task.task_id
    
    async def get_result(self, task_id: str, timeout: float = 60.0) -> Any:
        """Get task result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.results:
                result = self.results.pop(task_id)
                return result
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} timed out")
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
        avg_load = sum(n.current_load for n in self.nodes.values()) / max(len(self.nodes), 1)
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "average_load": avg_load,
            "current_strategy": self.current_strategy,
            "task_count": self.task_count,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(self.task_count, 1),
            "avg_execution_time": self.total_execution_time / max(self.completed_tasks, 1)
        }


class DistributedCoordinator:
    """Coordinates distributed Fleet-Mind operations across multiple nodes."""
    
    def __init__(self, node_id: str = "coordinator"):
        self.node_id = node_id
        self.load_balancer = LoadBalancer()
        self.task_executor = ThreadPoolExecutor(max_workers=10)
        self.logger = get_logger("distributed_coordinator")
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.start_time = time.time()
        self.processed_tasks = 0
        self.processing_times: List[float] = []
    
    async def start(self):
        """Start distributed coordination."""
        self.running = True
        
        # Start worker tasks
        for i in range(5):  # 5 worker tasks
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        # Start health monitoring
        monitor_task = asyncio.create_task(self._health_monitor())
        self.worker_tasks.append(monitor_task)
        
        self.logger.info(f"Started distributed coordinator with {len(self.worker_tasks)} workers")
    
    async def stop(self):
        """Stop distributed coordination."""
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.task_executor.shutdown(wait=True)
        
        self.logger.info("Stopped distributed coordinator")
    
    @retry(max_attempts=3, initial_delay=1.0)
    @circuit_breaker("distributed_task", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60))
    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing distributed tasks."""
        self.logger.debug(f"Started worker {worker_id}")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task, node = await asyncio.wait_for(
                    self.load_balancer.task_queue.get(), 
                    timeout=5.0
                )
                
                # Execute task
                start_time = time.time()
                result = await self._execute_task(task, node)
                execution_time = time.time() - start_time
                
                # Store result
                self.load_balancer.results[task.task_id] = result
                self.load_balancer.completed_tasks += 1
                self.load_balancer.total_execution_time += execution_time
                
                # Update metrics
                self.processed_tasks += 1
                self.processing_times.append(execution_time)
                
                # Keep only recent processing times (last 100)
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
                
                self.logger.debug(f"Worker {worker_id} completed task {task.task_id} in {execution_time:.2f}s")
                
            except asyncio.TimeoutError:
                continue  # No tasks available, continue waiting
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                if 'task' in locals():
                    self.load_balancer.failed_tasks += 1
        
        self.logger.debug(f"Stopped worker {worker_id}")
    
    async def _execute_task(self, task: DistributedTask, node: ComputeNode) -> Any:
        """Execute a distributed task on specified node."""
        # Update node status
        self.load_balancer.update_node_status(node.node_id, node.current_load + 0.1, NodeStatus.BUSY)
        
        try:
            # Mock task execution - in real implementation this would call node APIs
            if task.task_type == "llm_planning":
                result = await self._execute_llm_planning(task.payload)
            elif task.task_type == "image_processing":
                result = await self._execute_image_processing(task.payload)
            elif task.task_type == "path_calculation":
                result = await self._execute_path_calculation(task.payload)
            else:
                result = await self._execute_generic_task(task.payload)
            
            return {"success": True, "result": result, "node_id": node.node_id}
            
        finally:
            # Update node status back to active
            self.load_balancer.update_node_status(node.node_id, max(0, node.current_load - 0.1), NodeStatus.ACTIVE)
    
    async def _execute_llm_planning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM planning task."""
        # Simulate LLM processing
        await asyncio.sleep(0.5)  # Mock processing time
        
        return {
            "plan": {
                "objectives": ["Execute mission", "Maintain safety"],
                "actions": ["form_formation", "navigate", "land"],
                "duration": 300
            },
            "confidence": 0.85
        }
    
    async def _execute_image_processing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image processing task."""
        await asyncio.sleep(0.2)  # Mock processing time
        
        return {
            "objects_detected": ["building", "vehicle", "person"],
            "confidence_scores": [0.9, 0.8, 0.7],
            "processing_time_ms": 200
        }
    
    async def _execute_path_calculation(self, payload: Dict[str, Any]) -> List[tuple]:
        """Execute path calculation task."""
        await asyncio.sleep(0.1)  # Mock processing time
        
        # Mock path calculation
        start = payload.get("start", (0, 0))
        end = payload.get("end", (100, 100))
        
        # Simple path (in real implementation would use A* or similar)
        path = [
            start,
            (start[0] + (end[0] - start[0]) * 0.5, start[1] + (end[1] - start[1]) * 0.5),
            end
        ]
        
        return path
    
    async def _execute_generic_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic task."""
        await asyncio.sleep(0.05)  # Mock processing time
        return {"status": "completed", "payload_size": len(str(payload))}
    
    async def _health_monitor(self):
        """Monitor health of distributed nodes."""
        while self.running:
            try:
                # Check node health
                current_time = time.time()
                offline_nodes = []
                
                for node_id, node in self.load_balancer.nodes.items():
                    if current_time - node.last_heartbeat > 30:
                        if node.status != NodeStatus.OFFLINE:
                            self.logger.warning(f"Node {node_id} went offline")
                            node.status = NodeStatus.OFFLINE
                            offline_nodes.append(node_id)
                
                # Log health summary periodically
                if int(current_time) % 60 == 0:  # Every minute
                    stats = self.get_performance_stats()
                    self.logger.info(f"Distributed system health: {stats['active_nodes']} active nodes, "
                                   f"{stats['avg_processing_time']:.3f}s avg processing time")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def execute_distributed_planning(self, mission_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mission planning using distributed nodes."""
        task = DistributedTask(
            task_id=f"plan_{int(time.time() * 1000)}",
            task_type="llm_planning",
            payload=mission_context,
            required_capabilities={"llm_processing"}
        )
        
        task_id = await self.load_balancer.submit_task(task)
        result = await self.load_balancer.get_result(task_id, timeout=30.0)
        
        return result
    
    async def execute_distributed_image_processing(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute image processing using distributed nodes."""
        task = DistributedTask(
            task_id=f"img_{int(time.time() * 1000)}",
            task_type="image_processing",
            payload=image_data,
            required_capabilities={"image_processing", "gpu"}
        )
        
        task_id = await self.load_balancer.submit_task(task)
        result = await self.load_balancer.get_result(task_id, timeout=15.0)
        
        return result
    
    async def execute_distributed_path_planning(self, start: tuple, end: tuple, obstacles: List = None) -> List[tuple]:
        """Execute path planning using distributed nodes."""
        task = DistributedTask(
            task_id=f"path_{int(time.time() * 1000)}",
            task_type="path_calculation",
            payload={"start": start, "end": end, "obstacles": obstacles or []},
            required_capabilities={"path_planning"}
        )
        
        task_id = await self.load_balancer.submit_task(task)
        result = await self.load_balancer.get_result(task_id, timeout=10.0)
        
        return result
    
    def add_compute_node(self, host: str, port: int, cpu_cores: int, memory_gb: float, capabilities: Set[str] = None):
        """Add a compute node to the distributed system."""
        node_id = f"{host}:{port}"
        node = ComputeNode(
            node_id=node_id,
            host=host,
            port=port,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            capabilities=capabilities or {"general"}
        )
        
        self.load_balancer.register_node(node)
        return node_id
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        load_stats = self.load_balancer.get_load_stats()
        
        avg_processing_time = sum(self.processing_times) / max(len(self.processing_times), 1)
        
        return {
            **load_stats,
            "uptime_seconds": time.time() - self.start_time,
            "processed_tasks": self.processed_tasks,
            "avg_processing_time": avg_processing_time,
            "worker_count": len(self.worker_tasks),
            "queue_size": self.load_balancer.task_queue.qsize(),
        }


# Global distributed coordinator instance
_distributed_coordinator: Optional[DistributedCoordinator] = None

async def get_distributed_coordinator() -> DistributedCoordinator:
    """Get or create global distributed coordinator."""
    global _distributed_coordinator
    
    if _distributed_coordinator is None:
        _distributed_coordinator = DistributedCoordinator()
        
        # Add default local nodes for testing
        _distributed_coordinator.add_compute_node("localhost", 8001, 4, 8.0, {"general", "llm_processing"})
        _distributed_coordinator.add_compute_node("localhost", 8002, 8, 16.0, {"general", "image_processing", "gpu"})
        _distributed_coordinator.add_compute_node("localhost", 8003, 4, 8.0, {"general", "path_planning"})
        
        await _distributed_coordinator.start()
    
    return _distributed_coordinator


async def shutdown_distributed_coordinator():
    """Shutdown global distributed coordinator."""
    global _distributed_coordinator
    
    if _distributed_coordinator:
        await _distributed_coordinator.stop()
        _distributed_coordinator = None