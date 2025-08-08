"""Distributed processing system for Fleet-Mind with intelligent load balancing and resource optimization."""

import asyncio
import time
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import queue
import heapq
import hashlib
import statistics


class ProcessingMode(Enum):
    """Processing execution modes."""
    SINGLE_THREADED = "single_threaded"
    MULTI_THREADED = "multi_threaded"
    MULTI_PROCESS = "multi_process"
    ASYNC_CONCURRENT = "async_concurrent"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    HASH_BASED = "hash_based"
    PRIORITY_QUEUE = "priority_queue"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    timeout: Optional[float] = None
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    estimated_duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        if not self.task_id:
            # Generate unique ID based on function and args
            content = f"{self.function.__name__}_{str(self.args)}_{str(self.kwargs)}_{self.created_at}"
            self.task_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    current_load: float = 0.0  # 0.0 to 1.0
    last_task_time: Optional[float] = None
    error_rate: float = 0.0
    average_task_duration: float = 0.0
    queue_size: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 1.0


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    worker_id: Optional[str] = None
    retry_count: int = 0
    completed_at: float = field(default_factory=time.time)


class DistributedProcessor:
    """Advanced distributed processing system with intelligent load balancing."""
    
    def __init__(
        self,
        max_workers: int = None,
        processing_mode: ProcessingMode = ProcessingMode.HYBRID,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        enable_auto_scaling: bool = True,
        task_timeout: float = 300.0,
        queue_size_limit: int = 10000,
        worker_health_check_interval: float = 30.0,
    ):
        """Initialize distributed processor.
        
        Args:
            max_workers: Maximum number of workers (auto-detected if None)
            processing_mode: Processing execution mode
            load_balancing: Load balancing strategy
            enable_auto_scaling: Enable dynamic scaling
            task_timeout: Default task timeout in seconds
            queue_size_limit: Maximum queue size
            worker_health_check_interval: Worker health check interval
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.processing_mode = processing_mode
        self.load_balancing = load_balancing
        self.enable_auto_scaling = enable_auto_scaling
        self.task_timeout = task_timeout
        self.queue_size_limit = queue_size_limit
        self.worker_health_check_interval = worker_health_check_interval
        
        # Processing infrastructure
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size_limit)
        self.pending_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        self.failed_tasks: Dict[str, ProcessingResult] = {}
        
        # Worker management
        self.workers: Dict[str, Any] = {}  # worker_id -> worker
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.worker_queues: Dict[str, queue.Queue] = {}
        self.active_workers = 0
        self.next_worker_index = 0
        
        # Executors
        self.thread_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.process_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.task_history: deque = deque(maxlen=1000)
        
        # Auto-scaling metrics
        self.queue_size_history: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=100)
        self.load_history: deque = deque(maxlen=100)
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Background tasks
        self.monitor_thread: Optional[threading.Thread] = None
        self.health_check_thread: Optional[threading.Thread] = None
        
        # Initialize processing infrastructure
        self._initialize_workers()
        
        print(f"Distributed processor initialized with {self.active_workers} workers")
    
    def _initialize_workers(self):
        """Initialize worker pool based on processing mode."""
        if self.processing_mode in [ProcessingMode.MULTI_THREADED, ProcessingMode.HYBRID]:
            self.thread_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="FleetMind-Worker"
            )
        
        if self.processing_mode in [ProcessingMode.MULTI_PROCESS, ProcessingMode.HYBRID]:
            try:
                self.process_executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(self.max_workers // 2, multiprocessing.cpu_count())
                )
            except Exception as e:
                print(f"Failed to initialize process executor: {e}")
        
        # Initialize worker stats
        for i in range(self.max_workers):
            worker_id = f"worker_{i:03d}"
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
            self.worker_queues[worker_id] = queue.Queue()
        
        self.active_workers = self.max_workers
    
    def start(self):
        """Start the distributed processor."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start background monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ProcessorMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="WorkerHealthCheck",
            daemon=True
        )
        self.health_check_thread.start()
        
        print("Distributed processor started")
    
    def stop(self, timeout: float = 30.0):
        """Stop the distributed processor."""
        if not self.running:
            return
        
        print("Stopping distributed processor...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for background threads
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True, timeout=timeout)
        if self.process_executor:
            self.process_executor.shutdown(wait=True, timeout=timeout)
        
        print("Distributed processor stopped")
    
    def submit_task(
        self,
        function: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        estimated_duration: Optional[float] = None,
        resource_requirements: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """Submit a task for processing.
        
        Args:
            function: Function to execute
            *args: Function arguments
            task_id: Optional task ID
            priority: Task priority (higher = more priority)
            timeout: Task timeout
            estimated_duration: Estimated duration in seconds
            resource_requirements: Required resources
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if not self.running:
            raise RuntimeError("Processor not running")
        
        task = ProcessingTask(
            task_id=task_id or "",
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.task_timeout,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements or {}
        )
        
        # Add to pending tasks
        self.pending_tasks[task.task_id] = task
        
        # Add to queue with priority
        try:
            # Priority queue uses negative priority for max-heap behavior
            self.task_queue.put((-priority, time.time(), task.task_id), timeout=1.0)
        except queue.Full:
            # Queue full - could implement backpressure or reject
            self.pending_tasks.pop(task.task_id, None)
            raise RuntimeError("Task queue full")
        
        return task.task_id
    
    async def submit_task_async(
        self,
        function: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """Submit task asynchronously."""
        return self.submit_task(
            function, *args,
            task_id=task_id,
            priority=priority,
            timeout=timeout,
            **kwargs
        )
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """Get task result (blocking)."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            # Check if task still pending
            if task_id not in self.pending_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def get_result_async(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """Get task result asynchronously."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks  
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            # Check if task still pending
            if task_id not in self.pending_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def _select_worker(self, task: ProcessingTask) -> str:
        """Select worker based on load balancing strategy."""
        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            worker_id = f"worker_{self.next_worker_index:03d}"
            self.next_worker_index = (self.next_worker_index + 1) % self.active_workers
            return worker_id
        
        elif self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
            # Find worker with lowest current load
            min_load = float('inf')
            selected_worker = None
            
            for worker_id, stats in self.worker_stats.items():
                if stats.current_load < min_load:
                    min_load = stats.current_load
                    selected_worker = worker_id
            
            return selected_worker or "worker_000"
        
        elif self.load_balancing == LoadBalancingStrategy.HASH_BASED:
            # Hash-based assignment for task affinity
            task_hash = hash(task.task_id) % self.active_workers
            return f"worker_{task_hash:03d}"
        
        elif self.load_balancing == LoadBalancingStrategy.PRIORITY_QUEUE:
            # Use priority-based assignment
            worker_priorities = []
            for worker_id, stats in self.worker_stats.items():
                # Priority based on inverse load and success rate
                priority = stats.success_rate / (stats.current_load + 0.01)
                heapq.heappush(worker_priorities, (-priority, worker_id))
            
            if worker_priorities:
                return heapq.heappop(worker_priorities)[1]
            else:
                return "worker_000"
        
        else:  # ADAPTIVE
            # Adaptive strategy considers multiple factors
            best_score = float('-inf')
            selected_worker = None
            
            for worker_id, stats in self.worker_stats.items():
                # Composite score based on multiple factors
                load_factor = 1.0 - stats.current_load  # Higher is better
                success_factor = stats.success_rate
                speed_factor = 1.0 / (stats.average_task_duration + 0.01)  # Faster is better
                queue_factor = 1.0 / (stats.queue_size + 1)  # Shorter queue is better
                
                score = (load_factor * 0.4 + 
                        success_factor * 0.3 + 
                        speed_factor * 0.2 + 
                        queue_factor * 0.1)
                
                if score > best_score:
                    best_score = score
                    selected_worker = worker_id
            
            return selected_worker or "worker_000"
    
    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Select appropriate executor
            if self.processing_mode == ProcessingMode.SINGLE_THREADED:
                result = task.function(*task.args, **task.kwargs)
                future = None
            
            elif self.processing_mode == ProcessingMode.MULTI_THREADED:
                if self.thread_executor:
                    future = self.thread_executor.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    result = task.function(*task.args, **task.kwargs)
            
            elif self.processing_mode == ProcessingMode.MULTI_PROCESS:
                if self.process_executor:
                    future = self.process_executor.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    result = task.function(*task.args, **task.kwargs)
            
            else:  # HYBRID or ASYNC_CONCURRENT
                # Choose based on task characteristics
                if task.estimated_duration and task.estimated_duration > 10.0:  # Long tasks
                    if self.process_executor:
                        future = self.process_executor.submit(task.function, *task.args, **task.kwargs)
                        result = future.result(timeout=task.timeout)
                    else:
                        result = task.function(*task.args, **task.kwargs)
                else:  # Short tasks
                    if self.thread_executor:
                        future = self.thread_executor.submit(task.function, *task.args, **task.kwargs)
                        result = future.result(timeout=task.timeout)
                    else:
                        result = task.function(*task.args, **task.kwargs)
            
            # Success
            duration = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result,
                duration=duration
            )
        
        except Exception as e:
            # Failure
            duration = time.time() - start_time
            
            return ProcessingResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                duration=duration
            )
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while self.running and not self.shutdown_event.is_set():
                # Process tasks from queue
                try:
                    # Get task from queue (non-blocking with timeout)
                    priority, timestamp, task_id = self.task_queue.get(timeout=1.0)
                    
                    if task_id in self.pending_tasks:
                        task = self.pending_tasks[task_id]
                        
                        # Select worker
                        worker_id = self._select_worker(task)
                        
                        # Execute task
                        result = self._execute_task(task)
                        result.worker_id = worker_id
                        
                        # Update results
                        if result.success:
                            self.completed_tasks[task_id] = result
                        else:
                            self.failed_tasks[task_id] = result
                        
                        # Remove from pending
                        self.pending_tasks.pop(task_id, None)
                        
                        # Update statistics
                        self._update_worker_stats(worker_id, result)
                        self._update_global_stats(result)
                        
                        # Mark task as done
                        self.task_queue.task_done()
                
                except queue.Empty:
                    # No tasks in queue, continue monitoring
                    pass
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                
                # Auto-scaling check
                if self.enable_auto_scaling:
                    self._check_auto_scaling()
                
                # Update performance metrics
                self._update_performance_metrics()
        
        except Exception as e:
            print(f"Monitoring loop error: {e}")
    
    def _health_check_loop(self):
        """Background health check loop."""
        try:
            while self.running and not self.shutdown_event.is_set():
                self._perform_health_checks()
                time.sleep(self.worker_health_check_interval)
        except Exception as e:
            print(f"Health check loop error: {e}")
    
    def _update_worker_stats(self, worker_id: str, result: ProcessingResult):
        """Update worker statistics."""
        if worker_id in self.worker_stats:
            stats = self.worker_stats[worker_id]
            
            if result.success:
                stats.tasks_completed += 1
            else:
                stats.tasks_failed += 1
            
            stats.total_processing_time += result.duration
            stats.last_task_time = result.completed_at
            
            # Update averages
            total_tasks = stats.tasks_completed + stats.tasks_failed
            if total_tasks > 0:
                stats.error_rate = stats.tasks_failed / total_tasks
                stats.average_task_duration = stats.total_processing_time / total_tasks
    
    def _update_global_stats(self, result: ProcessingResult):
        """Update global statistics."""
        self.total_tasks_processed += 1
        self.total_processing_time += result.duration
        
        # Add to history
        self.task_history.append({
            'task_id': result.task_id,
            'success': result.success,
            'duration': result.duration,
            'completed_at': result.completed_at
        })
    
    def _update_performance_metrics(self):
        """Update performance metrics for auto-scaling."""
        current_time = time.time()
        
        # Queue size
        queue_size = self.task_queue.qsize()
        self.queue_size_history.append((current_time, queue_size))
        
        # Calculate throughput (tasks per second)
        if len(self.task_history) > 1:
            recent_tasks = [t for t in self.task_history 
                           if current_time - t['completed_at'] <= 60]  # Last minute
            throughput = len(recent_tasks) / 60.0
            self.throughput_history.append((current_time, throughput))
        
        # Calculate average load
        if self.worker_stats:
            avg_load = statistics.mean(stats.current_load for stats in self.worker_stats.values())
            self.load_history.append((current_time, avg_load))
    
    def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        if not self.enable_auto_scaling:
            return
        
        # Simple auto-scaling logic
        queue_size = self.task_queue.qsize()
        avg_load = statistics.mean(stats.current_load for stats in self.worker_stats.values()) if self.worker_stats else 0
        
        # Scale up if queue is growing or workers are overloaded
        if queue_size > self.active_workers * 2 or avg_load > 0.8:
            self._scale_up()
        
        # Scale down if queue is empty and workers are underloaded
        elif queue_size == 0 and avg_load < 0.2 and self.active_workers > 1:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up workers."""
        if self.active_workers < self.max_workers:
            self.active_workers += 1
            print(f"Scaled up to {self.active_workers} workers")
    
    def _scale_down(self):
        """Scale down workers."""
        if self.active_workers > 1:
            self.active_workers -= 1
            print(f"Scaled down to {self.active_workers} workers")
    
    def _perform_health_checks(self):
        """Perform health checks on workers."""
        current_time = time.time()
        
        for worker_id, stats in self.worker_stats.items():
            # Check if worker is responsive
            if (stats.last_task_time and 
                current_time - stats.last_task_time > 300):  # 5 minutes
                # Worker may be stuck or idle
                pass
            
            # Update current load estimate
            recent_tasks = [t for t in self.task_history 
                           if (t.get('worker_id') == worker_id and 
                               current_time - t['completed_at'] <= 60)]
            
            if recent_tasks:
                avg_duration = statistics.mean(t['duration'] for t in recent_tasks)
                # Simple load estimate based on task frequency and duration
                stats.current_load = min(1.0, len(recent_tasks) * avg_duration / 60.0)
            else:
                stats.current_load = 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        current_time = time.time()
        
        # Calculate metrics
        total_tasks = self.total_tasks_processed
        avg_duration = self.total_processing_time / total_tasks if total_tasks > 0 else 0
        
        # Recent performance
        recent_tasks = [t for t in self.task_history 
                       if current_time - t['completed_at'] <= 300]  # Last 5 minutes
        
        recent_throughput = len(recent_tasks) / 300.0 if recent_tasks else 0
        recent_success_rate = (sum(1 for t in recent_tasks if t['success']) / 
                              len(recent_tasks)) if recent_tasks else 1.0
        
        # Worker stats
        worker_summary = {
            'total_workers': len(self.worker_stats),
            'active_workers': self.active_workers,
            'avg_load': statistics.mean(s.current_load for s in self.worker_stats.values()) if self.worker_stats else 0,
            'avg_success_rate': statistics.mean(s.success_rate for s in self.worker_stats.values()) if self.worker_stats else 1.0,
        }
        
        return {
            'total_tasks_processed': total_tasks,
            'pending_tasks': len(self.pending_tasks),
            'queue_size': self.task_queue.qsize(),
            'average_task_duration': avg_duration,
            'recent_throughput_per_sec': recent_throughput,
            'recent_success_rate': recent_success_rate,
            'worker_summary': worker_summary,
            'processing_mode': self.processing_mode.value,
            'load_balancing': self.load_balancing.value,
            'auto_scaling_enabled': self.enable_auto_scaling,
        }
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed worker statistics."""
        return {worker_id: {
            'tasks_completed': stats.tasks_completed,
            'tasks_failed': stats.tasks_failed,
            'total_processing_time': stats.total_processing_time,
            'current_load': stats.current_load,
            'error_rate': stats.error_rate,
            'success_rate': stats.success_rate,
            'average_task_duration': stats.average_task_duration,
            'queue_size': stats.queue_size,
        } for worker_id, stats in self.worker_stats.items()}


# Global processor instance
_global_processor: Optional[DistributedProcessor] = None


def get_distributed_processor(
    max_workers: Optional[int] = None,
    **kwargs
) -> DistributedProcessor:
    """Get global distributed processor instance."""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = DistributedProcessor(
            max_workers=max_workers,
            **kwargs
        )
        _global_processor.start()
    
    return _global_processor


def shutdown_distributed_processor():
    """Shutdown global distributed processor."""
    global _global_processor
    
    if _global_processor:
        _global_processor.stop()
        _global_processor = None


# Convenience functions
def process_parallel(
    tasks: List[Tuple[Callable, Tuple, Dict]],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None
) -> List[ProcessingResult]:
    """Process multiple tasks in parallel.
    
    Args:
        tasks: List of (function, args, kwargs) tuples
        max_workers: Maximum workers to use
        timeout: Overall timeout
        
    Returns:
        List of processing results
    """
    processor = get_distributed_processor(max_workers=max_workers)
    
    # Submit all tasks
    task_ids = []
    for func, args, kwargs in tasks:
        task_id = processor.submit_task(func, *args, **kwargs)
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        try:
            result = processor.get_result(task_id, timeout=timeout)
            results.append(result)
        except Exception as e:
            results.append(ProcessingResult(
                task_id=task_id,
                success=False,
                error=str(e)
            ))
    
    return results


async def process_parallel_async(
    tasks: List[Tuple[Callable, Tuple, Dict]],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None
) -> List[ProcessingResult]:
    """Process multiple tasks in parallel asynchronously."""
    processor = get_distributed_processor(max_workers=max_workers)
    
    # Submit all tasks
    task_ids = []
    for func, args, kwargs in tasks:
        task_id = await processor.submit_task_async(func, *args, **kwargs)
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        try:
            result = await processor.get_result_async(task_id, timeout=timeout)
            results.append(result)
        except Exception as e:
            results.append(ProcessingResult(
                task_id=task_id,
                success=False,
                error=str(e)
            ))
    
    return results