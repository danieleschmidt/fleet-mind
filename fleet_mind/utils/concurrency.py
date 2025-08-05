"""Advanced concurrency utilities for Fleet-Mind with auto-scaling and load balancing."""

import asyncio
import threading
import multiprocessing
import time
import weakref
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from enum import Enum
import uuid
import queue
import signal
import os


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


T = TypeVar('T')


class WorkerType(Enum):
    """Worker execution types."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CPU_AWARE = "cpu_aware"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerStats:
    """Statistics for individual workers."""
    worker_id: str
    worker_type: WorkerType
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    current_load: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_activity: float = 0.0
    is_healthy: bool = True
    created_at: float = field(default_factory=time.time)
    
    def update_stats(self, execution_time: float, success: bool = True):
        """Update worker statistics."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time += execution_time
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.average_response_time = self.total_execution_time / total_tasks
        
        self.last_activity = time.time()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 1.0
    
    @property
    def uptime(self) -> float:
        """Calculate worker uptime."""
        return time.time() - self.created_at


@dataclass
class TaskInfo:
    """Information about a task."""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    result: Any = None
    error: Optional[Exception] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def wait_time(self) -> Optional[float]:
        """Get time spent waiting in queue."""
        if self.started_at:
            return self.started_at - self.submitted_at
        return None


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Optional[type] = None
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to close circuit
            expected_exception: Expected exception type to track
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception or Exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker."""
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == 'OPEN':
                    if time.time() - self.last_failure_time < self.recovery_timeout:
                        raise Exception("Circuit breaker is OPEN")
                    else:
                        self.state = 'HALF_OPEN'
                
                try:
                    result = func(*args, **kwargs)
                    
                    if self.state == 'HALF_OPEN':
                        self.state = 'CLOSED'
                        self.failure_count = 0
                    
                    return result
                    
                except self.expected_exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'OPEN'
                    
                    raise e
        
        return wrapper


class AdaptiveWorkerPool:
    """Auto-scaling worker pool with intelligent load balancing."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = 20,
        worker_type: WorkerType = WorkerType.THREAD,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        monitoring_interval: float = 10.0
    ):
        """Initialize adaptive worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            worker_type: Type of workers to create
            load_balancing: Load balancing strategy
            scale_up_threshold: CPU threshold to scale up
            scale_down_threshold: CPU threshold to scale down
            monitoring_interval: Monitoring interval in seconds
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_type = worker_type
        self.load_balancing = load_balancing
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitoring_interval = monitoring_interval
        
        # Worker management
        self.workers: Dict[str, Any] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.task_queue = asyncio.Queue()
        self.result_futures: Dict[str, asyncio.Future] = {}
        
        # Load balancing state
        self._round_robin_index = 0
        self._load_history = deque(maxlen=100)
        
        # Executors for different worker types
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        # Control flags
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Circuit breakers for workers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the worker pool."""
        self._running = True
        
        # Create initial workers
        for _ in range(self.min_workers):
            self._add_worker()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Task distribution will be started on first use
        self._task_distribution_started = False
    
    def _add_worker(self) -> str:
        """Add a new worker to the pool."""
        worker_id = str(uuid.uuid4())
        
        if self.worker_type == WorkerType.THREAD:
            if self.thread_executor is None:
                self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
            worker = self.thread_executor
            
        elif self.worker_type == WorkerType.PROCESS:
            if self.process_executor is None:
                self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
            worker = self.process_executor
            
        elif self.worker_type == WorkerType.ASYNC:
            # Async workers are managed differently
            worker = None
            
        else:  # HYBRID
            # Create both thread and process executors
            if self.thread_executor is None:
                self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers // 2)
            if self.process_executor is None:
                self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers // 2)
            worker = (self.thread_executor, self.process_executor)
        
        self.workers[worker_id] = worker
        self.worker_stats[worker_id] = WorkerStats(worker_id, self.worker_type)
        self.circuit_breakers[worker_id] = CircuitBreaker()
        
        return worker_id
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker from the pool."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_stats[worker_id]
            del self.circuit_breakers[worker_id]
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        task_id = str(uuid.uuid4())
        
        task_info = TaskInfo(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # Create future for result
        future = asyncio.Future()
        self.result_futures[task_id] = future
        
        # Start task distribution on first use
        if not self._task_distribution_started and self.worker_type in [WorkerType.ASYNC, WorkerType.HYBRID]:
            try:
                asyncio.create_task(self._task_distribution_loop())
                self._task_distribution_started = True
            except RuntimeError:
                # No event loop, will be started later
                pass
        
        # Add to queue
        await self.task_queue.put(task_info)
        
        # Wait for result
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            
            return result
            
        finally:
            # Clean up
            self.result_futures.pop(task_id, None)
    
    async def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Submit a batch of tasks.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            priority: Priority for all tasks
            timeout: Timeout for all tasks
            
        Returns:
            List of results in same order as input
        """
        # Submit all tasks
        futures = []
        for func, args, kwargs in tasks:
            future = asyncio.create_task(
                self.submit_task(func, *args, priority=priority, timeout=timeout, **kwargs)
            )
            futures.append(future)
        
        # Wait for all results
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*futures, return_exceptions=True)
        
        return results
    
    def _select_worker(self) -> Optional[str]:
        """Select the best worker using load balancing strategy."""
        if not self.workers:
            return None
        
        if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            worker_ids = list(self.workers.keys())
            worker_id = worker_ids[self._round_robin_index % len(worker_ids)]
            self._round_robin_index += 1
            return worker_id
            
        elif self.load_balancing == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(
                self.workers.keys(),
                key=lambda w: self.worker_stats[w].current_load
            )
            
        elif self.load_balancing == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(
                self.workers.keys(),
                key=lambda w: self.worker_stats[w].average_response_time
            )
            
        elif self.load_balancing == LoadBalancingStrategy.CPU_AWARE:
            if PSUTIL_AVAILABLE:
                return min(
                    self.workers.keys(),
                    key=lambda w: self.worker_stats[w].cpu_usage
                )
            else:
                # Fallback to least connections
                return min(
                    self.workers.keys(),
                    key=lambda w: self.worker_stats[w].current_load
                )
                
        else:  # ADAPTIVE
            # Use a combination of factors
            def score(worker_id: str) -> float:
                stats = self.worker_stats[worker_id]
                return (
                    stats.current_load * 0.4 +
                    stats.average_response_time * 0.3 +
                    (1.0 - stats.success_rate) * 0.2 +
                    stats.cpu_usage * 0.1
                )
            
            return min(self.workers.keys(), key=score)
    
    async def _task_distribution_loop(self):
        """Main task distribution loop."""
        while self._running:
            try:
                # Get task from queue
                task_info = await self.task_queue.get()
                
                # Select worker
                worker_id = self._select_worker()
                if not worker_id:
                    # No workers available, put task back
                    await self.task_queue.put(task_info)
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                asyncio.create_task(self._execute_task(task_info, worker_id))
                
            except Exception as e:
                print(f"Error in task distribution loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task_info: TaskInfo, worker_id: str):
        """Execute a task on the specified worker."""
        task_info.started_at = time.time()
        task_info.worker_id = worker_id
        
        # Update worker load
        self.worker_stats[worker_id].current_load += 1
        
        try:
            # Apply circuit breaker
            circuit_breaker = self.circuit_breakers[worker_id]
            
            if circuit_breaker.state == 'OPEN':
                raise Exception(f"Circuit breaker is OPEN for worker {worker_id}")
            
            # Execute based on worker type
            if self.worker_type == WorkerType.THREAD:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_executor,
                    task_info.func,
                    *task_info.args,
                    **task_info.kwargs
                )
                
            elif self.worker_type == WorkerType.PROCESS:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_executor,
                    task_info.func,
                    *task_info.args,
                    **task_info.kwargs
                )
                
            elif self.worker_type == WorkerType.ASYNC:
                if asyncio.iscoroutinefunction(task_info.func):
                    result = await task_info.func(*task_info.args, **task_info.kwargs)
                else:
                    result = task_info.func(*task_info.args, **task_info.kwargs)
                    
            else:  # HYBRID
                # Decide between thread and process based on task characteristics
                if self._should_use_process(task_info):
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.process_executor,
                        task_info.func,
                        *task_info.args,
                        **task_info.kwargs
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.thread_executor,
                        task_info.func,
                        *task_info.args,
                        **task_info.kwargs
                    )
            
            # Task completed successfully
            task_info.completed_at = time.time()
            task_info.result = result
            
            # Update statistics
            execution_time = task_info.duration
            self.worker_stats[worker_id].update_stats(execution_time, True)
            
            # Set result
            future = self.result_futures.get(task_info.task_id)
            if future and not future.done():
                future.set_result(result)
                
        except Exception as e:
            # Task failed
            task_info.completed_at = time.time()
            task_info.error = e
            
            # Update statistics
            execution_time = task_info.duration or 0
            self.worker_stats[worker_id].update_stats(execution_time, False)
            
            # Handle retries
            if task_info.retries < task_info.max_retries:
                task_info.retries += 1
                task_info.started_at = None
                task_info.completed_at = None
                await self.task_queue.put(task_info)
            else:
                # Set error
                future = self.result_futures.get(task_info.task_id)
                if future and not future.done():
                    future.set_exception(e)
        
        finally:
            # Update worker load
            self.worker_stats[worker_id].current_load -= 1
    
    def _should_use_process(self, task_info: TaskInfo) -> bool:
        """Decide whether to use process executor for hybrid mode."""
        # Simple heuristic: use processes for CPU-intensive tasks
        # In practice, this could be more sophisticated
        func_name = getattr(task_info.func, '__name__', 'unknown')
        
        # Use processes for known CPU-intensive operations
        cpu_intensive_patterns = ['compute', 'calculate', 'process', 'analyze', 'encode']
        return any(pattern in func_name.lower() for pattern in cpu_intensive_patterns)
    
    def _monitor_loop(self):
        """Background monitoring and auto-scaling loop."""
        while not self._shutdown_event.wait(self.monitoring_interval):
            try:
                self._update_worker_metrics()
                self._auto_scale()
                self._health_check()
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
    
    def _update_worker_metrics(self):
        """Update worker performance metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Update system metrics for each worker
            for worker_id, stats in self.worker_stats.items():
                # For simplicity, use system-wide metrics
                # In practice, you'd track per-worker metrics
                stats.cpu_usage = psutil.cpu_percent()
                stats.memory_usage = psutil.virtual_memory().percent
                
        except Exception as e:
            print(f"Error updating worker metrics: {e}")
    
    def _auto_scale(self):
        """Auto-scale the worker pool based on load."""
        try:
            current_workers = len(self.workers)
            
            # Calculate average load
            if self.worker_stats:
                avg_load = sum(
                    stats.current_load for stats in self.worker_stats.values()
                ) / len(self.worker_stats)
                
                avg_cpu = sum(
                    stats.cpu_usage for stats in self.worker_stats.values()
                ) / len(self.worker_stats)
                
                # Scale up if high load
                if (avg_load > self.scale_up_threshold or avg_cpu > 80) and current_workers < self.max_workers:
                    self._add_worker()
                    print(f"Scaled up to {len(self.workers)} workers")
                
                # Scale down if low load
                elif (avg_load < self.scale_down_threshold and avg_cpu < 30) and current_workers > self.min_workers:
                    # Remove least utilized worker
                    worker_id = min(
                        self.workers.keys(),
                        key=lambda w: self.worker_stats[w].current_load
                    )
                    self._remove_worker(worker_id)
                    print(f"Scaled down to {len(self.workers)} workers")
                
        except Exception as e:
            print(f"Error in auto-scaling: {e}")
    
    def _health_check(self):
        """Perform health checks on workers."""
        try:
            current_time = time.time()
            unhealthy_workers = []
            
            for worker_id, stats in self.worker_stats.items():
                # Check if worker is responsive
                if current_time - stats.last_activity > 300:  # 5 minutes
                    stats.is_healthy = False
                    unhealthy_workers.append(worker_id)
                
                # Check success rate
                if stats.success_rate < 0.5 and stats.tasks_completed > 10:
                    stats.is_healthy = False
                    unhealthy_workers.append(worker_id)
            
            # Remove unhealthy workers
            for worker_id in unhealthy_workers:
                print(f"Removing unhealthy worker: {worker_id}")
                self._remove_worker(worker_id)
                
                # Add replacement if needed
                if len(self.workers) < self.min_workers:
                    self._add_worker()
                    
        except Exception as e:
            print(f"Error in health check: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        total_tasks = sum(stats.tasks_completed + stats.tasks_failed for stats in self.worker_stats.values())
        total_success = sum(stats.tasks_completed for stats in self.worker_stats.values())
        
        return {
            'worker_count': len(self.workers),
            'worker_type': self.worker_type.value,
            'load_balancing': self.load_balancing.value,
            'total_tasks': total_tasks,
            'success_rate': total_success / total_tasks if total_tasks > 0 else 1.0,
            'average_response_time': sum(
                stats.average_response_time for stats in self.worker_stats.values()
            ) / len(self.worker_stats) if self.worker_stats else 0,
            'queue_size': self.task_queue.qsize(),
            'worker_stats': {
                worker_id: {
                    'tasks_completed': stats.tasks_completed,
                    'tasks_failed': stats.tasks_failed,
                    'success_rate': stats.success_rate,
                    'average_response_time': stats.average_response_time,
                    'current_load': stats.current_load,
                    'cpu_usage': stats.cpu_usage,
                    'memory_usage': stats.memory_usage,
                    'uptime': stats.uptime,
                    'is_healthy': stats.is_healthy,
                }
                for worker_id, stats in self.worker_stats.items()
            }
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        self._running = False
        self._shutdown_event.set()
        
        # Shutdown executors
        if self.thread_executor:
            self.thread_executor.shutdown(wait=wait)
        if self.process_executor:
            self.process_executor.shutdown(wait=wait)
        
        # Cancel remaining futures
        for future in self.result_futures.values():
            if not future.done():
                future.cancel()


# Global adaptive pool instance (lazy-initialized)
_global_worker_pool = None

def _get_global_worker_pool():
    """Get or create global worker pool."""
    global _global_worker_pool
    if _global_worker_pool is None:
        _global_worker_pool = AdaptiveWorkerPool(
            min_workers=2,
            max_workers=20,
            worker_type=WorkerType.HYBRID,
            load_balancing=LoadBalancingStrategy.ADAPTIVE
        )
    return _global_worker_pool


async def execute_concurrent(
    func: Callable,
    items: List[Any],
    max_concurrent: int = 10,
    timeout: Optional[float] = None,
    priority: int = 0
) -> List[Any]:
    """Execute function concurrently on multiple items.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_concurrent: Maximum concurrent executions
        timeout: Timeout per task
        priority: Task priority
        
    Returns:
        List of results in same order as input
    """
    # Create tasks
    tasks = [(func, (item,), {}) for item in items]
    
    # Execute using global pool
    pool = _get_global_worker_pool()
    return await pool.submit_batch(
        tasks,
        priority=priority,
        timeout=timeout
    )


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    use_processes: bool = False
) -> List[Any]:
    """Parallel map using thread or process pool.
    
    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum number of workers
        use_processes: Use processes instead of threads
        
    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = min(len(items), (psutil.cpu_count() if PSUTIL_AVAILABLE else 4))
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results = [future.result() for future in as_completed(futures)]
    
    return results


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int = 1):
        """Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.last_update = now
                
                # Add tokens based on elapsed time
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return False
            
            # Wait a bit
            time.sleep(0.01)
    
    async def acquire_async(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Async version of acquire."""
        start_time = time.time()
        
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.last_update = now
                
                # Add tokens based on elapsed time
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return False
            
            # Wait a bit
            await asyncio.sleep(0.01)


def rate_limited(rate: float, burst: int = 1):
    """Decorator for rate limiting function calls.
    
    Args:
        rate: Calls per second
        burst: Maximum burst size
        
    Returns:
        Decorated function
    """
    limiter = RateLimiter(rate, burst)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if limiter.acquire():
                return func(*args, **kwargs)
            else:
                raise Exception("Rate limit exceeded")
        
        return wrapper
    
    return decorator


def get_concurrency_stats() -> Dict[str, Any]:
    """Get comprehensive concurrency statistics."""
    try:
        pool = _get_global_worker_pool()
        pool_stats = pool.get_stats()
    except Exception:
        pool_stats = {'error': 'Pool not initialized'}
    
    return {
        'global_pool_stats': pool_stats,
        'system_resources': {
            'cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 4,
            'cpu_percent': psutil.cpu_percent() if PSUTIL_AVAILABLE else 50,
            'memory_percent': psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 60,
        },
        'timestamp': time.time(),
    }