"""Advanced performance optimization and caching system for Fleet-Mind."""

import asyncio
import time
import threading
import functools
import weakref
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import json
import pickle
import hashlib


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, system metrics collection limited")


T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class PriorityLevel(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    function_name: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_execution_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, execution_time_ms: float, error: bool = False, cache_hit: bool = False):
        """Update metrics with new execution data."""
        self.execution_count += 1
        self.last_execution_time = time.time()
        
        if error:
            self.error_count += 1
            return
        
        self.total_time_ms += execution_time_ms
        self.average_time_ms = self.total_time_ms / self.execution_count
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Record cache access."""
        self.access_count += 1


class AdaptiveCache:
    """High-performance adaptive cache with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default_ttl: Optional[float] = None,
        max_memory_mb: Optional[float] = None
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cache entries
            strategy: Cache eviction strategy
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._access_order: Dict[str, float] = {}
        self._frequency: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive strategy parameters
        self._hit_rate_window = 100
        self._recent_hits = []
        self._strategy_performance = {
            CacheStrategy.LRU: 0.0,
            CacheStrategy.LFU: 0.0,
            CacheStrategy.TTL: 0.0,
        }
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments."""
        try:
            # Create a hashable representation
            key_data = {
                'func': func_name,
                'args': args,
                'kwargs': sorted(kwargs.items()) if kwargs else ()
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return f"{func_name}_{id(args)}_{id(kwargs)}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._frequency.pop(key, None)
                self._access_order.pop(key, None)
                self.misses += 1
                return None
            
            # Update access patterns
            entry.access()
            self._access_order[key] = time.time()
            self._frequency[key] += 1
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            self.hits += 1
            self._track_hit_rate(True)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except (TypeError, pickle.PicklingError):
                size_bytes = 1024  # Estimate for non-serializable objects
            
            # Check memory limit
            if self.max_memory_bytes and self._get_total_size() + size_bytes > self.max_memory_bytes:
                self._evict_for_memory(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Add to cache
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                
                # Evict if necessary
                if len(self._cache) > self.max_size:
                    self._evict()
            
            # Update tracking
            self._access_order[key] = time.time()
            self._frequency[key] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'strategy': self.strategy.value,
                'memory_usage_bytes': self._get_total_size(),
                'avg_entry_size_bytes': self._get_average_entry_size(),
            }
    
    def _evict(self) -> None:
        """Evict cache entry based on strategy."""
        if not self._cache:
            return
        
        strategy = self.strategy
        if strategy == CacheStrategy.ADAPTIVE:
            strategy = self._choose_adaptive_strategy()
        
        if strategy == CacheStrategy.LRU:
            # Remove least recently used
            key = next(iter(self._cache))
        elif strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._frequency.keys(), key=lambda k: self._frequency[k])
        elif strategy == CacheStrategy.TTL:
            # Remove expired entries first, then LRU
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = next(iter(self._cache))
        else:
            key = next(iter(self._cache))
        
        # Remove the entry
        del self._cache[key]
        self._frequency.pop(key, None)
        self._access_order.pop(key, None)
        self.evictions += 1
    
    def _evict_for_memory(self, needed_bytes: int) -> None:
        """Evict entries to free up memory."""
        while (self._get_total_size() + needed_bytes > self.max_memory_bytes and 
               self._cache):
            self._evict()
    
    def _get_total_size(self) -> int:
        """Get total cache memory usage."""
        return sum(entry.size_bytes for entry in self._cache.values())
    
    def _get_average_entry_size(self) -> float:
        """Get average entry size."""
        if not self._cache:
            return 0.0
        return self._get_total_size() / len(self._cache)
    
    def _track_hit_rate(self, hit: bool) -> None:
        """Track hit rate for adaptive strategy."""
        self._recent_hits.append(hit)
        if len(self._recent_hits) > self._hit_rate_window:
            self._recent_hits.pop(0)
    
    def _choose_adaptive_strategy(self) -> CacheStrategy:
        """Choose best strategy based on performance."""
        if len(self._recent_hits) < 10:
            return CacheStrategy.LRU  # Default
        
        hit_rate = sum(self._recent_hits) / len(self._recent_hits)
        
        # Simple heuristic: choose strategy based on hit rate
        if hit_rate > 0.8:
            return CacheStrategy.LRU
        elif hit_rate > 0.5:
            return CacheStrategy.LFU
        else:
            return CacheStrategy.TTL


class ResourcePool:
    """High-performance resource pool with auto-scaling."""
    
    def __init__(
        self,
        resource_factory: Callable[[], T],
        min_size: int = 5,
        max_size: int = 50,
        idle_timeout: float = 300.0,
        resource_validator: Optional[Callable[[T], bool]] = None
    ):
        """Initialize resource pool.
        
        Args:
            resource_factory: Function to create new resources
            min_size: Minimum pool size
            max_size: Maximum pool size
            idle_timeout: Timeout for idle resources in seconds
            resource_validator: Function to validate resource health
        """
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.resource_validator = resource_validator
        
        self._pool: List[Tuple[T, float]] = []  # (resource, last_used)
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.RLock()
        self._created_count = 0
        self._destroyed_count = 0
        
        # Initialize minimum pool
        self._fill_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """Acquire resource from pool."""
        start_time = time.time()
        
        while True:
            with self._lock:
                # Try to get from pool
                while self._pool:
                    resource, last_used = self._pool.pop(0)
                    
                    # Validate resource
                    if self._is_resource_valid(resource):
                        self._in_use.add(resource)
                        return resource
                    else:
                        self._destroy_resource(resource)
                
                # Create new resource if under limit
                if len(self._in_use) < self.max_size:
                    resource = self._create_resource()
                    if resource:
                        self._in_use.add(resource)
                        return resource
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return None
            
            # Wait a bit before retrying
            time.sleep(0.01)
    
    def release(self, resource: T) -> None:
        """Release resource back to pool."""
        with self._lock:
            if resource in self._in_use:
                self._in_use.discard(resource)
                
                if self._is_resource_valid(resource):
                    self._pool.append((resource, time.time()))
                else:
                    self._destroy_resource(resource)
                    
                # Maintain minimum pool size
                self._fill_pool()
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'created_count': self._created_count,
                'destroyed_count': self._destroyed_count,
                'utilization': len(self._in_use) / self.max_size,
            }
    
    def _create_resource(self) -> Optional[T]:
        """Create new resource."""
        try:
            resource = self.resource_factory()
            self._created_count += 1
            return resource
        except Exception as e:
            print(f"Failed to create resource: {e}")
            return None
    
    def _destroy_resource(self, resource: T) -> None:
        """Destroy resource."""
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, '__del__'):
                resource.__del__()
            self._destroyed_count += 1
        except Exception as e:
            print(f"Failed to destroy resource: {e}")
    
    def _is_resource_valid(self, resource: T) -> bool:
        """Check if resource is valid."""
        try:
            if self.resource_validator:
                return self.resource_validator(resource)
            return True
        except Exception:
            return False
    
    def _fill_pool(self) -> None:
        """Fill pool to minimum size."""
        while len(self._pool) + len(self._in_use) < self.min_size:
            resource = self._create_resource()
            if resource:
                self._pool.append((resource, time.time()))
            else:
                break
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self._lock:
                    current_time = time.time()
                    active_pool = []
                    
                    for resource, last_used in self._pool:
                        if current_time - last_used > self.idle_timeout:
                            # Resource is idle, destroy it
                            self._destroy_resource(resource)
                        else:
                            active_pool.append((resource, last_used))
                    
                    self._pool = active_pool
                    
                    # Maintain minimum pool size
                    self._fill_pool()
                    
            except Exception as e:
                print(f"Error in resource pool cleanup: {e}")


class TaskScheduler:
    """Advanced task scheduler with priority and load balancing."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        enable_profiling: bool = True
    ):
        """Initialize task scheduler.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            enable_profiling: Enable performance profiling
        """
        if max_workers is None:
            max_workers = (psutil.cpu_count() if PSUTIL_AVAILABLE else 4)
        
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.enable_profiling = enable_profiling
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task queues by priority
        self.task_queues = {
            priority: asyncio.Queue() for priority in PriorityLevel
        }
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._task_counter = 0
        
        # Load balancing
        self._worker_loads: Dict[int, float] = defaultdict(float)
        
        # Start scheduler loop
        self._scheduler_task = None
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: PriorityLevel = PriorityLevel.NORMAL,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Submit task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority level
            timeout: Task timeout in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        task_id = self._task_counter
        self._task_counter += 1
        
        # Create task info
        task_info = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'timeout': timeout,
            'submitted_at': time.time(),
        }
        
        # Add to appropriate queue
        await self.task_queues[priority].put(task_info)
        
        # Execute task
        if self.use_processes:
            future = self.executor.submit(self._execute_task_sync, task_info)
        else:
            future = self.executor.submit(self._execute_task_sync, task_info)
        
        # Wait for result with timeout
        try:
            if timeout:
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=timeout
                )
            else:
                result = await asyncio.wrap_future(future)
            
            return result
            
        except asyncio.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
    
    def submit_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        priority: PriorityLevel = PriorityLevel.NORMAL,
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Submit batch of tasks for concurrent execution.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            priority: Priority level for all tasks
            max_concurrent: Maximum concurrent tasks
            
        Returns:
            List of results in same order as input
        """
        if max_concurrent is None:
            max_concurrent = min(len(tasks), self.max_workers)
        
        # Submit all tasks
        futures = []
        for func, args, kwargs in tasks:
            future = self.executor.submit(func, *args, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def _execute_task_sync(self, task_info: dict) -> Any:
        """Execute task synchronously with profiling."""
        func = task_info['func']
        args = task_info['args']
        kwargs = task_info['kwargs']
        
        start_time = time.time()
        func_name = getattr(func, '__name__', str(func))
        
        try:
            result = func(*args, **kwargs)
            
            # Update metrics
            if self.enable_profiling:
                execution_time = (time.time() - start_time) * 1000
                self._update_metrics(func_name, execution_time, False)
            
            return result
            
        except Exception as e:
            # Update error metrics
            if self.enable_profiling:
                execution_time = (time.time() - start_time) * 1000
                self._update_metrics(func_name, execution_time, True)
            
            raise e
    
    def _update_metrics(self, func_name: str, execution_time: float, error: bool) -> None:
        """Update performance metrics."""
        if func_name not in self.metrics:
            self.metrics[func_name] = PerformanceMetrics(func_name)
        
        self.metrics[func_name].update(execution_time, error)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all functions."""
        return {
            name: {
                'execution_count': metrics.execution_count,
                'average_time_ms': metrics.average_time_ms,
                'min_time_ms': metrics.min_time_ms,
                'max_time_ms': metrics.max_time_ms,
                'error_count': metrics.error_count,
                'error_rate': metrics.error_count / max(1, metrics.execution_count),
            }
            for name, metrics in self.metrics.items()
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown task scheduler."""
        self.executor.shutdown(wait=wait)


# Global instances (lazy-initialized)
_global_cache = None
_global_scheduler = None

def _get_global_cache():
    """Get or create global cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdaptiveCache(max_size=10000, strategy=CacheStrategy.ADAPTIVE)
    return _global_cache

def _get_global_scheduler():
    """Get or create global scheduler."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = TaskScheduler(enable_profiling=True)
    return _global_scheduler


def cached(
    ttl: Optional[float] = None,
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    max_size: int = 1000
) -> Callable:
    """Decorator for function caching with advanced strategies.
    
    Args:
        ttl: Time-to-live in seconds
        strategy: Cache strategy
        max_size: Maximum cache size for this function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Create function-specific cache
        cache = AdaptiveCache(max_size=max_size, strategy=strategy, default_ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Store in cache
                cache.put(cache_key, result, ttl)
                
                # Update global metrics
                func_name = func.__name__
                scheduler = _get_global_scheduler()
                if func_name not in scheduler.metrics:
                    scheduler.metrics[func_name] = PerformanceMetrics(func_name)
                
                scheduler.metrics[func_name].update(execution_time, False, False)
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Update error metrics
                func_name = func.__name__
                scheduler = _get_global_scheduler()
                if func_name not in scheduler.metrics:
                    scheduler.metrics[func_name] = PerformanceMetrics(func_name)
                
                scheduler.metrics[func_name].update(execution_time, True, False)
                
                raise e
        
        # Add cache stats method
        wrapper.cache_stats = cache.stats
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator


def async_cached(
    ttl: Optional[float] = None,
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
    max_size: int = 1000
) -> Callable:
    """Decorator for async function caching.
    
    Args:
        ttl: Time-to-live in seconds
        strategy: Cache strategy
        max_size: Maximum cache size
        
    Returns:
        Decorated async function
    """
    def decorator(func: Callable) -> Callable:
        cache = AdaptiveCache(max_size=max_size, strategy=strategy, default_ttl=ttl)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # Store in cache
                cache.put(cache_key, result, ttl)
                
                return result
                
            except Exception as e:
                raise e
        
        wrapper.cache_stats = cache.stats
        wrapper.clear_cache = cache.clear
        
        return wrapper
    
    return decorator


def performance_monitor(func: Callable) -> Callable:
    """Decorator for performance monitoring.
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Update metrics
            scheduler = _get_global_scheduler()
            if func_name not in scheduler.metrics:
                scheduler.metrics[func_name] = PerformanceMetrics(func_name)
            
            scheduler.metrics[func_name].update(execution_time, False)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Update error metrics
            scheduler = _get_global_scheduler()
            if func_name not in scheduler.metrics:
                scheduler.metrics[func_name] = PerformanceMetrics(func_name)
            
            scheduler.metrics[func_name].update(execution_time, True)
            
            raise e
    
    return wrapper


async def parallel_execute(
    tasks: List[Callable],
    max_concurrent: Optional[int] = None,
    timeout: Optional[float] = None
) -> List[Any]:
    """Execute multiple tasks in parallel with concurrency control.
    
    Args:
        tasks: List of async callables or sync callables
        max_concurrent: Maximum concurrent tasks
        timeout: Overall timeout for all tasks
        
    Returns:
        List of results in same order as input
    """
    if max_concurrent is None:
        max_concurrent = min(len(tasks), 10)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_task(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, task)
    
    # Create task coroutines
    coroutines = [execute_task(task) for task in tasks]
    
    # Execute with timeout
    if timeout:
        results = await asyncio.wait_for(
            asyncio.gather(*coroutines, return_exceptions=True),
            timeout=timeout
        )
    else:
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    return results


def get_system_metrics() -> Dict[str, Any]:
    """Get comprehensive system performance metrics."""
    metrics = {
        'timestamp': time.time(),
        'cpu_available': True,
        'memory_available': True,
        'disk_available': True,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics.update({
                'cpu_percent': cpu_percent,
                'cpu_count_logical': cpu_count,
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.update({
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics.update({
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
            })
            
            # Network metrics
            net_io = psutil.net_io_counters()
            if net_io:
                metrics.update({
                    'network_bytes_sent': net_io.bytes_sent,
                    'network_bytes_recv': net_io.bytes_recv,
                    'network_packets_sent': net_io.packets_sent,
                    'network_packets_recv': net_io.packets_recv,
                })
                
        except Exception as e:
            metrics['psutil_error'] = str(e)
    else:
        # Fallback metrics
        metrics.update({
            'cpu_percent': 50.0,  # Assume 50% usage
            'memory_percent': 60.0,  # Assume 60% usage
            'disk_percent': 70.0,  # Assume 70% usage
        })
    
    return metrics


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    try:
        cache = _get_global_cache()
        scheduler = _get_global_scheduler()
        cache_stats = cache.stats()
        scheduler_stats = scheduler.get_metrics()
    except Exception:
        cache_stats = {'error': 'Cache not initialized'}
        scheduler_stats = {'error': 'Scheduler not initialized'}
    
    return {
        'cache_stats': cache_stats,
        'scheduler_stats': scheduler_stats,
        'system_metrics': get_system_metrics(),
        'timestamp': time.time(),
    }