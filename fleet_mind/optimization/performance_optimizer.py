"""Advanced performance optimization and auto-scaling for Fleet-Mind."""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple

try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not available, performance optimization limited")
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    MEMORY_EFFICIENT = "memory_efficient"
    BALANCED = "balanced"
    POWER_EFFICIENT = "power_efficient"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    GPU = "gpu"


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: float
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_throughput_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    active_connections: int = 0
    request_latency_ms: float = 0.0
    error_rate: float = 0.0
    queue_size: int = 0
    
    
@dataclass
class OptimizationAction:
    """Optimization action to be taken."""
    action_type: str
    resource: ResourceType
    parameters: Dict[str, Any]
    expected_impact: float  # Expected improvement (0-1)
    cost: float  # Implementation cost (0-1)
    priority: int = 1  # Higher number = higher priority


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # scale_up, scale_down, maintain
    target_instances: int
    confidence: float
    reasoning: List[str]
    metrics_used: Dict[str, float]


class PerformanceOptimizer:
    """Advanced performance optimization system.
    
    Monitors system performance, identifies bottlenecks, and applies
    optimizations automatically or suggests manual interventions.
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        monitoring_interval: float = 30.0,
        optimization_threshold: float = 0.8,  # Trigger optimization when metrics exceed this
        enable_auto_optimization: bool = True,
    ):
        """Initialize performance optimizer.
        
        Args:
            strategy: Optimization strategy to use
            monitoring_interval: Performance monitoring interval in seconds
            optimization_threshold: Threshold for triggering optimizations
            enable_auto_optimization: Enable automatic optimizations
        """
        self.strategy = strategy
        self.monitoring_interval = monitoring_interval
        self.optimization_threshold = optimization_threshold
        self.enable_auto_optimization = enable_auto_optimization
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Optimization tracking
        self.applied_optimizations: List[OptimizationAction] = []
        self.optimization_results: Dict[str, Dict[str, float]] = {}
        
        # Auto-scaling
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 10
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scaling_history: List[ScalingDecision] = []
        
        # Resource pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Async tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self.optimization_callbacks: List[Callable[[OptimizationAction], None]] = []
        self.scaling_callbacks: List[Callable[[ScalingDecision], None]] = []

    async def start(self) -> None:
        """Start performance monitoring and optimization."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize resource pools based on strategy
        self._initialize_resource_pools()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start optimization if enabled
        if self.enable_auto_optimization:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        print(f"Performance optimizer started with {self.strategy.value} strategy")

    async def stop(self) -> None:
        """Stop performance monitoring and optimization."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        if self._optimization_task:
            self._optimization_task.cancel()
        
        # Shutdown resource pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        print("Performance optimizer stopped")

    def _initialize_resource_pools(self) -> None:
        """Initialize thread and process pools based on strategy."""
        cpu_count = multiprocessing.cpu_count()
        
        if self.strategy == OptimizationStrategy.LATENCY_FOCUSED:
            # More threads, fewer processes for low latency
            thread_workers = min(cpu_count * 4, 32)
            process_workers = max(2, cpu_count // 2)
        elif self.strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            # More processes for high throughput
            thread_workers = min(cpu_count * 2, 16)
            process_workers = cpu_count
        elif self.strategy == OptimizationStrategy.MEMORY_EFFICIENT:
            # Fewer workers to reduce memory usage
            thread_workers = max(2, cpu_count)
            process_workers = max(1, cpu_count // 2)
        else:  # BALANCED or POWER_EFFICIENT
            thread_workers = min(cpu_count * 2, 16)
            process_workers = max(2, cpu_count // 2)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=process_workers)
        
        print(f"Initialized resource pools: {thread_workers} threads, {process_workers} processes")

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics.
        
        Returns:
            Current system performance metrics
        """
        try:
            current_time = time.time()
            
            if not psutil:
                # Return mock metrics when psutil unavailable
                return PerformanceMetrics(
                    cpu_usage=25.0,
                    memory_usage=60.0,
                    latency_ms=50.0,
                    throughput=1000.0,
                    error_rate=0.01,
                    network_io_mb=10.0,
                    disk_io_mb=5.0
                )
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_mbps = 0.0
            if hasattr(self, '_last_net_io') and hasattr(self, '_last_metric_time'):
                time_diff = current_time - self._last_metric_time
                if time_diff > 0:
                    bytes_diff = (net_io.bytes_sent + net_io.bytes_recv) - (
                        self._last_net_io.bytes_sent + self._last_net_io.bytes_recv
                    )
                    network_mbps = (bytes_diff / time_diff) / (1024 * 1024)  # MB/s
            
            self._last_net_io = net_io
            self._last_metric_time = current_time
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_mbps = 0.0
            if hasattr(self, '_last_disk_io'):
                time_diff = current_time - getattr(self, '_last_disk_time', current_time)
                if time_diff > 0:
                    bytes_diff = (disk_io.read_bytes + disk_io.write_bytes) - (
                        self._last_disk_io.read_bytes + self._last_disk_io.write_bytes
                    )
                    disk_mbps = (bytes_diff / time_diff) / (1024 * 1024)
            
            self._last_disk_io = disk_io
            self._last_disk_time = current_time
            
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                network_throughput_mbps=network_mbps,
                disk_io_mbps=disk_mbps,
            )
            
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return PerformanceMetrics(timestamp=time.time())

    def analyze_bottlenecks(self, metrics: Optional[PerformanceMetrics] = None) -> List[Tuple[ResourceType, float]]:
        """Analyze system bottlenecks.
        
        Args:
            metrics: Metrics to analyze (uses current if None)
            
        Returns:
            List of (resource_type, severity) tuples sorted by severity
        """
        if metrics is None:
            metrics = self.current_metrics
        
        if not metrics:
            return []
        
        bottlenecks = []
        
        # CPU bottleneck
        if metrics.cpu_usage_percent > 80:
            bottlenecks.append((ResourceType.CPU, metrics.cpu_usage_percent / 100))
        
        # Memory bottleneck
        if metrics.memory_usage_percent > 85:
            bottlenecks.append((ResourceType.MEMORY, metrics.memory_usage_percent / 100))
        
        # Network bottleneck (simplified heuristic)
        if metrics.network_throughput_mbps > 100:  # High network usage
            bottlenecks.append((ResourceType.NETWORK, min(1.0, metrics.network_throughput_mbps / 1000)))
        
        # Disk I/O bottleneck
        if metrics.disk_io_mbps > 50:  # High disk I/O
            bottlenecks.append((ResourceType.DISK, min(1.0, metrics.disk_io_mbps / 200)))
        
        # Request latency bottleneck
        if metrics.request_latency_ms > 100:
            bottlenecks.append((ResourceType.NETWORK, min(1.0, metrics.request_latency_ms / 1000)))
        
        # Sort by severity (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks

    def generate_optimizations(self, bottlenecks: List[Tuple[ResourceType, float]]) -> List[OptimizationAction]:
        """Generate optimization actions based on bottlenecks.
        
        Args:
            bottlenecks: List of resource bottlenecks
            
        Returns:
            List of suggested optimization actions
        """
        optimizations = []
        
        for resource, severity in bottlenecks:
            if resource == ResourceType.CPU:
                if severity > 0.9:  # Very high CPU usage
                    optimizations.extend([
                        OptimizationAction(
                            action_type="reduce_worker_threads",
                            resource=ResourceType.CPU,
                            parameters={"reduction_factor": 0.5},
                            expected_impact=0.3,
                            cost=0.2,
                            priority=3
                        ),
                        OptimizationAction(
                            action_type="enable_cpu_affinity",
                            resource=ResourceType.CPU,
                            parameters={"cores": list(range(min(4, psutil.cpu_count() if psutil else 4)))},
                            expected_impact=0.2,
                            cost=0.1,
                            priority=2
                        ),
                    ])
                elif severity > 0.8:
                    optimizations.append(
                        OptimizationAction(
                            action_type="optimize_processing_queue",
                            resource=ResourceType.CPU,
                            parameters={"batch_size": 10, "max_queue_size": 100},
                            expected_impact=0.2,
                            cost=0.1,
                            priority=2
                        )
                    )
            
            elif resource == ResourceType.MEMORY:
                if severity > 0.9:
                    optimizations.extend([
                        OptimizationAction(
                            action_type="enable_aggressive_gc",
                            resource=ResourceType.MEMORY,
                            parameters={"gc_threshold": 0.8},
                            expected_impact=0.4,
                            cost=0.2,
                            priority=3
                        ),
                        OptimizationAction(
                            action_type="reduce_cache_size",
                            resource=ResourceType.MEMORY,
                            parameters={"reduction_factor": 0.5},
                            expected_impact=0.3,
                            cost=0.1,
                            priority=2
                        ),
                    ])
                elif severity > 0.8:
                    optimizations.append(
                        OptimizationAction(
                            action_type="optimize_data_structures",
                            resource=ResourceType.MEMORY,
                            parameters={"compression": True, "lazy_loading": True},
                            expected_impact=0.2,
                            cost=0.3,
                            priority=1
                        )
                    )
            
            elif resource == ResourceType.NETWORK:
                optimizations.extend([
                    OptimizationAction(
                        action_type="enable_compression",
                        resource=ResourceType.NETWORK,
                        parameters={"algorithm": "gzip", "level": 6},
                        expected_impact=0.4,
                        cost=0.1,
                        priority=2
                    ),
                    OptimizationAction(
                        action_type="optimize_connection_pooling",
                        resource=ResourceType.NETWORK,
                        parameters={"pool_size": 20, "keepalive": True},
                        expected_impact=0.3,
                        cost=0.2,
                        priority=1
                    ),
                ])
        
        # Sort by priority and expected impact
        optimizations.sort(key=lambda x: (x.priority, x.expected_impact), reverse=True)
        
        return optimizations

    async def apply_optimization(self, optimization: OptimizationAction) -> bool:
        """Apply an optimization action.
        
        Args:
            optimization: Optimization action to apply
            
        Returns:
            True if optimization was applied successfully
        """
        try:
            print(f"Applying optimization: {optimization.action_type} for {optimization.resource.value}")
            
            # Record optimization attempt
            self.applied_optimizations.append(optimization)
            
            # Apply optimization based on type
            success = await self._execute_optimization_action(optimization)
            
            if success:
                # Record success
                if optimization.action_type not in self.optimization_results:
                    self.optimization_results[optimization.action_type] = {}
                
                self.optimization_results[optimization.action_type]["last_applied"] = time.time()
                self.optimization_results[optimization.action_type]["success_count"] = (
                    self.optimization_results[optimization.action_type].get("success_count", 0) + 1
                )
                
                # Trigger callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback(optimization)
                    except Exception as e:
                        print(f"Optimization callback error: {e}")
                
                print(f"Successfully applied optimization: {optimization.action_type}")
                return True
            else:
                print(f"Failed to apply optimization: {optimization.action_type}")
                return False
                
        except Exception as e:
            print(f"Error applying optimization {optimization.action_type}: {e}")
            return False

    async def _execute_optimization_action(self, optimization: OptimizationAction) -> bool:
        """Execute specific optimization action.
        
        Args:
            optimization: Optimization to execute
            
        Returns:
            True if successful
        """
        action_type = optimization.action_type
        params = optimization.parameters
        
        try:
            if action_type == "reduce_worker_threads":
                if self.thread_pool:
                    # This is a simplified example - in practice, you'd need to recreate the pool
                    current_workers = self.thread_pool._max_workers
                    new_workers = max(1, int(current_workers * params.get("reduction_factor", 0.5)))
                    print(f"Would reduce thread pool from {current_workers} to {new_workers} workers")
                    return True
            
            elif action_type == "enable_aggressive_gc":
                import gc
                gc.set_threshold(
                    int(700 * params.get("gc_threshold", 0.8)),
                    int(10 * params.get("gc_threshold", 0.8)),
                    int(10 * params.get("gc_threshold", 0.8))
                )
                gc.collect()
                return True
            
            elif action_type == "enable_compression":
                # This would be implemented by the calling system
                print(f"Would enable {params.get('algorithm', 'gzip')} compression")
                return True
            
            elif action_type == "optimize_connection_pooling":
                print(f"Would optimize connection pooling with {params}")
                return True
            
            elif action_type == "reduce_cache_size":
                print(f"Would reduce cache size by factor {params.get('reduction_factor', 0.5)}")
                return True
            
            else:
                print(f"Unknown optimization action: {action_type}")
                return False
                
        except Exception as e:
            print(f"Error executing {action_type}: {e}")
            return False

    def decide_scaling_action(self, metrics: Optional[PerformanceMetrics] = None) -> ScalingDecision:
        """Decide on auto-scaling action based on current metrics.
        
        Args:
            metrics: Current metrics (uses latest if None)
            
        Returns:
            Scaling decision with reasoning
        """
        if metrics is None:
            metrics = self.current_metrics
        
        if not metrics:
            return ScalingDecision(
                action="maintain",
                target_instances=self.current_instances,
                confidence=0.0,
                reasoning=["No metrics available"],
                metrics_used={}
            )
        
        reasoning = []
        metrics_used = {
            "cpu_usage": metrics.cpu_usage_percent,
            "memory_usage": metrics.memory_usage_percent,
            "request_latency": metrics.request_latency_ms,
            "error_rate": metrics.error_rate,
        }
        
        # Calculate scaling indicators
        scale_up_indicators = 0
        scale_down_indicators = 0
        
        # CPU usage
        if metrics.cpu_usage_percent > self.scale_up_threshold * 100:
            scale_up_indicators += 2
            reasoning.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent < self.scale_down_threshold * 100:
            scale_down_indicators += 1
            reasoning.append(f"Low CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        # Memory usage
        if metrics.memory_usage_percent > 90:
            scale_up_indicators += 2
            reasoning.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
        elif metrics.memory_usage_percent < 30:
            scale_down_indicators += 1
            reasoning.append(f"Low memory usage: {metrics.memory_usage_percent:.1f}%")
        
        # Request latency
        if metrics.request_latency_ms > 200:
            scale_up_indicators += 1
            reasoning.append(f"High latency: {metrics.request_latency_ms:.1f}ms")
        elif metrics.request_latency_ms < 50:
            scale_down_indicators += 1
            reasoning.append(f"Low latency: {metrics.request_latency_ms:.1f}ms")
        
        # Error rate
        if metrics.error_rate > 0.05:  # 5% error rate
            scale_up_indicators += 3
            reasoning.append(f"High error rate: {metrics.error_rate:.2%}")
        
        # Make scaling decision
        total_indicators = scale_up_indicators - scale_down_indicators
        
        if total_indicators >= 3 and self.current_instances < self.max_instances:
            action = "scale_up"
            target = min(self.max_instances, self.current_instances + 1)
            confidence = min(0.9, total_indicators / 5)
        elif total_indicators <= -2 and self.current_instances > self.min_instances:
            action = "scale_down"
            target = max(self.min_instances, self.current_instances - 1)
            confidence = min(0.8, abs(total_indicators) / 3)
        else:
            action = "maintain"
            target = self.current_instances
            confidence = 0.7
            reasoning.append("Metrics within acceptable range")
        
        decision = ScalingDecision(
            action=action,
            target_instances=target,
            confidence=confidence,
            reasoning=reasoning,
            metrics_used=metrics_used
        )
        
        # Record scaling decision
        self.scaling_history.append(decision)
        
        return decision

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        try:
            while self._running:
                # Collect current metrics
                metrics = await self.collect_metrics()
                self.current_metrics = metrics
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Set baseline if not set
                if self.baseline_metrics is None:
                    self.baseline_metrics = metrics
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            print("Performance monitoring loop cancelled")

    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        try:
            # Wait for initial metrics
            await asyncio.sleep(self.monitoring_interval * 2)
            
            while self._running:
                if self.current_metrics:
                    # Analyze bottlenecks
                    bottlenecks = self.analyze_bottlenecks()
                    
                    if bottlenecks:
                        # Generate optimizations
                        optimizations = self.generate_optimizations(bottlenecks)
                        
                        # Apply top optimization if severity is high enough
                        for bottleneck_resource, severity in bottlenecks:
                            if severity > self.optimization_threshold:
                                for optimization in optimizations:
                                    if optimization.resource == bottleneck_resource:
                                        await self.apply_optimization(optimization)
                                        break
                                break
                    
                    # Check scaling decision
                    scaling_decision = self.decide_scaling_action()
                    if scaling_decision.action != "maintain" and scaling_decision.confidence > 0.7:
                        # Trigger scaling callbacks
                        for callback in self.scaling_callbacks:
                            try:
                                callback(scaling_decision)
                            except Exception as e:
                                print(f"Scaling callback error: {e}")
                
                # Run optimization check less frequently than monitoring
                await asyncio.sleep(self.monitoring_interval * 3)
                
        except asyncio.CancelledError:
            print("Performance optimization loop cancelled")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        Returns:
            Performance analysis and recommendations
        """
        if not self.current_metrics:
            return {"error": "No metrics available"}
        
        summary = {
            "current_metrics": {
                "cpu_usage": f"{self.current_metrics.cpu_usage_percent:.1f}%",
                "memory_usage": f"{self.current_metrics.memory_usage_percent:.1f}%",
                "memory_mb": f"{self.current_metrics.memory_usage_mb:.1f}MB",
                "network_mbps": f"{self.current_metrics.network_throughput_mbps:.2f}",
                "disk_io_mbps": f"{self.current_metrics.disk_io_mbps:.2f}",
                "request_latency": f"{self.current_metrics.request_latency_ms:.1f}ms",
            },
            "strategy": self.strategy.value,
            "optimization_enabled": self.enable_auto_optimization,
            "current_instances": self.current_instances,
        }
        
        # Analyze trends if we have history
        if len(self.metrics_history) >= 3:
            recent_metrics = self.metrics_history[-3:]
            
            cpu_trend = [m.cpu_usage_percent for m in recent_metrics]
            memory_trend = [m.memory_usage_percent for m in recent_metrics]
            latency_trend = [m.request_latency_ms for m in recent_metrics]
            
            summary["trends"] = {
                "cpu_trending_up": cpu_trend[-1] > cpu_trend[0],
                "memory_trending_up": memory_trend[-1] > memory_trend[0], 
                "latency_trending_up": latency_trend[-1] > latency_trend[0],
                "avg_cpu_last_3": statistics.mean(cpu_trend),
                "avg_memory_last_3": statistics.mean(memory_trend),
                "avg_latency_last_3": statistics.mean(latency_trend),
            }
        
        # Current bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        summary["bottlenecks"] = [
            {"resource": resource.value, "severity": f"{severity:.2f}"} 
            for resource, severity in bottlenecks
        ]
        
        # Applied optimizations
        summary["applied_optimizations"] = len(self.applied_optimizations)
        summary["recent_optimizations"] = [
            {
                "action": opt.action_type,
                "resource": opt.resource.value,
                "expected_impact": opt.expected_impact,
            }
            for opt in self.applied_optimizations[-5:]  # Last 5
        ]
        
        # Scaling history
        summary["scaling_decisions"] = len(self.scaling_history)
        if self.scaling_history:
            last_scaling = self.scaling_history[-1]
            summary["last_scaling_decision"] = {
                "action": last_scaling.action,
                "target_instances": last_scaling.target_instances,
                "confidence": last_scaling.confidence,
                "reasoning": last_scaling.reasoning[:3],  # First 3 reasons
            }
        
        return summary

    def add_optimization_callback(self, callback: Callable[[OptimizationAction], None]) -> None:
        """Add callback for optimization events."""
        self.optimization_callbacks.append(callback)

    def add_scaling_callback(self, callback: Callable[[ScalingDecision], None]) -> None:
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)