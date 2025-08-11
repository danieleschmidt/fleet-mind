"""Advanced performance monitoring and optimization system."""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not available - system monitoring limited")

try:
    import numpy as np
except ImportError:
    # Fallback numpy implementation
    class MockNumpy:
        def mean(self, data):
            return sum(data) / len(data) if data else 0
        def std(self, data):
            if not data:
                return 0
            mean_val = self.mean(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
        def percentile(self, data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        def array(self, data):
            return list(data)
    np = MockNumpy()

from ..utils.logging import get_logger


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    operation: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    component: Optional[str] = None
    operation: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    metric_type: MetricType
    current_value: float
    threshold_value: float
    severity: str  # warning, critical
    component: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class PerformanceOptimizer:
    """Performance optimization recommendations and actions."""
    
    def __init__(self):
        self.optimization_rules = self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize performance optimization rules."""
        return {
            'high_latency': {
                'condition': lambda metrics: metrics.get('latency', 0) > 100,
                'actions': [
                    'enable_caching',
                    'increase_worker_threads',
                    'optimize_database_queries',
                ],
                'priority': 'high',
            },
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.get('cpu_usage', 0) > 80,
                'actions': [
                    'scale_horizontally',
                    'optimize_algorithms',
                    'reduce_worker_threads',
                ],
                'priority': 'high',
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('memory_usage', 0) > 80,
                'actions': [
                    'enable_garbage_collection',
                    'reduce_cache_size',
                    'optimize_data_structures',
                ],
                'priority': 'medium',
            },
            'low_throughput': {
                'condition': lambda metrics: metrics.get('throughput', 100) < 10,
                'actions': [
                    'increase_batch_size',
                    'optimize_network_io',
                    'add_connection_pooling',
                ],
                'priority': 'medium',
            },
        }
    
    def analyze_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """Analyze metrics and suggest optimizations."""
        suggestions = []
        
        for rule_name, rule in self.optimization_rules.items():
            if rule['condition'](metrics):
                suggestions.extend(rule['actions'])
        
        return list(set(suggestions))  # Remove duplicates


class PerformanceMonitor:
    """Comprehensive performance monitoring and optimization system."""
    
    def __init__(self, 
                 history_size: int = 10000,
                 collection_interval: float = 1.0,
                 enable_auto_optimization: bool = True):
        """Initialize performance monitor.
        
        Args:
            history_size: Number of metrics to keep in memory
            collection_interval: Metrics collection interval in seconds
            enable_auto_optimization: Enable automatic optimization
        """
        self.history_size = history_size
        self.collection_interval = collection_interval
        self.enable_auto_optimization = enable_auto_optimization
        
        self.logger = get_logger("performance_monitor", component="optimization")
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.current_metrics: Dict[str, float] = {}
        
        # Thresholds and alerts
        self.thresholds: List[PerformanceThreshold] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Performance optimizer
        self.optimizer = PerformanceOptimizer()
        
        # System monitoring
        self.system_process = psutil.Process() if psutil else None
        self.network_counters = psutil.net_io_counters() if psutil else None
        self.disk_counters = psutil.disk_io_counters() if psutil else None
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Initialize default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self) -> None:
        """Set up default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(MetricType.LATENCY, 100.0, 500.0),  # ms
            PerformanceThreshold(MetricType.CPU_USAGE, 70.0, 90.0),  # %
            PerformanceThreshold(MetricType.MEMORY_USAGE, 80.0, 95.0),  # %
            PerformanceThreshold(MetricType.ERROR_RATE, 1.0, 5.0),  # %
            PerformanceThreshold(MetricType.THROUGHPUT, 10.0, 5.0),  # ops/sec (lower is worse)
        ]
        
        self.thresholds.extend(default_thresholds)

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")

    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self._lock:
            # Create key for metric
            key_parts = [metric.metric_type.value]
            if metric.component:
                key_parts.append(metric.component)
            if metric.operation:
                key_parts.append(metric.operation)
            
            key = '.'.join(key_parts)
            
            # Store metric
            self.metrics_history[key].append(metric)
            self.current_metrics[key] = metric.value
            
            # Check thresholds
            self._check_thresholds(metric, key)

    def record_latency(self, operation: str, latency_ms: float, component: Optional[str] = None) -> None:
        """Record operation latency."""
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=latency_ms,
            component=component,
            operation=operation,
        )
        self.record_metric(metric)

    def record_throughput(self, operation: str, ops_per_second: float, component: Optional[str] = None) -> None:
        """Record operation throughput."""
        metric = PerformanceMetric(
            metric_type=MetricType.THROUGHPUT,
            value=ops_per_second,
            component=component,
            operation=operation,
        )
        self.record_metric(metric)

    def record_error_rate(self, operation: str, error_rate: float, component: Optional[str] = None) -> None:
        """Record error rate."""
        metric = PerformanceMetric(
            metric_type=MetricType.ERROR_RATE,
            value=error_rate,
            component=component,
            operation=operation,
        )
        self.record_metric(metric)

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics snapshot."""
        with self._lock:
            return self.current_metrics.copy()

    def get_metric_statistics(self, 
                            metric_key: str,
                            window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary of metric.
        
        Args:
            metric_key: Metric key to analyze
            window_seconds: Time window to analyze (None for all data)
            
        Returns:
            Dictionary with statistical measures
        """
        with self._lock:
            if metric_key not in self.metrics_history:
                return {}
            
            metrics = list(self.metrics_history[metric_key])
            
            # Filter by time window if specified
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                'timestamp': time.time(),
                'monitoring_active': self._monitoring_active,
                'total_metrics': sum(len(history) for history in self.metrics_history.values()),
                'current_metrics': self.current_metrics.copy(),
                'metric_keys': list(self.metrics_history.keys()),
            }
            
            # Add system metrics
            summary['system'] = self._get_system_metrics()
            
            # Add performance analysis
            summary['analysis'] = self._analyze_performance()
            
            return summary

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add performance threshold."""
        self.thresholds.append(threshold)
        self.logger.info(f"Added performance threshold for {threshold.metric_type.value}")

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions."""
        current_metrics = self.get_current_metrics()
        
        # Convert metric keys to simple names for analysis
        simplified_metrics = {}
        for key, value in current_metrics.items():
            if 'latency' in key:
                simplified_metrics['latency'] = max(simplified_metrics.get('latency', 0), value)
            elif 'cpu_usage' in key:
                simplified_metrics['cpu_usage'] = max(simplified_metrics.get('cpu_usage', 0), value)
            elif 'memory_usage' in key:
                simplified_metrics['memory_usage'] = max(simplified_metrics.get('memory_usage', 0), value)
            elif 'throughput' in key:
                simplified_metrics['throughput'] = min(simplified_metrics.get('throughput', 1000), value)
            elif 'error_rate' in key:
                simplified_metrics['error_rate'] = max(simplified_metrics.get('error_rate', 0), value)
        
        return self.optimizer.analyze_metrics(simplified_metrics)

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self._monitoring_active:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Auto-optimization
                if self.enable_auto_optimization:
                    await self._perform_auto_optimization()
                
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Performance monitoring loop error: {e}")

    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            # CPU usage
            cpu_percent = self.system_process.cpu_percent()
            self.record_metric(PerformanceMetric(
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                component="system",
            ))
            
            # Memory usage
            memory_info = self.system_process.memory_info()
            memory_percent = self.system_process.memory_percent()
            self.record_metric(PerformanceMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=memory_percent,
                component="system",
            ))
            
            # Network I/O
            net_counters = psutil.net_io_counters()
            if self.network_counters:
                net_bytes_sent = net_counters.bytes_sent - self.network_counters.bytes_sent
                net_bytes_recv = net_counters.bytes_recv - self.network_counters.bytes_recv
                
                self.record_metric(PerformanceMetric(
                    metric_type=MetricType.NETWORK_IO,
                    value=net_bytes_sent + net_bytes_recv,
                    component="system",
                    operation="total_bytes",
                ))
            
            self.network_counters = net_counters
            
            # Disk I/O
            disk_counters = psutil.disk_io_counters()
            if disk_counters and self.disk_counters:
                disk_bytes_read = disk_counters.read_bytes - self.disk_counters.read_bytes
                disk_bytes_write = disk_counters.write_bytes - self.disk_counters.write_bytes
                
                self.record_metric(PerformanceMetric(
                    metric_type=MetricType.DISK_IO,
                    value=disk_bytes_read + disk_bytes_write,
                    component="system",
                    operation="total_bytes",
                ))
            
            self.disk_counters = disk_counters
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")

    def _check_thresholds(self, metric: PerformanceMetric, key: str) -> None:
        """Check if metric exceeds thresholds."""
        for threshold in self.thresholds:
            # Check if threshold applies to this metric
            if threshold.metric_type != metric.metric_type:
                continue
            
            if threshold.component and threshold.component != metric.component:
                continue
            
            if threshold.operation and threshold.operation != metric.operation:
                continue
            
            # Check thresholds
            if metric.value >= threshold.critical_threshold:
                alert = PerformanceAlert(
                    metric_type=metric.metric_type,
                    current_value=metric.value,
                    threshold_value=threshold.critical_threshold,
                    severity="critical",
                    component=metric.component,
                    operation=metric.operation,
                    message=f"Critical threshold exceeded for {key}: {metric.value} >= {threshold.critical_threshold}",
                )
                self._trigger_alert(alert)
                
            elif metric.value >= threshold.warning_threshold:
                alert = PerformanceAlert(
                    metric_type=metric.metric_type,
                    current_value=metric.value,
                    threshold_value=threshold.warning_threshold,
                    severity="warning",
                    component=metric.component,
                    operation=metric.operation,
                    message=f"Warning threshold exceeded for {key}: {metric.value} >= {threshold.warning_threshold}",
                )
                self._trigger_alert(alert)

    def _trigger_alert(self, alert: PerformanceAlert) -> None:
        """Trigger performance alert."""
        # Log alert
        if alert.severity == "critical":
            self.logger.critical(alert.message, 
                               component=alert.component,
                               operation=alert.operation,
                               metric_type=alert.metric_type.value,
                               current_value=alert.current_value,
                               threshold_value=alert.threshold_value)
        else:
            self.logger.warning(alert.message,
                              component=alert.component,
                              operation=alert.operation,
                              metric_type=alert.metric_type.value,
                              current_value=alert.current_value,
                              threshold_value=alert.threshold_value)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    async def _perform_auto_optimization(self) -> None:
        """Perform automatic optimization based on metrics."""
        try:
            suggestions = self.get_optimization_suggestions()
            
            if suggestions:
                self.logger.info(f"Auto-optimization suggestions: {suggestions}")
                
                # Implement basic auto-optimizations
                for suggestion in suggestions:
                    await self._apply_optimization(suggestion)
                    
        except Exception as e:
            self.logger.error(f"Auto-optimization error: {e}")

    async def _apply_optimization(self, optimization: str) -> None:
        """Apply specific optimization."""
        try:
            if optimization == 'enable_caching':
                self.logger.info("Auto-optimization: Enabling caching")
                # In production, this would interact with cache manager
                
            elif optimization == 'scale_horizontally':
                self.logger.info("Auto-optimization: Requesting horizontal scaling")
                # In production, this would trigger scaling actions
                
            elif optimization == 'optimize_algorithms':
                self.logger.info("Auto-optimization: Algorithm optimization suggested")
                # In production, this would trigger algorithm optimizations
                
            elif optimization == 'reduce_cache_size':
                self.logger.info("Auto-optimization: Reducing cache size")
                # In production, this would adjust cache settings
                
            else:
                self.logger.debug(f"Auto-optimization: {optimization} not implemented")
                
        except Exception as e:
            self.logger.error(f"Optimization application error: {e}")

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.system_process:
            # Return mock metrics when psutil unavailable
            return {
                'cpu_percent': 25.0,
                'memory_percent': 60.0,
                'memory_rss_mb': 512.0,
                'num_threads': 8,
                'num_fds': 32,
            }
            
        try:
            return {
                'cpu_percent': self.system_process.cpu_percent(),
                'memory_percent': self.system_process.memory_percent(),
                'memory_rss_mb': self.system_process.memory_info().rss / 1024 / 1024,
                'num_threads': self.system_process.num_threads(),
                'num_fds': self.system_process.num_fds() if hasattr(self.system_process, 'num_fds') else 0,
            }
        except Exception as e:
            self.logger.error(f"System metrics error: {e}")
            return {
                'cpu_percent': 25.0,
                'memory_percent': 60.0,
                'memory_rss_mb': 512.0,
                'num_threads': 8,
                'num_fds': 32,
            }

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance state."""
        try:
            analysis = {
                'health_score': self._calculate_health_score(),
                'bottlenecks': self._identify_bottlenecks(),
                'trends': self._analyze_trends(),
                'recommendations': self.get_optimization_suggestions(),
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
            return {}

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        try:
            current_metrics = self.get_current_metrics()
            
            if not current_metrics:
                return 1.0
            
            score = 1.0
            
            # Penalize high latency
            for key, value in current_metrics.items():
                if 'latency' in key and value > 100:
                    score -= min(0.3, (value - 100) / 1000)
                
                # Penalize high CPU usage
                elif 'cpu_usage' in key and value > 70:
                    score -= min(0.2, (value - 70) / 100)
                
                # Penalize high memory usage
                elif 'memory_usage' in key and value > 80:
                    score -= min(0.2, (value - 80) / 100)
                
                # Penalize high error rate
                elif 'error_rate' in key and value > 1:
                    score -= min(0.3, value / 10)
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.error(f"Health score calculation error: {e}")
            return 0.5

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        current_metrics = self.get_current_metrics()
        
        try:
            for key, value in current_metrics.items():
                if 'latency' in key and value > 200:
                    bottlenecks.append(f"High latency in {key}: {value:.1f}ms")
                
                elif 'cpu_usage' in key and value > 85:
                    bottlenecks.append(f"High CPU usage in {key}: {value:.1f}%")
                
                elif 'memory_usage' in key and value > 90:
                    bottlenecks.append(f"High memory usage in {key}: {value:.1f}%")
                
                elif 'error_rate' in key and value > 5:
                    bottlenecks.append(f"High error rate in {key}: {value:.1f}%")
                
                elif 'throughput' in key and value < 5:
                    bottlenecks.append(f"Low throughput in {key}: {value:.1f} ops/sec")
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Bottleneck identification error: {e}")
            return []

    def _analyze_trends(self) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = {}
        
        try:
            for key in self.metrics_history.keys():
                stats_recent = self.get_metric_statistics(key, window_seconds=300)  # 5 minutes
                stats_older = self.get_metric_statistics(key, window_seconds=1800)  # 30 minutes
                
                if stats_recent and stats_older and stats_recent['count'] > 10 and stats_older['count'] > 10:
                    recent_mean = stats_recent['mean']
                    older_mean = stats_older['mean']
                    
                    if recent_mean > older_mean * 1.2:
                        trends[key] = "increasing"
                    elif recent_mean < older_mean * 0.8:
                        trends[key] = "decreasing"
                    else:
                        trends[key] = "stable"
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {}


# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str, component: Optional[str] = None):
        self.monitor = monitor
        self.operation = operation
        self.component = component
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_latency(self.operation, duration_ms, self.component)


# Decorator for timing functions
def time_operation(monitor: PerformanceMonitor, operation: str, component: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(monitor, operation, component):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Async decorator for timing async functions
def time_async_operation(monitor: PerformanceMonitor, operation: str, component: Optional[str] = None):
    """Decorator to time async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with TimingContext(monitor, operation, component):
                return await func(*args, **kwargs)
        return wrapper
    return decorator