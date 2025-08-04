"""Tests for PerformanceMonitor functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from fleet_mind.optimization.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetric,
    PerformanceThreshold,
    PerformanceAlert,
    MetricType,
    TimingContext,
    time_operation
)


@pytest.fixture
async def monitor():
    """Create PerformanceMonitor instance."""
    monitor = PerformanceMonitor(
        history_size=100,
        collection_interval=0.1,
        enable_auto_optimization=False  # Disable for testing
    )
    await monitor.start_monitoring()
    yield monitor
    await monitor.stop_monitoring()


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(
            history_size=500,
            collection_interval=2.0,
            enable_auto_optimization=True
        )
        
        assert monitor.history_size == 500
        assert monitor.collection_interval == 2.0
        assert monitor.enable_auto_optimization is True
        assert not monitor._monitoring_active
        assert len(monitor.thresholds) > 0  # Default thresholds

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = PerformanceMonitor()
        
        assert not monitor._monitoring_active
        
        await monitor.start_monitoring()
        assert monitor._monitoring_active
        assert monitor._monitoring_task is not None
        
        await monitor.stop_monitoring()
        assert not monitor._monitoring_active

    def test_record_metric(self, monitor):
        """Test metric recording."""
        metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=50.0,
            component="test_component",
            operation="test_operation"
        )
        
        monitor.record_metric(metric)
        
        # Check metric was stored
        key = "latency.test_component.test_operation"
        assert key in monitor.metrics_history
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] == 50.0

    def test_record_latency(self, monitor):
        """Test latency recording convenience method."""
        monitor.record_latency("api_call", 75.5, "web_service")
        
        key = "latency.web_service.api_call"
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] == 75.5

    def test_record_throughput(self, monitor):
        """Test throughput recording."""
        monitor.record_throughput("requests", 150.0, "api_server")
        
        key = "throughput.api_server.requests"
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] == 150.0

    def test_record_error_rate(self, monitor):
        """Test error rate recording."""
        monitor.record_error_rate("authentication", 2.5, "auth_service")
        
        key = "error_rate.auth_service.authentication"
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] == 2.5

    def test_get_current_metrics(self, monitor):
        """Test current metrics retrieval."""
        # Record some metrics
        monitor.record_latency("test_op", 100.0)
        monitor.record_throughput("test_op", 50.0)
        
        metrics = monitor.get_current_metrics()
        
        assert "latency.test_op" in metrics
        assert "throughput.test_op" in metrics
        assert metrics["latency.test_op"] == 100.0
        assert metrics["throughput.test_op"] == 50.0

    def test_get_metric_statistics(self, monitor):
        """Test metric statistics calculation."""
        # Record multiple values for same metric
        for i in range(10):
            monitor.record_latency("test_op", float(i * 10))
        
        stats = monitor.get_metric_statistics("latency.test_op")
        
        assert stats['count'] == 10
        assert stats['min'] == 0.0
        assert stats['max'] == 90.0
        assert stats['mean'] == 45.0
        assert 'median' in stats
        assert 'std' in stats
        assert 'p95' in stats
        assert 'p99' in stats

    def test_get_metric_statistics_with_time_window(self, monitor):
        """Test metric statistics with time window."""
        # Record metrics at different times
        current_time = time.time()
        
        # Old metric (should be filtered out)
        old_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=100.0,
            component="test",
            operation="old_op",
            timestamp=current_time - 3600  # 1 hour ago
        )
        monitor.record_metric(old_metric)
        
        # Recent metric
        recent_metric = PerformanceMetric(
            metric_type=MetricType.LATENCY,
            value=50.0,
            component="test", 
            operation="old_op",
            timestamp=current_time
        )
        monitor.record_metric(recent_metric)
        
        # Get stats for last 30 minutes
        stats = monitor.get_metric_statistics("latency.test.old_op", window_seconds=1800)
        
        # Should only include recent metric
        assert stats['count'] == 1
        assert stats['mean'] == 50.0

    def test_threshold_checking(self, monitor):
        """Test threshold checking and alerts."""
        alert_triggered = False
        
        def alert_callback(alert):
            nonlocal alert_triggered
            alert_triggered = True
            assert isinstance(alert, PerformanceAlert)
            assert alert.severity in ["warning", "critical"]
        
        monitor.add_alert_callback(alert_callback)
        
        # Add threshold
        threshold = PerformanceThreshold(
            metric_type=MetricType.LATENCY,
            warning_threshold=100.0,
            critical_threshold=200.0
        )
        monitor.add_threshold(threshold)
        
        # Record metric that exceeds warning threshold
        monitor.record_latency("slow_operation", 150.0)
        
        assert alert_triggered

    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Record some metrics
        monitor.record_latency("api_call", 75.0)
        monitor.record_throughput("requests", 100.0)
        monitor.record_error_rate("errors", 1.5)
        
        summary = monitor.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'monitoring_active' in summary
        assert 'total_metrics' in summary
        assert 'current_metrics' in summary
        assert 'system' in summary
        assert 'analysis' in summary

    def test_optimization_suggestions(self, monitor):
        """Test optimization suggestions."""
        # Set high latency to trigger suggestions
        monitor.current_metrics['latency'] = 200.0
        monitor.current_metrics['cpu_usage'] = 85.0
        
        suggestions = monitor.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    @pytest.mark.asyncio
    async def test_monitoring_loop_execution(self):
        """Test monitoring loop execution."""
        monitor = PerformanceMonitor(collection_interval=0.1)
        
        with patch.object(monitor, '_collect_system_metrics', new=AsyncMock()) as mock_collect:
            await monitor.start_monitoring()
            
            # Let it run for a short time
            await asyncio.sleep(0.3)
            
            await monitor.stop_monitoring()
            
            # Should have collected metrics multiple times
            assert mock_collect.call_count >= 2

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, monitor):
        """Test system metrics collection."""
        with patch('fleet_mind.optimization.performance_monitor.psutil') as mock_psutil:
            # Mock system process
            mock_process = Mock()
            mock_process.cpu_percent.return_value = 50.0
            mock_process.memory_percent.return_value = 60.0
            mock_process.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
            
            monitor.system_process = mock_process
            
            await monitor._collect_system_metrics()
            
            # Check that system metrics were recorded
            assert any('cpu_usage' in key for key in monitor.current_metrics.keys())
            assert any('memory_usage' in key for key in monitor.current_metrics.keys())

    def test_health_score_calculation(self, monitor):
        """Test health score calculation."""
        # Good metrics
        monitor.current_metrics = {
            'latency.api': 50.0,
            'cpu_usage.system': 30.0,
            'memory_usage.system': 40.0,
            'error_rate.service': 0.5,
        }
        
        health_score = monitor._calculate_health_score()
        assert 0.7 <= health_score <= 1.0  # Should be high
        
        # Bad metrics
        monitor.current_metrics = {
            'latency.api': 500.0,
            'cpu_usage.system': 95.0,
            'memory_usage.system': 95.0,
            'error_rate.service': 10.0,
        }
        
        health_score = monitor._calculate_health_score()
        assert 0.0 <= health_score <= 0.3  # Should be low

    def test_bottleneck_identification(self, monitor):
        """Test bottleneck identification."""
        # Set problematic metrics
        monitor.current_metrics = {
            'latency.slow_service': 300.0,
            'cpu_usage.overloaded': 90.0,
            'memory_usage.memory_hog': 95.0,
            'error_rate.failing_service': 8.0,
            'throughput.slow_endpoint': 2.0,
        }
        
        bottlenecks = monitor._identify_bottlenecks()
        
        assert len(bottlenecks) > 0
        assert any('High latency' in b for b in bottlenecks)
        assert any('High CPU usage' in b for b in bottlenecks)
        assert any('High memory usage' in b for b in bottlenecks)
        assert any('High error rate' in b for b in bottlenecks)
        assert any('Low throughput' in b for b in bottlenecks)

    def test_trend_analysis(self, monitor):
        """Test trend analysis."""
        # Record metrics over time to create trends
        metric_key = "latency.test_service"
        
        # Older metrics (lower values)
        for i in range(20):
            metric = PerformanceMetric(
                metric_type=MetricType.LATENCY,
                value=50.0 + i,
                component="test_service",
                timestamp=time.time() - 1800 + i  # 30 minutes ago
            )
            monitor.metrics_history[metric_key].append(metric)
        
        # Recent metrics (higher values - increasing trend)
        for i in range(10):
            metric = PerformanceMetric(
                metric_type=MetricType.LATENCY,
                value=80.0 + i,
                component="test_service",
                timestamp=time.time() - 300 + i  # 5 minutes ago
            )
            monitor.metrics_history[metric_key].append(metric)
        
        trends = monitor._analyze_trends()
        
        # Should detect increasing trend
        if metric_key in trends:
            assert trends[metric_key] in ["increasing", "stable", "decreasing"]


class TestTimingContext:
    """Test timing context manager."""
    
    def test_timing_context(self):
        """Test timing context manager."""
        monitor = PerformanceMonitor()
        
        with TimingContext(monitor, "test_operation", "test_component"):
            time.sleep(0.01)  # Small delay
        
        # Check that latency was recorded
        key = "latency.test_component.test_operation"
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] > 0

    def test_timing_decorator(self):
        """Test timing decorator."""
        monitor = PerformanceMonitor()
        
        @time_operation(monitor, "decorated_function", "test_module")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        
        # Check that latency was recorded
        key = "latency.test_module.decorated_function"
        assert key in monitor.current_metrics
        assert monitor.current_metrics[key] > 0


class TestPerformanceMetric:
    """Test PerformanceMetric data class."""
    
    def test_metric_creation(self):
        """Test metric creation."""
        metric = PerformanceMetric(
            metric_type=MetricType.CPU_USAGE,
            value=75.5,
            component="web_server",
            operation="request_processing",
            tags={"region": "us-east", "env": "prod"}
        )
        
        assert metric.metric_type == MetricType.CPU_USAGE
        assert metric.value == 75.5
        assert metric.component == "web_server"
        assert metric.operation == "request_processing"
        assert metric.tags["region"] == "us-east"
        assert metric.timestamp <= time.time()


class TestPerformanceThreshold:
    """Test PerformanceThreshold data class."""
    
    def test_threshold_creation(self):
        """Test threshold creation."""
        threshold = PerformanceThreshold(
            metric_type=MetricType.LATENCY,
            warning_threshold=100.0,
            critical_threshold=500.0,
            component="api_service",
            operation="user_login"
        )
        
        assert threshold.metric_type == MetricType.LATENCY
        assert threshold.warning_threshold == 100.0
        assert threshold.critical_threshold == 500.0
        assert threshold.component == "api_service"
        assert threshold.operation == "user_login"


class TestPerformanceAlert:
    """Test PerformanceAlert data class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = PerformanceAlert(
            metric_type=MetricType.ERROR_RATE,
            current_value=5.5,
            threshold_value=3.0,
            severity="critical",
            component="payment_service",
            message="Error rate exceeded critical threshold"
        )
        
        assert alert.metric_type == MetricType.ERROR_RATE
        assert alert.current_value == 5.5
        assert alert.threshold_value == 3.0
        assert alert.severity == "critical"
        assert alert.component == "payment_service"
        assert "exceeded" in alert.message.lower()
        assert alert.timestamp <= time.time()


@pytest.mark.asyncio
async def test_auto_optimization():
    """Test auto-optimization functionality."""
    monitor = PerformanceMonitor(
        collection_interval=0.1,
        enable_auto_optimization=True
    )
    
    # Mock optimization methods
    with patch.object(monitor, '_apply_optimization', new=AsyncMock()) as mock_apply:
        with patch.object(monitor, 'get_optimization_suggestions', return_value=['enable_caching']):
            await monitor.start_monitoring()
            
            # Let it run for a short time
            await asyncio.sleep(0.3)
            
            await monitor.stop_monitoring()
            
            # Should have attempted optimization
            mock_apply.assert_called()