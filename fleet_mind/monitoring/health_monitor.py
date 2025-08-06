"""Advanced health monitoring and alerting system for Fleet-Mind."""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available - system monitoring limited")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class HealthStatus(Enum):
    """Health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""
    last_updated: float = field(default_factory=time.time)


@dataclass
class HealthAlert:
    """Health monitoring alert."""
    timestamp: float
    severity: AlertSeverity
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_name: str
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0


class HealthMonitor:
    """Comprehensive health monitoring system for Fleet-Mind components.
    
    Monitors system health, performance metrics, and generates alerts
    for various fleet components including coordinator, drones, and communication.
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        alert_cooldown: float = 300.0,  # 5 minutes
        enable_system_monitoring: bool = True,
        enable_network_monitoring: bool = True
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
            alert_cooldown: Minimum time between repeated alerts
            enable_system_monitoring: Enable system resource monitoring
            enable_network_monitoring: Enable network connectivity monitoring
        """
        self.check_interval = check_interval
        self.alert_cooldown = alert_cooldown
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_network_monitoring = enable_network_monitoring
        
        # Health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.alerts: List[HealthAlert] = []
        self.alert_history: List[HealthAlert] = []
        self.last_alert_times: Dict[str, float] = {}  # component:metric -> timestamp
        
        # Monitoring tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Performance tracking
        self.check_durations: List[float] = []
        self.start_time = time.time()
        
        # Component-specific thresholds
        self.default_thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "network_latency": {"warning": 100.0, "critical": 200.0},
            "error_rate": {"warning": 0.05, "critical": 0.10},
            "battery_level": {"warning": 20.0, "critical": 10.0},
            "communication_quality": {"warning": 0.7, "critical": 0.5},
        }
        
        print("Health monitor initialized")

    async def start_monitoring(self) -> None:
        """Start health monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        print(f"Health monitoring started (check interval: {self.check_interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring tasks."""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        print("Health monitoring stopped")

    def register_component(
        self, 
        component_name: str, 
        custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """Register a component for health monitoring.
        
        Args:
            component_name: Name of the component to monitor
            custom_thresholds: Custom threshold values for this component
        """
        self.component_health[component_name] = ComponentHealth(
            component_name=component_name,
            overall_status=HealthStatus.GOOD
        )
        
        # Set custom thresholds if provided
        if custom_thresholds:
            # Store custom thresholds (implementation would extend this)
            pass
        
        print(f"Registered component for health monitoring: {component_name}")

    def update_metric(
        self,
        component: str,
        metric_name: str,
        value: float,
        unit: str = "",
        description: str = ""
    ) -> None:
        """Update a health metric for a component.
        
        Args:
            component: Component name
            metric_name: Metric name
            value: Metric value
            unit: Value unit
            description: Metric description
        """
        if component not in self.component_health:
            self.register_component(component)
        
        # Determine health status based on thresholds
        thresholds = self.default_thresholds.get(metric_name, {})
        warning_threshold = thresholds.get("warning")
        critical_threshold = thresholds.get("critical")
        
        status = HealthStatus.GOOD
        if critical_threshold is not None and value >= critical_threshold:
            status = HealthStatus.CRITICAL
        elif warning_threshold is not None and value >= warning_threshold:
            status = HealthStatus.WARNING
        elif value < 0:  # Assume negative values are bad
            status = HealthStatus.FAILED
        
        # Special cases for metrics where lower is worse
        if metric_name in ["battery_level", "communication_quality"]:
            if critical_threshold is not None and value <= critical_threshold:
                status = HealthStatus.CRITICAL
            elif warning_threshold is not None and value <= warning_threshold:
                status = HealthStatus.WARNING
        
        metric = HealthMetric(
            name=metric_name,
            value=value,
            status=status,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold,
            unit=unit,
            description=description
        )
        
        self.component_health[component].metrics[metric_name] = metric
        self.component_health[component].last_check = time.time()
        
        # Update component overall status
        self._update_component_status(component)
        
        # Check for alerts
        self._check_alert_conditions(component, metric)

    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get health status for specific component.
        
        Args:
            component: Component name
            
        Returns:
            Component health information or None if not found
        """
        return self.component_health.get(component)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary.
        
        Returns:
            Comprehensive system health report
        """
        total_components = len(self.component_health)
        status_counts = {status.value: 0 for status in HealthStatus}
        
        for component in self.component_health.values():
            status_counts[component.overall_status.value] += 1
        
        # Calculate overall system status
        if status_counts["failed"] > 0:
            overall_status = HealthStatus.FAILED
        elif status_counts["critical"] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts["warning"] > total_components * 0.3:  # >30% warnings
            overall_status = HealthStatus.WARNING
        elif status_counts["good"] + status_counts["excellent"] == total_components:
            overall_status = HealthStatus.EXCELLENT
        else:
            overall_status = HealthStatus.GOOD
        
        # Active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        return {
            "overall_status": overall_status.value,
            "total_components": total_components,
            "status_breakdown": status_counts,
            "active_alerts": len(active_alerts),
            "total_alerts_24h": len([a for a in self.alert_history if time.time() - a.timestamp < 86400]),
            "uptime_seconds": time.time() - self.start_time,
            "last_check": max([c.last_check for c in self.component_health.values()]) if self.component_health else 0,
            "monitoring_performance": {
                "avg_check_duration_ms": statistics.mean(self.check_durations[-100:]) * 1000 if self.check_durations else 0,
                "total_checks": len(self.check_durations)
            }
        }

    def get_alerts(self, active_only: bool = True) -> List[HealthAlert]:
        """Get system alerts.
        
        Args:
            active_only: Return only active (unresolved) alerts
            
        Returns:
            List of health alerts
        """
        if active_only:
            return [alert for alert in self.alerts if not alert.resolved]
        else:
            return self.alert_history.copy()

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved.
        
        Args:
            alert_id: Alert identifier (timestamp + component + metric)
            
        Returns:
            True if alert was found and resolved
        """
        for alert in self.alerts:
            if not alert.resolved:
                # Simple ID matching based on timestamp and component
                current_id = f"{alert.timestamp}_{alert.component}_{alert.metric_name}"
                if current_id == alert_id:
                    alert.resolved = True
                    alert.resolution_timestamp = time.time()
                    return True
        return False

    def add_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Add callback function for alert notifications.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)

    async def check_network_connectivity(self, targets: List[str] = None) -> Dict[str, float]:
        """Check network connectivity to target hosts.
        
        Args:
            targets: List of target hosts to check
            
        Returns:
            Dictionary of target -> latency (ms)
        """
        if not self.enable_network_monitoring or not AIOHTTP_AVAILABLE:
            return {}
        
        if targets is None:
            targets = ["8.8.8.8", "1.1.1.1"]  # Google DNS, Cloudflare DNS
        
        results = {}
        
        for target in targets:
            try:
                start_time = time.time()
                
                # Simple HTTP connectivity check
                timeout = aiohttp.ClientTimeout(total=5.0)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://{target}", timeout=timeout) as response:
                        latency = (time.time() - start_time) * 1000
                        results[target] = latency
                        
            except Exception:
                results[target] = -1  # Connection failed
        
        return results

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self._running:
                start_time = time.time()
                
                try:
                    # System monitoring
                    if self.enable_system_monitoring:
                        await self._check_system_health()
                    
                    # Network monitoring
                    if self.enable_network_monitoring:
                        await self._check_network_health()
                    
                    # Component health aging (mark stale components)
                    self._check_component_staleness()
                    
                    # Cleanup old alerts
                    self._cleanup_old_alerts()
                    
                    # Record check duration
                    check_duration = time.time() - start_time
                    self.check_durations.append(check_duration)
                    
                    # Keep only recent check durations
                    if len(self.check_durations) > 1000:
                        self.check_durations = self.check_durations[-500:]
                    
                except Exception as e:
                    print(f"Health monitoring error: {e}")
                
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            print("Health monitoring loop cancelled")

    async def _check_system_health(self) -> None:
        """Check system resource health."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self.update_metric("system", "cpu_usage", cpu_percent, "%", "CPU utilization")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric("system", "memory_usage", memory.percent, "%", "Memory utilization")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.update_metric("system", "disk_usage", disk_percent, "%", "Disk utilization")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_rate = (net_io.bytes_sent - self._last_net_io.bytes_sent) / self.check_interval
                bytes_recv_rate = (net_io.bytes_recv - self._last_net_io.bytes_recv) / self.check_interval
                
                self.update_metric("system", "network_sent_rate", bytes_sent_rate / 1024 / 1024, "MB/s", "Network send rate")
                self.update_metric("system", "network_recv_rate", bytes_recv_rate / 1024 / 1024, "MB/s", "Network receive rate")
            
            self._last_net_io = net_io
            
        except Exception as e:
            print(f"System health check error: {e}")

    async def _check_network_health(self) -> None:
        """Check network connectivity health."""
        try:
            connectivity = await self.check_network_connectivity()
            
            if connectivity:
                avg_latency = statistics.mean([l for l in connectivity.values() if l >= 0])
                failed_connections = sum(1 for l in connectivity.values() if l < 0)
                
                self.update_metric("network", "average_latency", avg_latency, "ms", "Average network latency")
                self.update_metric("network", "failed_connections", failed_connections, "", "Failed connection attempts")
                
        except Exception as e:
            print(f"Network health check error: {e}")

    def _check_component_staleness(self) -> None:
        """Check for components with stale health data."""
        current_time = time.time()
        stale_threshold = self.check_interval * 3  # 3 check intervals
        
        for component_name, component in self.component_health.items():
            if current_time - component.last_check > stale_threshold:
                # Mark component as stale
                if component.overall_status != HealthStatus.FAILED:
                    component.overall_status = HealthStatus.WARNING
                    
                    # Create alert for stale component
                    alert = HealthAlert(
                        timestamp=current_time,
                        severity=AlertSeverity.WARNING,
                        component=component_name,
                        metric_name="staleness",
                        current_value=current_time - component.last_check,
                        threshold_value=stale_threshold,
                        message=f"Component {component_name} has stale health data"
                    )
                    
                    self._trigger_alert(alert)

    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        cutoff_time = time.time() - 86400  # 24 hours
        
        # Move old alerts to history
        old_alerts = [alert for alert in self.alerts if alert.resolved and 
                     (alert.resolution_timestamp or alert.timestamp) < cutoff_time]
        
        self.alert_history.extend(old_alerts)
        
        # Remove from active alerts
        self.alerts = [alert for alert in self.alerts if alert not in old_alerts]
        
        # Limit alert history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]

    def _update_component_status(self, component: str) -> None:
        """Update overall status for a component based on its metrics."""
        if component not in self.component_health:
            return
        
        component_health = self.component_health[component]
        
        if not component_health.metrics:
            component_health.overall_status = HealthStatus.GOOD
            return
        
        # Determine worst status among all metrics
        worst_status = HealthStatus.EXCELLENT
        status_priority = {
            HealthStatus.EXCELLENT: 0,
            HealthStatus.GOOD: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 3,
            HealthStatus.FAILED: 4
        }
        
        for metric in component_health.metrics.values():
            if status_priority[metric.status] > status_priority[worst_status]:
                worst_status = metric.status
        
        component_health.overall_status = worst_status

    def _check_alert_conditions(self, component: str, metric: HealthMetric) -> None:
        """Check if metric conditions warrant an alert."""
        if metric.status in [HealthStatus.GOOD, HealthStatus.EXCELLENT]:
            return
        
        # Check cooldown
        alert_key = f"{component}:{metric.name}"
        current_time = time.time()
        
        if alert_key in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_key]
            if time_since_last < self.alert_cooldown:
                return
        
        # Determine alert severity
        if metric.status == HealthStatus.FAILED:
            severity = AlertSeverity.CRITICAL
            threshold = 0  # Failed metrics don't have meaningful thresholds
        elif metric.status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
            threshold = metric.threshold_critical or 0
        else:  # WARNING
            severity = AlertSeverity.WARNING
            threshold = metric.threshold_warning or 0
        
        # Create alert
        alert = HealthAlert(
            timestamp=current_time,
            severity=severity,
            component=component,
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold,
            message=f"{component} {metric.name} is {metric.status.value}: {metric.value}{metric.unit} (threshold: {threshold}{metric.unit})"
        )
        
        self._trigger_alert(alert)
        self.last_alert_times[alert_key] = current_time

    def _trigger_alert(self, alert: HealthAlert) -> None:
        """Trigger an alert and notify callbacks."""
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        # Print critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            print(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            print(f"ERROR ALERT: {alert.message}")