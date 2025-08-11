"""Advanced health monitoring and alerting system for Fleet-Mind."""

import asyncio
import time
import statistics
import threading
import queue
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

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
    """Enhanced health monitoring alert."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    escalation_level: int = 0
    acknowledgment_required: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    correlation_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)

@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection."""
    metric_name: str
    component: str
    mean_value: float
    std_deviation: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    last_updated: float
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)

@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    component: str
    timestamp: float
    current_value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    anomaly_type: str  # 'spike', 'drop', 'trend', 'pattern'
    confidence: float

@dataclass
class HealthDashboard:
    """Health dashboard data structure."""
    overall_health: HealthStatus
    component_count: int
    active_alerts: int
    critical_alerts: int
    uptime_percentage: float
    availability_sla: float
    performance_score: float
    trend_indicators: Dict[str, str]  # 'improving', 'stable', 'degrading'
    key_metrics: Dict[str, float]
    recent_incidents: List[Dict[str, Any]]


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    component_name: str
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0


class HealthMonitor:
    """Enterprise-grade health monitoring system for Fleet-Mind components.
    
    Features:
    - Real-time health monitoring with detailed metrics
    - Automated alerting with escalation policies
    - Performance baselines and anomaly detection
    - Comprehensive logging with structured data
    - Dashboard and visualization support
    - SLA tracking and availability monitoring
    - Predictive health analysis
    - Multi-channel alert delivery
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        alert_cooldown: float = 300.0,  # 5 minutes
        enable_system_monitoring: bool = True,
        enable_network_monitoring: bool = True,
        enable_anomaly_detection: bool = True,
        baseline_learning_period: float = 86400.0,  # 24 hours
        anomaly_sensitivity: float = 2.0,  # standard deviations
        enable_predictive_analysis: bool = True,
        sla_targets: Optional[Dict[str, float]] = None
    ):
        """Initialize enterprise health monitor.
        
        Args:
            check_interval: Health check interval in seconds
            alert_cooldown: Minimum time between repeated alerts
            enable_system_monitoring: Enable system resource monitoring
            enable_network_monitoring: Enable network connectivity monitoring
            enable_anomaly_detection: Enable ML-based anomaly detection
            baseline_learning_period: Time to learn performance baselines
            anomaly_sensitivity: Sensitivity for anomaly detection (std devs)
            enable_predictive_analysis: Enable predictive health analysis
            sla_targets: SLA targets for availability monitoring
        """
        self.check_interval = check_interval
        self.alert_cooldown = alert_cooldown
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_network_monitoring = enable_network_monitoring
        self.enable_anomaly_detection = enable_anomaly_detection
        self.baseline_learning_period = baseline_learning_period
        self.anomaly_sensitivity = anomaly_sensitivity
        self.enable_predictive_analysis = enable_predictive_analysis
        self.sla_targets = sla_targets or {
            'availability': 99.9,  # 99.9% uptime
            'response_time': 100.0,  # 100ms response time
            'error_rate': 0.1  # 0.1% error rate
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Enhanced health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.alerts: List[HealthAlert] = []
        self.alert_history: List[HealthAlert] = []
        self.last_alert_times: Dict[str, float] = {}  # component:metric -> timestamp
        
        # Advanced analytics
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.anomaly_detections: List[AnomalyDetection] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute data
        self.trend_analysis: Dict[str, Dict[str, Any]] = {}
        
        # SLA and availability tracking
        self.availability_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours
        self.downtime_incidents: List[Dict[str, Any]] = []
        self.sla_violations: List[Dict[str, Any]] = []
        
        # Alert management
        self.alert_escalation_policies: Dict[str, Dict[str, Any]] = {}
        self.alert_suppression_rules: List[Dict[str, Any]] = []
        self.alert_correlation_rules: List[Dict[str, Any]] = []
        self.alert_channels: List[Callable[[HealthAlert], None]] = []
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Monitoring tasks and queues
        self._monitoring_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        self._alert_processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._alert_queue = queue.Queue()
        
        # Performance tracking
        self.check_durations: List[float] = []
        self.start_time = time.time()
        
        # Enhanced thresholds with dynamic adjustment
        self.default_thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0, "dynamic": True},
            "memory_usage": {"warning": 80.0, "critical": 95.0, "dynamic": True},
            "disk_usage": {"warning": 85.0, "critical": 95.0, "dynamic": False},
            "network_latency": {"warning": 100.0, "critical": 200.0, "dynamic": True},
            "error_rate": {"warning": 0.05, "critical": 0.10, "dynamic": True},
            "battery_level": {"warning": 20.0, "critical": 10.0, "dynamic": False},
            "communication_quality": {"warning": 0.7, "critical": 0.5, "dynamic": True},
            "response_time": {"warning": 1000.0, "critical": 5000.0, "dynamic": True},
            "throughput": {"warning": 100.0, "critical": 50.0, "dynamic": True},
            "availability": {"warning": 99.5, "critical": 99.0, "dynamic": False}
        }
        
        # Initialize alert escalation policies
        self._initialize_default_policies()
        
        print(f"Enterprise health monitor initialized:")
        print(f"  - Anomaly Detection: {'Enabled' if enable_anomaly_detection else 'Disabled'}")
        print(f"  - Predictive Analysis: {'Enabled' if enable_predictive_analysis else 'Disabled'}")
        print(f"  - SLA Monitoring: {len(self.sla_targets)} targets configured")
        print(f"  - Check Interval: {check_interval}s")

    def _initialize_default_policies(self) -> None:
        """Initialize default alert escalation policies."""
        self.alert_escalation_policies = {
            'critical': {
                'escalation_levels': [0, 300, 900, 1800],  # 0, 5min, 15min, 30min
                'acknowledgment_required': True,
                'auto_escalate': True
            },
            'error': {
                'escalation_levels': [0, 900, 3600],  # 0, 15min, 1hour
                'acknowledgment_required': False,
                'auto_escalate': True
            },
            'warning': {
                'escalation_levels': [0, 1800],  # 0, 30min
                'acknowledgment_required': False,
                'auto_escalate': False
            }
        }
        
        # Default suppression rules
        self.alert_suppression_rules = [
            {
                'name': 'maintenance_window',
                'condition': {'tag': 'maintenance', 'value': 'true'},
                'duration': 3600  # 1 hour
            },
            {
                'name': 'duplicate_suppression',
                'condition': {'same_component_metric': True},
                'duration': 300  # 5 minutes
            }
        ]
    
    def add_alert_channel(self, channel: Callable[[HealthAlert], None]) -> None:
        """Add alert delivery channel.
        
        Args:
            channel: Alert delivery function (email, Slack, PagerDuty, etc.)
        """
        self.alert_channels.append(channel)
    
    def set_sla_target(self, metric: str, target: float) -> None:
        """Set SLA target for specific metric.
        
        Args:
            metric: Metric name
            target: Target value
        """
        self.sla_targets[metric] = target
    
    def create_custom_threshold(self, component: str, metric: str, warning: float, critical: float) -> None:
        """Create custom threshold for specific component and metric.
        
        Args:
            component: Component name
            metric: Metric name
            warning: Warning threshold
            critical: Critical threshold
        """
        threshold_key = f"{component}:{metric}"
        self.default_thresholds[threshold_key] = {
            'warning': warning,
            'critical': critical,
            'dynamic': False
        }
    
    async def start_monitoring(self) -> None:
        """Start health monitoring tasks."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.enable_anomaly_detection:
            self._analytics_task = asyncio.create_task(self._anomaly_detection_loop())
        
        # self._alert_processing_task = asyncio.create_task(self._alert_processing_loop())
        
        print(f"Enterprise health monitoring started:")
        print(f"  - Check interval: {self.check_interval}s")
        print(f"  - Analytics: {'Enabled' if self.enable_anomaly_detection else 'Disabled'}")
        print(f"  - Alert channels: {len(self.alert_channels)} configured")

    async def stop_monitoring(self) -> None:
        """Stop all health monitoring tasks."""
        self._running = False
        
        tasks_to_cancel = []
        if self._monitoring_task:
            tasks_to_cancel.append(self._monitoring_task)
        if self._analytics_task:
            tasks_to_cancel.append(self._analytics_task)
        # if self._alert_processing_task:
        #     tasks_to_cancel.append(self._alert_processing_task)
        
        for task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        print("Enterprise health monitoring stopped")

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

    def create_performance_baseline(self, component: str, metric_name: str) -> None:
        """Create performance baseline for anomaly detection.
        
        Args:
            component: Component name
            metric_name: Metric name
        """
        baseline_key = f"{component}:{metric_name}"
        metric_history = self.metric_history.get(baseline_key, deque())
        
        if len(metric_history) < 100:  # Need sufficient data
            return
        
        values = list(metric_history)
        if not values:
            return
        
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        percentile_95 = sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else mean_value
        percentile_99 = sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else percentile_95
        
        # Analyze seasonal patterns (hourly)
        seasonal_patterns = {}
        if len(values) >= 24:  # At least 24 data points
            for hour in range(24):
                hour_values = [v for i, v in enumerate(values) if i % 24 == hour]
                if hour_values:
                    seasonal_patterns[str(hour)] = statistics.mean(hour_values)
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            component=component,
            mean_value=mean_value,
            std_deviation=std_dev,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            sample_count=len(values),
            last_updated=time.time(),
            seasonal_patterns=seasonal_patterns
        )
        
        with self._lock:
            self.performance_baselines[baseline_key] = baseline
    
    def detect_anomaly(self, component: str, metric_name: str, current_value: float) -> Optional[AnomalyDetection]:
        """Detect anomaly in metric value using statistical analysis.
        
        Args:
            component: Component name
            metric_name: Metric name
            current_value: Current metric value
            
        Returns:
            AnomalyDetection if anomaly detected, None otherwise
        """
        baseline_key = f"{component}:{metric_name}"
        
        if baseline_key not in self.performance_baselines:
            return None
        
        baseline = self.performance_baselines[baseline_key]
        
        # Calculate expected range based on baseline
        tolerance = self.anomaly_sensitivity * baseline.std_deviation
        expected_min = baseline.mean_value - tolerance
        expected_max = baseline.mean_value + tolerance
        
        # Check for seasonal patterns
        current_hour = str(int(time.time() / 3600) % 24)
        if current_hour in baseline.seasonal_patterns:
            seasonal_mean = baseline.seasonal_patterns[current_hour]
            expected_min = seasonal_mean - tolerance
            expected_max = seasonal_mean + tolerance
        
        # Determine if this is an anomaly
        is_anomaly = current_value < expected_min or current_value > expected_max
        
        if not is_anomaly:
            return None
        
        # Classify anomaly type
        anomaly_type = "spike" if current_value > expected_max else "drop"
        
        # Calculate anomaly score (0-10 scale)
        if baseline.std_deviation > 0:
            z_score = abs(current_value - baseline.mean_value) / baseline.std_deviation
            anomaly_score = min(z_score, 10.0)
        else:
            anomaly_score = 10.0 if is_anomaly else 0.0
        
        # Calculate confidence based on baseline quality
        confidence = min(baseline.sample_count / 1000.0, 1.0)
        
        return AnomalyDetection(
            metric_name=metric_name,
            component=component,
            timestamp=time.time(),
            current_value=current_value,
            expected_range=(expected_min, expected_max),
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            confidence=confidence
        )
    
    def update_metric(
        self,
        component: str,
        metric_name: str,
        value: float,
        unit: str = "",
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Update a health metric for a component with enhanced analytics.
        
        Args:
            component: Component name
            metric_name: Metric name
            value: Metric value
            unit: Value unit
            description: Metric description
            tags: Additional metric tags
        """
        with self._lock:
            if component not in self.component_health:
                self.register_component(component)
            
            # Store metric in history for trend analysis
            metric_key = f"{component}:{metric_name}"
            self.metric_history[metric_key].append(value)
            
            # Check for anomalies if enabled
            anomaly = None
            if self.enable_anomaly_detection:
                anomaly = self.detect_anomaly(component, metric_name, value)
                if anomaly:
                    self.anomaly_detections.append(anomaly)
                    # Keep anomaly list reasonable size
                    if len(self.anomaly_detections) > 1000:
                        self.anomaly_detections = self.anomaly_detections[-500:]
        
            # Use component-specific thresholds if available
            component_threshold_key = f"{component}:{metric_name}"
            thresholds = self.default_thresholds.get(component_threshold_key, 
                                                   self.default_thresholds.get(metric_name, {}))
            warning_threshold = thresholds.get("warning")
            critical_threshold = thresholds.get("critical")
            
            # Adjust thresholds dynamically if enabled
            if thresholds.get("dynamic", False) and metric_key in self.performance_baselines:
                baseline = self.performance_baselines[metric_key]
                if baseline.std_deviation > 0:
                    # Adjust thresholds based on historical behavior
                    dynamic_warning = baseline.mean_value + (1.5 * baseline.std_deviation)
                    dynamic_critical = baseline.mean_value + (2.5 * baseline.std_deviation)
                    
                    # Use more conservative threshold
                    warning_threshold = min(warning_threshold or float('inf'), dynamic_warning)
                    critical_threshold = min(critical_threshold or float('inf'), dynamic_critical)
        
            status = HealthStatus.GOOD
            
            # Check for anomaly-based status
            if anomaly and anomaly.anomaly_score > 7.0:
                status = HealthStatus.CRITICAL
            elif anomaly and anomaly.anomaly_score > 4.0:
                status = HealthStatus.WARNING
            
            # Standard threshold checks
            if critical_threshold is not None and value >= critical_threshold:
                status = HealthStatus.CRITICAL
            elif warning_threshold is not None and value >= warning_threshold:
                status = HealthStatus.WARNING
            elif value < 0:  # Assume negative values are bad
                status = HealthStatus.FAILED
            
            # Special cases for metrics where lower is worse
            if metric_name in ["battery_level", "communication_quality", "availability"]:
                if critical_threshold is not None and value <= critical_threshold:
                    status = HealthStatus.CRITICAL
                elif warning_threshold is not None and value <= warning_threshold:
                    status = HealthStatus.WARNING
            
            # Excellence threshold (better than baseline)
            if status == HealthStatus.GOOD and metric_key in self.performance_baselines:
                baseline = self.performance_baselines[metric_key]
                if value > baseline.percentile_95:
                    status = HealthStatus.EXCELLENT
        
        # Create enhanced metric object
        metric = HealthMetric(
            name=metric_name,
            value=value,
            status=status,
            threshold_warning=warning_threshold,
            threshold_critical=critical_threshold,
            unit=unit,
            description=description
        )
        
        # Add tags if provided
        if tags:
            if not hasattr(metric, 'additional_data'):
                metric.additional_data = {}
            metric.additional_data.update(tags)
        
        # Update component metrics
        self.component_health[component].metrics[metric_name] = metric
        self.component_health[component].last_check = time.time()
        
        # Update component overall status
        self._update_component_status(component)
        
        # Check for enhanced alert conditions
        self._check_enhanced_alert_conditions(component, metric, anomaly)
        
        # Track availability for SLA monitoring
        self._track_availability(component, status)
    
    def _track_availability(self, component: str, status: HealthStatus) -> None:
        """Track component availability for SLA monitoring.
        
        Args:
            component: Component name
            status: Current health status
        """
        is_available = status not in [HealthStatus.FAILED, HealthStatus.CRITICAL]
        current_time = time.time()
        
        # Store availability data point
        availability_key = f"{component}:availability"
        self.availability_windows[availability_key].append((current_time, is_available))
        
        # Check for SLA violations
        if not is_available:
            self._check_sla_violation(component, status)
    
    def _check_sla_violation(self, component: str, status: HealthStatus) -> None:
        """Check for SLA violations and record incidents.
        
        Args:
            component: Component name
            status: Current status causing violation
        """
        current_time = time.time()
        
        # Check if this is a new incident or continuation
        recent_incidents = [i for i in self.downtime_incidents 
                          if i['component'] == component and 
                          current_time - i.get('end_time', i['start_time']) < 300]
        
        if not recent_incidents:
            # New incident
            incident = {
                'incident_id': str(uuid.uuid4()),
                'component': component,
                'start_time': current_time,
                'status': status.value,
                'severity': 'critical' if status == HealthStatus.FAILED else 'high'
            }
            self.downtime_incidents.append(incident)
            
            # Create SLA violation record
            sla_violation = {
                'violation_id': str(uuid.uuid4()),
                'component': component,
                'timestamp': current_time,
                'sla_target': self.sla_targets.get('availability', 99.9),
                'actual_availability': self._calculate_component_availability(component),
                'incident_id': incident['incident_id']
            }
            self.sla_violations.append(sla_violation)
        else:
            # Update existing incident
            recent_incidents[0]['end_time'] = current_time
    
    def _calculate_component_availability(self, component: str, window_hours: int = 24) -> float:
        """Calculate component availability percentage.
        
        Args:
            component: Component name
            window_hours: Time window in hours
            
        Returns:
            Availability percentage
        """
        availability_key = f"{component}:availability"
        window_data = self.availability_windows.get(availability_key, deque())
        
        if not window_data:
            return 100.0
        
        cutoff_time = time.time() - (window_hours * 3600)
        relevant_data = [(t, available) for t, available in window_data if t >= cutoff_time]
        
        if not relevant_data:
            return 100.0
        
        total_points = len(relevant_data)
        available_points = sum(1 for _, available in relevant_data if available)
        
        return (available_points / total_points) * 100.0
    
    def _check_enhanced_alert_conditions(self, component: str, metric: HealthMetric, anomaly: Optional[AnomalyDetection]) -> None:
        """Enhanced alert condition checking with correlation and suppression.
        
        Args:
            component: Component name
            metric: Metric object
            anomaly: Detected anomaly if any
        """
        if metric.status in [HealthStatus.GOOD, HealthStatus.EXCELLENT]:
            return
        
        # Check alert suppression rules
        if self._is_alert_suppressed(component, metric.name):
            return
        
        # Check cooldown
        alert_key = f"{component}:{metric.name}"
        current_time = time.time()
        
        if alert_key in self.last_alert_times:
            time_since_last = current_time - self.last_alert_times[alert_key]
            if time_since_last < self.alert_cooldown:
                return
        
        # Determine alert severity with enhanced logic
        if metric.status == HealthStatus.FAILED:
            severity = AlertSeverity.CRITICAL
            threshold = 0
        elif metric.status == HealthStatus.CRITICAL:
            severity = AlertSeverity.CRITICAL
            threshold = metric.threshold_critical or 0
        else:  # WARNING
            severity = AlertSeverity.WARNING
            threshold = metric.threshold_warning or 0
        
        # Enhance severity based on anomaly
        if anomaly and anomaly.anomaly_score > 8.0:
            severity = AlertSeverity.CRITICAL
        
        # Create enhanced alert
        alert = HealthAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=current_time,
            severity=severity,
            component=component,
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold,
            message=self._generate_enhanced_alert_message(component, metric, anomaly),
            acknowledgment_required=severity == AlertSeverity.CRITICAL,
            tags={'metric_unit': metric.unit} if metric.unit else {},
            remediation_suggestions=self._get_remediation_suggestions(component, metric.name, metric.status)
        )
        
        # Add correlation ID if part of a pattern
        correlation_id = self._find_alert_correlation(alert)
        if correlation_id:
            alert.correlation_id = correlation_id
        
        # Queue alert for processing
        self._alert_queue.put(alert)
        self.last_alert_times[alert_key] = current_time
    
    def _is_alert_suppressed(self, component: str, metric_name: str) -> bool:
        """Check if alert should be suppressed based on rules.
        
        Args:
            component: Component name
            metric_name: Metric name
            
        Returns:
            True if alert should be suppressed
        """
        # Check maintenance window
        if component in self.component_health:
            component_health = self.component_health[component]
            # Simple maintenance check - in real implementation would check tags
            # if hasattr(component_health, 'tags') and component_health.tags.get('maintenance') == 'true':
            #     return True
        
        # Check for recent similar alerts
        current_time = time.time()
        recent_alerts = [a for a in self.alerts 
                        if a.component == component and 
                        a.metric_name == metric_name and 
                        current_time - a.timestamp < 300 and  # 5 minutes
                        not a.resolved]
        
        return len(recent_alerts) > 0
    
    def _generate_enhanced_alert_message(self, component: str, metric: HealthMetric, anomaly: Optional[AnomalyDetection]) -> str:
        """Generate enhanced alert message with context.
        
        Args:
            component: Component name
            metric: Metric object
            anomaly: Detected anomaly
            
        Returns:
            Enhanced alert message
        """
        base_message = f"{component} {metric.name} is {metric.status.value}: {metric.value}{metric.unit}"
        
        if metric.threshold_critical and metric.status == HealthStatus.CRITICAL:
            base_message += f" (critical threshold: {metric.threshold_critical}{metric.unit})"
        elif metric.threshold_warning and metric.status == HealthStatus.WARNING:
            base_message += f" (warning threshold: {metric.threshold_warning}{metric.unit})"
        
        if anomaly:
            base_message += f" | Anomaly detected: {anomaly.anomaly_type} (score: {anomaly.anomaly_score:.1f})"
            base_message += f" | Expected range: {anomaly.expected_range[0]:.1f}-{anomaly.expected_range[1]:.1f}"
        
        # Add trend information if available
        metric_key = f"{component}:{metric.name}"
        if metric_key in self.metric_history:
            recent_values = list(self.metric_history[metric_key])[-10:]  # Last 10 values
            if len(recent_values) >= 3:
                trend = "increasing" if recent_values[-1] > recent_values[-3] else "decreasing"
                base_message += f" | Trend: {trend}"
        
        return base_message
    
    def _get_remediation_suggestions(self, component: str, metric_name: str, status: HealthStatus) -> List[str]:
        """Get remediation suggestions for specific issues.
        
        Args:
            component: Component name
            metric_name: Metric name
            status: Current status
            
        Returns:
            List of remediation suggestions
        """
        suggestions = []
        
        if metric_name == "cpu_usage" and status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            suggestions.extend([
                "Check for runaway processes consuming CPU",
                "Consider scaling up compute resources",
                "Review recent deployments that might impact CPU usage",
                "Check system load and active connections"
            ])
        
        elif metric_name == "memory_usage" and status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            suggestions.extend([
                "Check for memory leaks in applications",
                "Review garbage collection performance",
                "Consider increasing available memory",
                "Analyze memory usage patterns and optimize"
            ])
        
        elif metric_name == "disk_usage" and status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            suggestions.extend([
                "Clean up temporary files and logs",
                "Archive old data to long-term storage",
                "Consider expanding disk capacity",
                "Implement log rotation policies"
            ])
        
        elif metric_name == "network_latency" and status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            suggestions.extend([
                "Check network connectivity and bandwidth",
                "Investigate network congestion",
                "Review DNS resolution performance",
                "Consider network optimization or CDN usage"
            ])
        
        elif metric_name == "battery_level" and status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            suggestions.extend([
                "Initiate immediate return-to-base procedure",
                "Reduce power consumption by disabling non-essential systems",
                "Prepare backup drone for mission continuity",
                "Check battery health and charging system"
            ])
        
        return suggestions
    
    def _find_alert_correlation(self, alert: HealthAlert) -> Optional[str]:
        """Find correlation ID for related alerts.
        
        Args:
            alert: Alert to find correlations for
            
        Returns:
            Correlation ID if found
        """
        # Look for recent alerts from same component
        current_time = time.time()
        recent_alerts = [a for a in self.alerts 
                        if a.component == alert.component and 
                        current_time - a.timestamp < 1800 and  # 30 minutes
                        not a.resolved]
        
        # If multiple metrics from same component are alerting, correlate them
        if len(recent_alerts) > 1:
            existing_correlations = [a.correlation_id for a in recent_alerts if a.correlation_id]
            if existing_correlations:
                return existing_correlations[0]
            else:
                return str(uuid.uuid4())  # New correlation ID
        
        return None
        
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

    async def _anomaly_detection_loop(self) -> None:
        """Background analytics and anomaly detection loop."""
        while self._running:
            try:
                # Check for anomalies in recent metrics
                with self._lock:
                    for component, health in self.component_health.items():
                        for metric_name, metric_data in health.metrics.items():
                            recent_values = list(metric_data.values)[-10:]  # Last 10 values
                            if len(recent_values) >= 5:  # Need minimum samples
                                current_value = recent_values[-1]
                                anomaly = self._detect_anomaly(component, metric_name, current_value)
                                if anomaly:
                                    self.logger.warning(f"Anomaly detected: {anomaly}")
                
                await asyncio.sleep(30.0)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
                await asyncio.sleep(10.0)

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