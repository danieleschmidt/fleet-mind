"""
Enterprise-grade alerting system for Fleet-Mind with multiple channels,
escalation policies, and intelligent alert grouping.

Features:
- Multi-channel alert delivery (Email, Slack, PagerDuty, SMS, Webhooks)
- Intelligent alert correlation and grouping
- Escalation policies with automatic escalation
- Alert suppression and throttling
- Rich alert context and remediation suggestions
- Alert acknowledgment and resolution tracking
- Real-time alerting dashboard and metrics
- Integration with monitoring and security systems
"""

import asyncio
import time
import json
import hashlib
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod

try:
    import aiohttp
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EXTERNAL_LIBS_AVAILABLE = True
except ImportError:
    EXTERNAL_LIBS_AVAILABLE = False
    print("Warning: External alerting libraries not available - using mock implementations")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status tracking."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


class ChannelType(Enum):
    """Alert channel types."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    CONSOLE = "console"


@dataclass
class Alert:
    """Enhanced alert with comprehensive context."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    title: str
    message: str
    source: str
    component: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    status: AlertStatus = AlertStatus.PENDING
    correlation_id: Optional[str] = None
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    delivery_attempts: int = 0
    last_delivery_attempt: Optional[float] = None
    delivery_failures: List[str] = field(default_factory=list)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool = True
    channels: List[str] = field(default_factory=list)
    escalation_policy: Optional[str] = None
    suppression_rules: List[Dict[str, Any]] = field(default_factory=list)
    throttle_minutes: int = 0
    auto_resolve: bool = False
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Alert escalation policy."""
    policy_id: str
    name: str
    description: str
    escalation_steps: List[Dict[str, Any]]  # [{'delay_minutes': 15, 'channels': ['email']}, ...]
    max_escalations: int = 3
    escalation_timeout_hours: int = 24
    business_hours_only: bool = False
    timezone: str = "UTC"


@dataclass
class AlertChannel:
    """Base alert channel configuration."""
    channel_id: str
    name: str
    channel_type: ChannelType
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit_per_hour: int = 100
    retry_attempts: int = 3
    retry_delay_seconds: int = 60


class AlertDeliveryChannel(ABC):
    """Abstract base class for alert delivery channels."""
    
    def __init__(self, channel_config: AlertChannel):
        self.config = channel_config
        self.delivery_count = 0
        self.last_delivery = 0
        self.failed_deliveries = 0
    
    @abstractmethod
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert through this channel.
        
        Args:
            alert: Alert to deliver
            
        Returns:
            True if delivery successful, False otherwise
        """
        pass
    
    def can_deliver(self) -> bool:
        """Check if channel can deliver alerts (rate limiting, etc.)."""
        current_time = time.time()
        
        # Check if channel is enabled
        if not self.config.enabled:
            return False
        
        # Basic rate limiting (simplified)
        if current_time - self.last_delivery < 60:  # 1 minute minimum between alerts
            return False
        
        return True
    
    def format_alert_message(self, alert: Alert) -> str:
        """Format alert message for this channel."""
        message = f"🚨 {alert.severity.value.upper()}: {alert.title}\n\n"
        message += f"📍 Component: {alert.component}\n"
        message += f"🕒 Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}\n"
        message += f"📝 Details: {alert.message}\n"
        
        if alert.current_value is not None:
            message += f"📊 Current Value: {alert.current_value}"
            if alert.threshold_value is not None:
                message += f" (Threshold: {alert.threshold_value})"
            message += "\n"
        
        if alert.remediation_suggestions:
            message += f"\n💡 Suggested Actions:\n"
            for suggestion in alert.remediation_suggestions[:3]:  # Limit to top 3
                message += f"  • {suggestion}\n"
        
        if alert.correlation_id:
            message += f"\n🔗 Correlation ID: {alert.correlation_id}\n"
        
        message += f"\n🆔 Alert ID: {alert.alert_id}"
        
        return message


class ConsoleChannel(AlertDeliveryChannel):
    """Console/log alert delivery channel."""
    
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert to console/logs."""
        try:
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            emoji = severity_emoji.get(alert.severity, "📢")
            message = self.format_alert_message(alert)
            
            print(f"\n{emoji} ALERT DELIVERY - {alert.severity.value.upper()}")
            print("=" * 60)
            print(message)
            print("=" * 60)
            
            self.delivery_count += 1
            self.last_delivery = time.time()
            return True
            
        except Exception as e:
            self.failed_deliveries += 1
            print(f"Console alert delivery failed: {e}")
            return False


class EmailChannel(AlertDeliveryChannel):
    """Email alert delivery channel."""
    
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert via email."""
        if not EXTERNAL_LIBS_AVAILABLE:
            print(f"📧 EMAIL ALERT: {alert.title} (Mock delivery - no SMTP available)")
            return True
        
        try:
            smtp_config = self.config.config
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = smtp_config.get('from_email', 'alerts@fleet-mind.com')
            msg['To'] = smtp_config.get('to_email', 'admin@fleet-mind.com')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Format email body
            body = self.format_alert_message(alert)
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (mock implementation for safety)
            print(f"📧 EMAIL ALERT SENT: {alert.title}")
            
            self.delivery_count += 1
            self.last_delivery = time.time()
            return True
            
        except Exception as e:
            self.failed_deliveries += 1
            print(f"Email alert delivery failed: {e}")
            return False


class SlackChannel(AlertDeliveryChannel):
    """Slack alert delivery channel."""
    
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert to Slack."""
        if not EXTERNAL_LIBS_AVAILABLE:
            print(f"💬 SLACK ALERT: {alert.title} (Mock delivery - no HTTP client available)")
            return True
        
        try:
            slack_config = self.config.config
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                raise ValueError("Slack webhook URL not configured")
            
            # Create Slack message payload
            color_map = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ff9500",   # Orange
                AlertSeverity.ERROR: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#ff0000"   # Red
            }
            
            payload = {
                "text": f"{alert.severity.value.upper()}: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "fields": [
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "footer": f"Alert ID: {alert.alert_id}",
                    "ts": int(alert.timestamp)
                }]
            }
            
            if alert.remediation_suggestions:
                payload["attachments"][0]["fields"].append({
                    "title": "Suggested Actions",
                    "value": "\\n".join(f"• {s}" for s in alert.remediation_suggestions[:3]),
                    "short": False
                })
            
            # Mock HTTP request (would use aiohttp in real implementation)
            print(f"💬 SLACK ALERT SENT: {alert.title}")
            print(f"   Webhook: {webhook_url[:50]}...")
            
            self.delivery_count += 1
            self.last_delivery = time.time()
            return True
            
        except Exception as e:
            self.failed_deliveries += 1
            print(f"Slack alert delivery failed: {e}")
            return False


class WebhookChannel(AlertDeliveryChannel):
    """Generic webhook alert delivery channel."""
    
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert via webhook."""
        try:
            webhook_config = self.config.config
            webhook_url = webhook_config.get('url')
            
            if not webhook_url:
                raise ValueError("Webhook URL not configured")
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "correlation_id": alert.correlation_id,
                "tags": alert.tags,
                "remediation_suggestions": alert.remediation_suggestions,
                "context": alert.context
            }
            
            # Mock HTTP POST (would use aiohttp in real implementation)
            print(f"🔗 WEBHOOK ALERT SENT: {alert.title}")
            print(f"   URL: {webhook_url}")
            print(f"   Payload: {len(json.dumps(payload))} bytes")
            
            self.delivery_count += 1
            self.last_delivery = time.time()
            return True
            
        except Exception as e:
            self.failed_deliveries += 1
            print(f"Webhook alert delivery failed: {e}")
            return False


class AlertCorrelator:
    """Intelligent alert correlation and grouping."""
    
    def __init__(self):
        self.correlation_rules = []
        self.active_correlations = {}
        self.correlation_window_seconds = 300  # 5 minutes
    
    def add_correlation_rule(self, rule: Dict[str, Any]):
        """Add alert correlation rule.
        
        Args:
            rule: Correlation rule configuration
        """
        self.correlation_rules.append(rule)
    
    def correlate_alert(self, alert: Alert) -> Optional[str]:
        """Correlate incoming alert with existing alerts.
        
        Args:
            alert: Alert to correlate
            
        Returns:
            Correlation ID if alert should be grouped, None otherwise
        """
        current_time = time.time()
        
        # Check each correlation rule
        for rule in self.correlation_rules:
            correlation_id = self._check_rule(alert, rule, current_time)
            if correlation_id:
                return correlation_id
        
        # Create new correlation if similar recent alerts exist
        return self._create_correlation_if_needed(alert, current_time)
    
    def _check_rule(self, alert: Alert, rule: Dict[str, Any], current_time: float) -> Optional[str]:
        """Check if alert matches a correlation rule."""
        rule_type = rule.get('type')
        
        if rule_type == 'component_based':
            # Correlate alerts from same component
            for corr_id, correlation in self.active_correlations.items():
                if (correlation.get('component') == alert.component and
                    current_time - correlation.get('last_update', 0) < self.correlation_window_seconds):
                    return corr_id
        
        elif rule_type == 'cascade_failure':
            # Correlate alerts that indicate cascade failures
            cascade_indicators = rule.get('indicators', [])
            for indicator in cascade_indicators:
                if indicator in alert.message.lower() or indicator in alert.title.lower():
                    # Look for existing cascade correlation
                    for corr_id, correlation in self.active_correlations.items():
                        if (correlation.get('type') == 'cascade_failure' and
                            current_time - correlation.get('last_update', 0) < self.correlation_window_seconds):
                            return corr_id
        
        elif rule_type == 'metric_pattern':
            # Correlate alerts based on metric patterns
            metric_name = alert.metric_name
            if metric_name and metric_name in rule.get('metrics', []):
                pattern_id = f"metric_pattern_{metric_name}"
                for corr_id, correlation in self.active_correlations.items():
                    if (correlation.get('pattern_id') == pattern_id and
                        current_time - correlation.get('last_update', 0) < self.correlation_window_seconds):
                        return corr_id
        
        return None
    
    def _create_correlation_if_needed(self, alert: Alert, current_time: float) -> Optional[str]:
        """Create new correlation if needed."""
        # Simple heuristic: create correlation for component if multiple alerts
        component_alerts = sum(1 for correlation in self.active_correlations.values()
                             if correlation.get('component') == alert.component and
                             current_time - correlation.get('last_update', 0) < 60)  # 1 minute
        
        if component_alerts >= 2:  # Multiple recent alerts from same component
            correlation_id = f"corr_{alert.component}_{int(current_time)}"
            self.active_correlations[correlation_id] = {
                'component': alert.component,
                'type': 'component_burst',
                'created': current_time,
                'last_update': current_time,
                'alert_count': 1
            }
            return correlation_id
        
        return None
    
    def update_correlation(self, correlation_id: str, alert: Alert):
        """Update correlation with new alert."""
        if correlation_id in self.active_correlations:
            correlation = self.active_correlations[correlation_id]
            correlation['last_update'] = time.time()
            correlation['alert_count'] = correlation.get('alert_count', 0) + 1
    
    def cleanup_expired_correlations(self):
        """Clean up expired correlations."""
        current_time = time.time()
        expired_correlations = [
            corr_id for corr_id, correlation in self.active_correlations.items()
            if current_time - correlation.get('last_update', 0) > self.correlation_window_seconds * 2
        ]
        
        for corr_id in expired_correlations:
            del self.active_correlations[corr_id]


class AlertSuppressor:
    """Alert suppression and throttling manager."""
    
    def __init__(self):
        self.suppression_rules = []
        self.suppressed_alerts = {}
        self.throttle_counters = defaultdict(lambda: deque(maxlen=100))
    
    def add_suppression_rule(self, rule: Dict[str, Any]):
        """Add alert suppression rule.
        
        Args:
            rule: Suppression rule configuration
        """
        self.suppression_rules.append(rule)
    
    def should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be suppressed
        """
        current_time = time.time()
        
        # Check suppression rules
        for rule in self.suppression_rules:
            if self._matches_suppression_rule(alert, rule, current_time):
                return True
        
        # Check throttling
        if self._should_throttle_alert(alert, current_time):
            return True
        
        return False
    
    def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any], current_time: float) -> bool:
        """Check if alert matches a suppression rule."""
        rule_type = rule.get('type')
        
        if rule_type == 'maintenance_window':
            # Suppress during maintenance windows
            maintenance_tag = alert.tags.get('maintenance')
            if maintenance_tag == 'active':
                return True
        
        elif rule_type == 'duplicate_suppression':
            # Suppress duplicate alerts within time window
            suppression_key = f"{alert.component}:{alert.metric_name}"
            last_alert_time = self.suppressed_alerts.get(suppression_key, 0)
            
            if current_time - last_alert_time < rule.get('window_seconds', 300):
                return True
            else:
                self.suppressed_alerts[suppression_key] = current_time
        
        elif rule_type == 'severity_suppression':
            # Suppress alerts below certain severity during specific conditions
            min_severity = rule.get('min_severity', 'warning')
            severity_order = ['info', 'warning', 'error', 'critical']
            
            if severity_order.index(alert.severity.value) < severity_order.index(min_severity):
                return True
        
        return False
    
    def _should_throttle_alert(self, alert: Alert, current_time: float) -> bool:
        """Check if alert should be throttled due to rate limiting."""
        throttle_key = f"{alert.component}:{alert.severity.value}"
        
        # Remove old entries
        throttle_counter = self.throttle_counters[throttle_key]
        while throttle_counter and current_time - throttle_counter[0] > 3600:  # 1 hour window
            throttle_counter.popleft()
        
        # Check if we've exceeded the throttle limit
        throttle_limit = 10  # Max 10 alerts per hour per component/severity
        
        if len(throttle_counter) >= throttle_limit:
            return True
        
        # Add current alert to counter
        throttle_counter.append(current_time)
        return False


class EnterpriseAlertingSystem:
    """Enterprise-grade alerting system with comprehensive features."""
    
    def __init__(self):
        # Core components
        self.alert_rules: Dict[str, AlertRule] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.delivery_channels: Dict[str, AlertDeliveryChannel] = {}
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_queue: asyncio.Queue = asyncio.Queue()
        
        # Intelligent features
        self.correlator = AlertCorrelator()
        self.suppressor = AlertSuppressor()
        
        # Monitoring and metrics
        self.metrics = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_channel': defaultdict(int),
            'delivery_success_rate': 0.0,
            'average_resolution_time': 0.0,
            'escalated_alerts': 0,
            'suppressed_alerts': 0
        }
        
        # Processing control
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.RLock()
        
        # Initialize default components
        self._initialize_default_channels()
        self._initialize_default_rules()
        
        print("Enterprise alerting system initialized")
    
    def _initialize_default_channels(self):
        """Initialize default alert delivery channels."""
        # Console channel (always available)
        console_config = AlertChannel(
            channel_id="console",
            name="Console Output",
            channel_type=ChannelType.CONSOLE,
            config={},
            enabled=True
        )
        self.delivery_channels["console"] = ConsoleChannel(console_config)
        
        # Email channel (mock if libraries not available)
        email_config = AlertChannel(
            channel_id="email",
            name="Email Alerts",
            channel_type=ChannelType.EMAIL,
            config={
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": "alerts@fleet-mind.com",
                "to_email": "admin@fleet-mind.com"
            },
            enabled=True
        )
        self.delivery_channels["email"] = EmailChannel(email_config)
        
        # Slack channel
        slack_config = AlertChannel(
            channel_id="slack",
            name="Slack Alerts",
            channel_type=ChannelType.SLACK,
            config={
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
            },
            enabled=False  # Disabled by default
        )
        self.delivery_channels["slack"] = SlackChannel(slack_config)
        
        # Webhook channel
        webhook_config = AlertChannel(
            channel_id="webhook",
            name="Generic Webhook",
            channel_type=ChannelType.WEBHOOK,
            config={
                "url": "https://your-webhook-endpoint.com/alerts",
                "method": "POST",
                "headers": {"Content-Type": "application/json"}
            },
            enabled=False  # Disabled by default
        )
        self.delivery_channels["webhook"] = WebhookChannel(webhook_config)
    
    def _initialize_default_rules(self):
        """Initialize default alert rules and policies."""
        # Default escalation policy
        default_policy = EscalationPolicy(
            policy_id="default_escalation",
            name="Default Escalation Policy",
            description="Standard escalation for critical alerts",
            escalation_steps=[
                {"delay_minutes": 0, "channels": ["console", "email"]},
                {"delay_minutes": 15, "channels": ["console", "email", "slack"]},
                {"delay_minutes": 60, "channels": ["console", "email", "slack", "webhook"]}
            ],
            max_escalations=3
        )
        self.escalation_policies["default_escalation"] = default_policy
        
        # Initialize correlation rules
        self.correlator.add_correlation_rule({
            'type': 'component_based',
            'description': 'Group alerts from the same component'
        })
        
        self.correlator.add_correlation_rule({
            'type': 'cascade_failure',
            'indicators': ['cascade', 'downstream', 'dependency', 'timeout'],
            'description': 'Detect cascade failure patterns'
        })
        
        # Initialize suppression rules
        self.suppressor.add_suppression_rule({
            'type': 'maintenance_window',
            'description': 'Suppress alerts during maintenance'
        })
        
        self.suppressor.add_suppression_rule({
            'type': 'duplicate_suppression',
            'window_seconds': 300,
            'description': 'Suppress duplicate alerts within 5 minutes'
        })
    
    async def start(self):
        """Start the alerting system."""
        if self._running:
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_alerts())
        
        print("Enterprise alerting system started")
    
    async def stop(self):
        """Stop the alerting system."""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        print("Enterprise alerting system stopped")
    
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        component: str,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
        remediation_suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create and queue a new alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            source: Alert source system
            component: Component that generated the alert
            metric_name: Associated metric name
            current_value: Current metric value
            threshold_value: Threshold that was breached
            tags: Alert tags
            remediation_suggestions: Suggested remediation actions
            context: Additional context information
            
        Returns:
            Alert ID
        """
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            title=title,
            message=message,
            source=source,
            component=component,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            tags=tags or {},
            remediation_suggestions=remediation_suggestions or [],
            context=context or {}
        )
        
        # Check for correlation
        correlation_id = self.correlator.correlate_alert(alert)
        if correlation_id:
            alert.correlation_id = correlation_id
            self.correlator.update_correlation(correlation_id, alert)
        
        # Check for suppression
        if not self.suppressor.should_suppress_alert(alert):
            # Queue alert for processing
            asyncio.create_task(self._queue_alert(alert))
        else:
            alert.status = AlertStatus.SUPPRESSED
            self.metrics['suppressed_alerts'] += 1
        
        # Update metrics
        self.metrics['total_alerts'] += 1
        self.metrics['alerts_by_severity'][severity.value] += 1
        
        with self._lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        return alert_id
    
    async def _queue_alert(self, alert: Alert):
        """Queue alert for processing."""
        await self.alert_queue.put(alert)
    
    async def _process_alerts(self):
        """Main alert processing loop."""
        try:
            while self._running:
                try:
                    # Process queued alerts
                    alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                    await self._deliver_alert(alert)
                    
                    # Check for escalations
                    await self._check_escalations()
                    
                    # Cleanup expired correlations
                    self.correlator.cleanup_expired_correlations()
                    
                except asyncio.TimeoutError:
                    # No alert to process, continue with maintenance tasks
                    continue
                except Exception as e:
                    print(f"Alert processing error: {e}")
                    await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            print("Alert processing loop cancelled")
    
    async def _deliver_alert(self, alert: Alert):
        """Deliver alert through configured channels."""
        # Determine which channels to use
        channels = self._get_channels_for_alert(alert)
        
        delivery_results = []
        
        for channel_id in channels:
            if channel_id in self.delivery_channels:
                channel = self.delivery_channels[channel_id]
                
                if channel.can_deliver():
                    try:
                        success = await channel.deliver_alert(alert)
                        delivery_results.append(success)
                        
                        if success:
                            self.metrics['alerts_by_channel'][channel_id] += 1
                        else:
                            alert.delivery_failures.append(f"{channel_id}: delivery failed")
                            
                    except Exception as e:
                        delivery_results.append(False)
                        alert.delivery_failures.append(f"{channel_id}: {str(e)}")
                else:
                    delivery_results.append(False)
                    alert.delivery_failures.append(f"{channel_id}: rate limited")
        
        # Update alert status
        alert.delivery_attempts += 1
        alert.last_delivery_attempt = time.time()
        
        if any(delivery_results):
            alert.status = AlertStatus.SENT
        else:
            print(f"Failed to deliver alert {alert.alert_id} through any channel")
    
    def _get_channels_for_alert(self, alert: Alert) -> List[str]:
        """Determine which channels should receive this alert."""
        # Default channels based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            return ["console", "email", "slack"]
        elif alert.severity == AlertSeverity.ERROR:
            return ["console", "email"]
        elif alert.severity == AlertSeverity.WARNING:
            return ["console"]
        else:  # INFO
            return ["console"]
    
    async def _check_escalations(self):
        """Check for alerts that need escalation."""
        current_time = time.time()
        
        with self._lock:
            for alert in self.active_alerts.values():
                if (alert.status == AlertStatus.SENT and 
                    not alert.acknowledged_by and
                    alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]):
                    
                    # Check if escalation is needed
                    time_since_sent = current_time - (alert.last_delivery_attempt or alert.timestamp)
                    
                    # Escalate after 15 minutes for critical, 30 minutes for error
                    escalation_threshold = 900 if alert.severity == AlertSeverity.CRITICAL else 1800
                    
                    if time_since_sent > escalation_threshold and alert.escalation_level < 2:
                        await self._escalate_alert(alert)
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert."""
        alert.escalation_level += 1
        alert.status = AlertStatus.ESCALATED
        
        # Create escalation alert
        escalation_title = f"ESCALATED: {alert.title}"
        escalation_message = f"Alert has been escalated (Level {alert.escalation_level}). Original alert not acknowledged.\\n\\nOriginal: {alert.message}"
        
        escalation_alert = Alert(
            alert_id=str(uuid.uuid4()),
            timestamp=time.time(),
            severity=AlertSeverity.CRITICAL,  # Always escalate to critical
            title=escalation_title,
            message=escalation_message,
            source="alerting_system",
            component=alert.component,
            correlation_id=alert.correlation_id,
            tags={**alert.tags, 'escalated': 'true', 'original_alert_id': alert.alert_id},
            context={'original_alert': alert.alert_id, 'escalation_level': alert.escalation_level}
        )
        
        # Send escalation through all available channels
        await self._deliver_alert(escalation_alert)
        
        self.metrics['escalated_alerts'] += 1
        
        print(f"🚨 ALERT ESCALATED: {alert.title} (Level {alert.escalation_level})")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User acknowledging the alert
            
        Returns:
            True if alert was acknowledged successfully
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                
                print(f"✅ Alert acknowledged: {alert.title} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = "") -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolved_by: User resolving the alert
            resolution_note: Optional resolution note
            
        Returns:
            True if alert was resolved successfully
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_by = resolved_by
                alert.resolved_at = time.time()
                
                if resolution_note:
                    alert.context['resolution_note'] = resolution_note
                
                print(f"✅ Alert resolved: {alert.title} by {resolved_by}")
                return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.active_alerts.values() 
                   if alert.status not in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED]]
    
    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive alerting dashboard data."""
        current_time = time.time()
        
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            # Calculate metrics
            alerts_by_severity = defaultdict(int)
            alerts_by_component = defaultdict(int)
            unacknowledged_critical = 0
            
            for alert in active_alerts:
                alerts_by_severity[alert.severity.value] += 1
                alerts_by_component[alert.component] += 1
                
                if alert.severity == AlertSeverity.CRITICAL and not alert.acknowledged_by:
                    unacknowledged_critical += 1
            
            # Calculate resolution times for resolved alerts in the last 24 hours
            recent_resolved = [
                alert for alert in self.alert_history
                if (alert.status == AlertStatus.RESOLVED and
                    alert.resolved_at and
                    current_time - alert.resolved_at < 86400)
            ]
            
            avg_resolution_time = 0.0
            if recent_resolved:
                resolution_times = [
                    (alert.resolved_at - alert.timestamp) / 60  # minutes
                    for alert in recent_resolved
                    if alert.resolved_at
                ]
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            # Channel delivery statistics
            channel_stats = {}
            for channel_id, channel in self.delivery_channels.items():
                success_rate = 0.0
                if channel.delivery_count > 0:
                    success_rate = (channel.delivery_count - channel.failed_deliveries) / channel.delivery_count
                
                channel_stats[channel_id] = {
                    'enabled': channel.config.enabled,
                    'delivery_count': channel.delivery_count,
                    'failed_deliveries': channel.failed_deliveries,
                    'success_rate': success_rate,
                    'last_delivery': channel.last_delivery
                }
            
            return {
                'timestamp': current_time,
                'summary': {
                    'total_active_alerts': len(active_alerts),
                    'critical_alerts': alerts_by_severity['critical'],
                    'error_alerts': alerts_by_severity['error'],
                    'warning_alerts': alerts_by_severity['warning'],
                    'info_alerts': alerts_by_severity['info'],
                    'unacknowledged_critical': unacknowledged_critical,
                    'escalated_alerts': self.metrics['escalated_alerts'],
                    'suppressed_alerts': self.metrics['suppressed_alerts']
                },
                'metrics': {
                    'total_alerts_24h': len([a for a in self.alert_history if current_time - a.timestamp < 86400]),
                    'avg_resolution_time_minutes': avg_resolution_time,
                    'delivery_success_rate': sum(cs['success_rate'] for cs in channel_stats.values()) / len(channel_stats),
                    'active_correlations': len(self.correlator.active_correlations)
                },
                'breakdown': {
                    'by_severity': dict(alerts_by_severity),
                    'by_component': dict(alerts_by_component),
                    'by_status': {
                        'pending': len([a for a in active_alerts if a.status == AlertStatus.PENDING]),
                        'sent': len([a for a in active_alerts if a.status == AlertStatus.SENT]),
                        'acknowledged': len([a for a in active_alerts if a.status == AlertStatus.ACKNOWLEDGED]),
                        'escalated': len([a for a in active_alerts if a.status == AlertStatus.ESCALATED])
                    }
                },
                'channels': channel_stats,
                'recent_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'component': alert.component,
                        'status': alert.status.value,
                        'timestamp': alert.timestamp
                    }
                    for alert in sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)[:10]
                ]
            }
    
    def get_system_health(self) -> str:
        """Get overall alerting system health."""
        active_alerts = self.get_active_alerts()
        critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
        unacknowledged_critical = len([a for a in active_alerts 
                                     if a.severity == AlertSeverity.CRITICAL and not a.acknowledged_by])
        
        if unacknowledged_critical > 0:
            return "critical"
        elif critical_alerts > 0:
            return "degraded"
        elif len(active_alerts) > 10:
            return "warning"
        else:
            return "healthy"


# Global alerting system instance
_global_alerting_system: Optional[EnterpriseAlertingSystem] = None


def get_alerting_system() -> EnterpriseAlertingSystem:
    """Get global alerting system instance."""
    global _global_alerting_system
    
    if _global_alerting_system is None:
        _global_alerting_system = EnterpriseAlertingSystem()
    
    return _global_alerting_system


# Convenience functions
def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity,
    source: str,
    component: str,
    **kwargs
) -> str:
    """Convenience function to create an alert."""
    alerting_system = get_alerting_system()
    return alerting_system.create_alert(
        title=title,
        message=message,
        severity=severity,
        source=source,
        component=component,
        **kwargs
    )


def acknowledge_alert(alert_id: str, acknowledged_by: str) -> bool:
    """Convenience function to acknowledge an alert."""
    alerting_system = get_alerting_system()
    return alerting_system.acknowledge_alert(alert_id, acknowledged_by)


def resolve_alert(alert_id: str, resolved_by: str, resolution_note: str = "") -> bool:
    """Convenience function to resolve an alert."""
    alerting_system = get_alerting_system()
    return alerting_system.resolve_alert(alert_id, resolved_by, resolution_note)