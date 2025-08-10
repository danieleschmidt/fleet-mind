"""Advanced alerting and notification system for Fleet-Mind."""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("Warning: aiohttp not available - webhook alerts disabled")

from .logging import get_logger


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email" 
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    CONSOLE = "console"
    DATABASE = "database"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    condition: str  # Condition expression
    priority: AlertPriority
    channels: List[AlertChannel]
    cooldown_seconds: float = 300.0  # 5 minutes
    enabled: bool = True
    tags: Set[str] = field(default_factory=set)
    
    # Channel-specific configurations
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    custom_message_template: Optional[str] = None


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    timestamp: float
    priority: AlertPriority
    title: str
    message: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    channels_sent: Set[AlertChannel] = field(default_factory=set)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            **asdict(self),
            'priority': self.priority.value,
            'channels_sent': [c.value for c in self.channels_sent],
            'tags': list(self.tags)
        }


@dataclass
class AlertingConfig:
    """Alerting system configuration."""
    # Email configuration
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    default_sender_email: str = "fleet-mind@example.com"
    
    # Webhook configuration
    default_webhook_timeout: float = 10.0
    webhook_retry_attempts: int = 3
    
    # Rate limiting
    max_alerts_per_hour: int = 100
    max_alerts_per_minute: int = 10
    
    # Storage
    max_alert_history: int = 10000
    alert_retention_days: int = 30


class AlertManager:
    """Comprehensive alerting and notification system."""
    
    def __init__(self, config: Optional[AlertingConfig] = None):
        """Initialize alert manager.
        
        Args:
            config: Alerting configuration
        """
        self.config = config or AlertingConfig()
        self.logger = get_logger("alert_manager")
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Rate limiting
        self.alert_timestamps: List[float] = []
        self.rule_last_triggered: Dict[str, float] = {}
        
        # Delivery tracking
        self.delivery_stats: Dict[AlertChannel, Dict[str, int]] = {
            channel: {"sent": 0, "failed": 0} for channel in AlertChannel
        }
        
        # Custom handlers
        self.custom_handlers: Dict[AlertChannel, Callable] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        print("Alert manager initialized")

    async def start(self) -> None:
        """Start alert manager background tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop alert manager background tasks."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Alert manager stopped")

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule.
        
        Args:
            rule_id: Rule identifier to remove
            
        Returns:
            True if rule was removed
        """
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable alert rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            True if rule was enabled
        """
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable alert rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            True if rule was disabled
        """
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            return True
        return False

    async def trigger_alert(
        self,
        rule_id: str,
        title: str,
        message: str,
        source: str,
        data: Optional[Dict[str, Any]] = None,
        override_channels: Optional[List[AlertChannel]] = None
    ) -> Optional[Alert]:
        """Trigger an alert.
        
        Args:
            rule_id: Alert rule identifier
            title: Alert title
            message: Alert message
            source: Alert source
            data: Additional alert data
            override_channels: Override delivery channels
            
        Returns:
            Created alert or None if rate limited/disabled
        """
        if rule_id not in self.alert_rules:
            self.logger.warning(f"Unknown alert rule: {rule_id}")
            return None
        
        rule = self.alert_rules[rule_id]
        
        if not rule.enabled:
            return None
        
        # Check rate limiting
        if not self._check_rate_limits(rule_id):
            self.logger.warning(f"Rate limit exceeded for rule: {rule_id}")
            return None
        
        # Check cooldown
        if rule_id in self.rule_last_triggered:
            time_since_last = time.time() - self.rule_last_triggered[rule_id]
            if time_since_last < rule.cooldown_seconds:
                return None
        
        # Create alert
        alert_id = f"{rule_id}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            timestamp=time.time(),
            priority=rule.priority,
            title=title,
            message=message,
            source=source,
            data=data or {},
            tags=rule.tags.copy()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.rule_last_triggered[rule_id] = time.time()
        
        # Deliver alert
        delivery_channels = override_channels or rule.channels
        await self._deliver_alert(alert, delivery_channels, rule)
        
        self.logger.info(f"Triggered alert: {title} (priority: {rule.priority.value})")
        return alert

    async def _deliver_alert(
        self, 
        alert: Alert, 
        channels: List[AlertChannel],
        rule: AlertRule
    ) -> None:
        """Deliver alert through specified channels.
        
        Args:
            alert: Alert to deliver
            channels: Delivery channels
            rule: Alert rule configuration
        """
        delivery_tasks = []
        
        for channel in channels:
            if channel == AlertChannel.CONSOLE:
                task = self._deliver_console(alert)
            elif channel == AlertChannel.LOG:
                task = self._deliver_log(alert)
            elif channel == AlertChannel.EMAIL:
                task = self._deliver_email(alert, rule)
            elif channel == AlertChannel.WEBHOOK:
                task = self._deliver_webhook(alert, rule)
            elif channel == AlertChannel.SLACK:
                task = self._deliver_slack(alert, rule)
            elif channel in self.custom_handlers:
                task = self.custom_handlers[channel](alert, rule)
            else:
                self.logger.warning(f"Unsupported alert channel: {channel}")
                continue
                
            delivery_tasks.append(self._safe_delivery(task, channel, alert))
        
        # Execute deliveries concurrently
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

    async def _safe_delivery(
        self, 
        task: asyncio.Task, 
        channel: AlertChannel, 
        alert: Alert
    ) -> None:
        """Safely execute delivery task with error handling.
        
        Args:
            task: Delivery task
            channel: Delivery channel
            alert: Alert being delivered
        """
        try:
            await task
            self.delivery_stats[channel]["sent"] += 1
            alert.channels_sent.add(channel)
            
        except Exception as e:
            self.logger.error(f"Alert delivery failed via {channel.value}: {e}")
            self.delivery_stats[channel]["failed"] += 1

    async def _deliver_console(self, alert: Alert) -> None:
        """Deliver alert to console."""
        priority_prefix = {
            AlertPriority.LOW: "INFO",
            AlertPriority.MEDIUM: "NOTICE", 
            AlertPriority.HIGH: "WARNING",
            AlertPriority.CRITICAL: "ERROR",
            AlertPriority.EMERGENCY: "EMERGENCY"
        }
        
        prefix = priority_prefix.get(alert.priority, "ALERT")
        print(f"[{prefix}] {alert.title}: {alert.message}")

    async def _deliver_log(self, alert: Alert) -> None:
        """Deliver alert to log system."""
        log_data = {
            'alert_id': alert.alert_id,
            'priority': alert.priority.value,
            'source': alert.source,
            'data': alert.data
        }
        
        if alert.priority == AlertPriority.EMERGENCY:
            self.logger.critical(f"{alert.title}: {alert.message}", **log_data)
        elif alert.priority == AlertPriority.CRITICAL:
            self.logger.critical(f"{alert.title}: {alert.message}", **log_data)
        elif alert.priority == AlertPriority.HIGH:
            self.logger.error(f"{alert.title}: {alert.message}", **log_data)
        elif alert.priority == AlertPriority.MEDIUM:
            self.logger.warning(f"{alert.title}: {alert.message}", **log_data)
        else:
            self.logger.info(f"{alert.title}: {alert.message}", **log_data)

    async def _deliver_email(self, alert: Alert, rule: AlertRule) -> None:
        """Deliver alert via email."""
        if not rule.email_recipients:
            raise ValueError("No email recipients configured")
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = self.config.default_sender_email
        msg['To'] = ', '.join(rule.email_recipients)
        msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"
        
        # Email body
        if rule.custom_message_template:
            body = rule.custom_message_template.format(
                title=alert.title,
                message=alert.message,
                source=alert.source,
                priority=alert.priority.value,
                timestamp=time.ctime(alert.timestamp),
                **alert.data
            )
        else:
            body = f"""
Alert: {alert.title}
Priority: {alert.priority.value.upper()}
Source: {alert.source}
Time: {time.ctime(alert.timestamp)}

Message: {alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}
"""
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                
                server.send_message(msg)
                
        except Exception as e:
            raise Exception(f"Email delivery failed: {e}")

    async def _deliver_webhook(self, alert: Alert, rule: AlertRule) -> None:
        """Deliver alert via webhook."""
        if not rule.webhook_url or not AIOHTTP_AVAILABLE:
            raise ValueError("Webhook URL not configured or aiohttp not available")
        
        payload = {
            'alert_id': alert.alert_id,
            'rule_id': alert.rule_id,
            'timestamp': alert.timestamp,
            'priority': alert.priority.value,
            'title': alert.title,
            'message': alert.message,
            'source': alert.source,
            'data': alert.data,
            'tags': list(alert.tags)
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.default_webhook_timeout)
        
        for attempt in range(self.config.webhook_retry_attempts):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        rule.webhook_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status < 400:
                            return
                        else:
                            raise Exception(f"Webhook returned status {response.status}")
                            
            except Exception as e:
                if attempt == self.config.webhook_retry_attempts - 1:
                    raise Exception(f"Webhook delivery failed after {self.config.webhook_retry_attempts} attempts: {e}")
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _deliver_slack(self, alert: Alert, rule: AlertRule) -> None:
        """Deliver alert via Slack webhook."""
        if not rule.webhook_url or not AIOHTTP_AVAILABLE:
            raise ValueError("Slack webhook URL not configured")
        
        # Slack color coding based on priority
        color_map = {
            AlertPriority.LOW: "#36a64f",      # Green
            AlertPriority.MEDIUM: "#ffeb3b",   # Yellow
            AlertPriority.HIGH: "#ff9800",     # Orange
            AlertPriority.CRITICAL: "#f44336", # Red
            AlertPriority.EMERGENCY: "#9c27b0" # Purple
        }
        
        payload = {
            "channel": rule.slack_channel or "#alerts",
            "username": "Fleet-Mind",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color_map.get(alert.priority, "#36a64f"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Priority",
                            "value": alert.priority.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Source",
                            "value": alert.source,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": time.ctime(alert.timestamp),
                            "short": False
                        }
                    ],
                    "footer": f"Fleet-Mind Alert System",
                    "ts": int(alert.timestamp)
                }
            ]
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.default_webhook_timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(rule.webhook_url, json=payload) as response:
                if response.status >= 400:
                    raise Exception(f"Slack delivery failed with status {response.status}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = time.time()
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was resolved
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            
            # Move to history and remove from active
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False

    def get_active_alerts(self, priority_filter: Optional[AlertPriority] = None) -> List[Alert]:
        """Get active alerts.
        
        Args:
            priority_filter: Filter by priority level
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if priority_filter:
            alerts = [a for a in alerts if a.priority == priority_filter]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics.
        
        Returns:
            Comprehensive alerting statistics
        """
        current_time = time.time()
        
        # Recent alerts (last 24 hours)
        recent_alerts = [a for a in self.alert_history if current_time - a.timestamp < 86400]
        
        # Priority breakdown
        priority_counts = {p.value: 0 for p in AlertPriority}
        for alert in recent_alerts:
            priority_counts[alert.priority.value] += 1
        
        # Channel statistics
        channel_stats = {}
        for channel, stats in self.delivery_stats.items():
            total_attempts = stats["sent"] + stats["failed"]
            success_rate = (stats["sent"] / total_attempts * 100) if total_attempts > 0 else 0
            
            channel_stats[channel.value] = {
                "sent": stats["sent"],
                "failed": stats["failed"],
                "success_rate": success_rate
            }
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "recent_alerts_24h": len(recent_alerts),
            "priority_breakdown": priority_counts,
            "channel_statistics": channel_stats,
            "rate_limiting": {
                "alerts_last_hour": len([t for t in self.alert_timestamps if current_time - t < 3600]),
                "max_per_hour": self.config.max_alerts_per_hour
            }
        }

    def add_custom_handler(self, channel: AlertChannel, handler: Callable) -> None:
        """Add custom alert delivery handler.
        
        Args:
            channel: Alert channel
            handler: Async handler function
        """
        self.custom_handlers[channel] = handler
        self.logger.info(f"Added custom handler for channel: {channel.value}")

    def _check_rate_limits(self, rule_id: str) -> bool:
        """Check if alert is within rate limits.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            True if within limits
        """
        current_time = time.time()
        
        # Clean old timestamps
        self.alert_timestamps = [t for t in self.alert_timestamps if current_time - t < 3600]
        
        # Check hourly limit
        if len(self.alert_timestamps) >= self.config.max_alerts_per_hour:
            return False
        
        # Check per-minute limit
        recent_minute = [t for t in self.alert_timestamps if current_time - t < 60]
        if len(recent_minute) >= self.config.max_alerts_per_minute:
            return False
        
        # Record timestamp
        self.alert_timestamps.append(current_time)
        return True

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        try:
            while self._running:
                # Clean up old resolved alerts
                self._cleanup_old_alerts()
                
                # Clean up rate limiting data
                current_time = time.time()
                self.alert_timestamps = [t for t in self.alert_timestamps if current_time - t < 3600]
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
        except asyncio.CancelledError:
            self.logger.info("Alert cleanup loop cancelled")

    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts from history."""
        if len(self.alert_history) <= self.config.max_alert_history:
            return
        
        # Keep most recent alerts
        self.alert_history = self.alert_history[-self.config.max_alert_history:]
        
        # Also clean by age
        cutoff_time = time.time() - (self.config.alert_retention_days * 86400)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]


# Convenience functions for common alert scenarios
def create_system_alert_rule() -> AlertRule:
    """Create standard system health alert rule."""
    return AlertRule(
        rule_id="system_health",
        name="System Health Alert",
        condition="system.cpu_usage > 90 OR system.memory_usage > 95",
        priority=AlertPriority.HIGH,
        channels=[AlertChannel.LOG, AlertChannel.CONSOLE],
        cooldown_seconds=300.0
    )


def create_security_alert_rule(webhook_url: str = None) -> AlertRule:
    """Create security incident alert rule."""
    channels = [AlertChannel.LOG, AlertChannel.CONSOLE]
    if webhook_url:
        channels.append(AlertChannel.WEBHOOK)
    
    return AlertRule(
        rule_id="security_incident",
        name="Security Incident Alert",
        condition="security.threat_detected",
        priority=AlertPriority.CRITICAL,
        channels=channels,
        webhook_url=webhook_url,
        cooldown_seconds=60.0,  # Shorter cooldown for security
        tags={"security", "critical"}
    )


def create_drone_failure_alert_rule(email_recipients: List[str] = None) -> AlertRule:
    """Create drone failure alert rule."""
    channels = [AlertChannel.LOG, AlertChannel.CONSOLE]
    if email_recipients:
        channels.append(AlertChannel.EMAIL)
    
    return AlertRule(
        rule_id="drone_failure",
        name="Drone Failure Alert", 
        condition="drone.status == 'failed' OR drone.battery_percent < 5",
        priority=AlertPriority.HIGH,
        channels=channels,
        email_recipients=email_recipients or [],
        cooldown_seconds=180.0,
        tags={"drone", "operational"}
    )