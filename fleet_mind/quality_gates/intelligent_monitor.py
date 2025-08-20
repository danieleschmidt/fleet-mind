"""Intelligent Quality Monitoring System with ML-Powered Prediction.

Advanced quality monitoring that uses machine learning to predict quality issues
before they occur and automatically trigger preventive measures.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from ..utils.performance import performance_monitor
from ..utils.advanced_logging import get_logger

logger = get_logger(__name__)


class QualityMetricType(Enum):
    """Types of quality metrics."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    USER_EXPERIENCE = "user_experience"
    RESOURCE_EFFICIENCY = "resource_efficiency"


@dataclass
class QualityMetric:
    """Quality metric definition with ML prediction capabilities."""
    name: str
    metric_type: QualityMetricType
    current_value: float
    target_value: float
    tolerance: float
    prediction_horizon: int = 300  # seconds
    prediction_accuracy: float = 0.0
    trend: str = "stable"  # stable, improving, degrading
    last_updated: datetime = field(default_factory=datetime.now)
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class QualityThreshold:
    """Quality threshold with adaptive adjustment."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    adaptive: bool = True
    adjustment_factor: float = 0.1
    adjustment_history: List[Tuple[datetime, float, float]] = field(default_factory=list)


class IntelligentQualityMonitor:
    """Advanced quality monitoring system with ML-powered prediction and automated response."""
    
    def __init__(self, 
                 prediction_model: Optional[str] = None,
                 enable_autofix: bool = True,
                 monitoring_interval: int = 30):
        """Initialize intelligent quality monitor.
        
        Args:
            prediction_model: ML model for quality prediction
            enable_autofix: Enable automatic quality issue resolution
            monitoring_interval: Monitoring interval in seconds
        """
        self.prediction_model = prediction_model or "ensemble"
        self.enable_autofix = enable_autofix
        self.monitoring_interval = monitoring_interval
        
        self.metrics: Dict[str, QualityMetric] = {}
        self.thresholds: Dict[str, QualityThreshold] = {}
        self.prediction_cache: Dict[str, Tuple[datetime, float]] = {}
        self.autofix_handlers: Dict[str, Callable] = {}
        
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self._setup_default_metrics()
        self._setup_prediction_models()
        
        logger.info(f"Intelligent Quality Monitor initialized with {self.prediction_model} prediction model")
    
    def _setup_default_metrics(self):
        """Setup default quality metrics for drone swarm systems."""
        default_metrics = [
            QualityMetric("coordination_latency", QualityMetricType.PERFORMANCE, 0.0, 100.0, 10.0),
            QualityMetric("system_reliability", QualityMetricType.RELIABILITY, 0.0, 99.9, 0.1),
            QualityMetric("security_score", QualityMetricType.SECURITY, 0.0, 100.0, 5.0),
            QualityMetric("compliance_adherence", QualityMetricType.COMPLIANCE, 0.0, 100.0, 2.0),
            QualityMetric("user_satisfaction", QualityMetricType.USER_EXPERIENCE, 0.0, 95.0, 5.0),
            QualityMetric("resource_efficiency", QualityMetricType.RESOURCE_EFFICIENCY, 0.0, 90.0, 10.0),
        ]
        
        for metric in default_metrics:
            self.metrics[metric.name] = metric
            self.thresholds[metric.name] = QualityThreshold(
                metric.name,
                metric.target_value * 0.8,  # 80% as warning
                metric.target_value * 0.6   # 60% as critical
            )
    
    def _setup_prediction_models(self):
        """Setup ML models for quality prediction."""
        self.prediction_models = {
            "linear": self._linear_prediction,
            "polynomial": self._polynomial_prediction,
            "exponential": self._exponential_prediction,
            "ensemble": self._ensemble_prediction
        }
        
        # Initialize prediction parameters
        self.model_weights = {
            "linear": 0.3,
            "polynomial": 0.4,
            "exponential": 0.3
        }
    
    async def start_monitoring(self):
        """Start continuous quality monitoring."""
        if self.monitoring_active:
            logger.warning("Quality monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Intelligent quality monitoring started")
    
    async def stop_monitoring(self):
        """Stop quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Quality monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop with intelligent analysis."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                await self._collect_metrics()
                
                # Generate predictions
                await self._generate_predictions()
                
                # Analyze quality trends
                await self._analyze_trends()
                
                # Check thresholds and trigger actions
                await self._check_thresholds()
                
                # Update adaptive thresholds
                await self._update_adaptive_thresholds()
                
                # Log quality report
                await self._log_quality_report()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _collect_metrics(self):
        """Collect current quality metrics from system."""
        current_time = datetime.now()
        
        # Performance metrics
        performance_data = performance_monitor.get_summary()
        if performance_data:
            latency_metric = self.metrics.get("coordination_latency")
            if latency_metric:
                current_latency = performance_data.get("avg_latency", 0.0)
                latency_metric.current_value = current_latency
                latency_metric.last_updated = current_time
                latency_metric.historical_values.append((current_time, current_latency))
                
                # Keep only last 100 values for trend analysis
                if len(latency_metric.historical_values) > 100:
                    latency_metric.historical_values = latency_metric.historical_values[-100:]
        
        # Simulate other metric collection (in real implementation, would integrate with actual systems)
        for metric_name, metric in self.metrics.items():
            if metric_name != "coordination_latency":  # Already handled above
                # Simulate metric collection with some variance
                base_value = metric.target_value * (0.8 + 0.4 * np.random.random())
                noise = np.random.normal(0, metric.tolerance * 0.1)
                metric.current_value = max(0, base_value + noise)
                metric.last_updated = current_time
                metric.historical_values.append((current_time, metric.current_value))
                
                if len(metric.historical_values) > 100:
                    metric.historical_values = metric.historical_values[-100:]
    
    async def _generate_predictions(self):
        """Generate ML-powered quality predictions."""
        for metric_name, metric in self.metrics.items():
            if len(metric.historical_values) < 5:
                continue  # Need minimum data for prediction
            
            try:
                # Use selected prediction model
                prediction_func = self.prediction_models.get(self.prediction_model)
                if prediction_func:
                    predicted_value = prediction_func(metric)
                    prediction_time = datetime.now() + timedelta(seconds=metric.prediction_horizon)
                    
                    # Cache prediction
                    self.prediction_cache[metric_name] = (prediction_time, predicted_value)
                    
                    # Calculate prediction accuracy (simplified)
                    if len(metric.historical_values) >= 2:
                        recent_values = [v[1] for v in metric.historical_values[-5:]]
                        variance = np.var(recent_values)
                        metric.prediction_accuracy = max(0, 1.0 - (variance / metric.target_value))
                
            except Exception as e:
                logger.error(f"Error generating prediction for {metric_name}: {e}")
    
    def _linear_prediction(self, metric: QualityMetric) -> float:
        """Linear trend prediction."""
        if len(metric.historical_values) < 2:
            return metric.current_value
        
        # Calculate linear trend
        values = [v[1] for v in metric.historical_values[-10:]]  # Last 10 values
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        # Predict future value
        future_x = len(values) + (metric.prediction_horizon / self.monitoring_interval)
        return coeffs[0] * future_x + coeffs[1]
    
    def _polynomial_prediction(self, metric: QualityMetric) -> float:
        """Polynomial trend prediction."""
        if len(metric.historical_values) < 3:
            return metric.current_value
        
        values = [v[1] for v in metric.historical_values[-15:]]
        x = np.arange(len(values))
        
        # Use degree 2 polynomial
        try:
            coeffs = np.polyfit(x, values, min(2, len(values) - 1))
            future_x = len(values) + (metric.prediction_horizon / self.monitoring_interval)
            return np.polyval(coeffs, future_x)
        except np.RankWarning:
            return self._linear_prediction(metric)
    
    def _exponential_prediction(self, metric: QualityMetric) -> float:
        """Exponential smoothing prediction."""
        if len(metric.historical_values) < 2:
            return metric.current_value
        
        alpha = 0.3  # Smoothing factor
        values = [v[1] for v in metric.historical_values[-10:]]
        
        # Apply exponential smoothing
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Simple extrapolation based on recent trend
        recent_change = values[-1] - values[-2] if len(values) >= 2 else 0
        return smoothed + recent_change * (metric.prediction_horizon / self.monitoring_interval)
    
    def _ensemble_prediction(self, metric: QualityMetric) -> float:
        """Ensemble prediction combining multiple models."""
        try:
            linear_pred = self._linear_prediction(metric)
            poly_pred = self._polynomial_prediction(metric)
            exp_pred = self._exponential_prediction(metric)
            
            # Weighted ensemble
            ensemble_pred = (
                self.model_weights["linear"] * linear_pred +
                self.model_weights["polynomial"] * poly_pred +
                self.model_weights["exponential"] * exp_pred
            )
            
            return ensemble_pred
            
        except Exception:
            return metric.current_value
    
    async def _analyze_trends(self):
        """Analyze quality trends and patterns."""
        for metric_name, metric in self.metrics.items():
            if len(metric.historical_values) < 5:
                continue
            
            # Calculate trend direction
            recent_values = [v[1] for v in metric.historical_values[-5:]]
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            if trend_slope > metric.tolerance * 0.1:
                metric.trend = "improving"
            elif trend_slope < -metric.tolerance * 0.1:
                metric.trend = "degrading"
            else:
                metric.trend = "stable"
    
    async def _check_thresholds(self):
        """Check quality thresholds and trigger automated responses."""
        for metric_name, metric in self.metrics.items():
            threshold = self.thresholds.get(metric_name)
            if not threshold:
                continue
            
            # Check current value against thresholds
            if metric.current_value < threshold.critical_threshold:
                await self._handle_critical_quality_issue(metric, threshold)
            elif metric.current_value < threshold.warning_threshold:
                await self._handle_warning_quality_issue(metric, threshold)
            
            # Check predictions against thresholds
            prediction = self.prediction_cache.get(metric_name)
            if prediction and prediction[1] < threshold.warning_threshold:
                await self._handle_predicted_quality_issue(metric, threshold, prediction[1])
    
    async def _handle_critical_quality_issue(self, metric: QualityMetric, threshold: QualityThreshold):
        """Handle critical quality issues with immediate response."""
        logger.critical(f"CRITICAL: {metric.name} at {metric.current_value:.2f} (threshold: {threshold.critical_threshold:.2f})")
        
        if self.enable_autofix:
            autofix_handler = self.autofix_handlers.get(metric.name)
            if autofix_handler:
                try:
                    await autofix_handler(metric, "critical")
                    logger.info(f"Autofix applied for critical issue: {metric.name}")
                except Exception as e:
                    logger.error(f"Autofix failed for {metric.name}: {e}")
    
    async def _handle_warning_quality_issue(self, metric: QualityMetric, threshold: QualityThreshold):
        """Handle warning-level quality issues."""
        logger.warning(f"WARNING: {metric.name} at {metric.current_value:.2f} (threshold: {threshold.warning_threshold:.2f})")
        
        if self.enable_autofix:
            autofix_handler = self.autofix_handlers.get(metric.name)
            if autofix_handler:
                try:
                    await autofix_handler(metric, "warning")
                    logger.info(f"Preventive action applied for: {metric.name}")
                except Exception as e:
                    logger.error(f"Preventive action failed for {metric.name}: {e}")
    
    async def _handle_predicted_quality_issue(self, metric: QualityMetric, threshold: QualityThreshold, predicted_value: float):
        """Handle predicted quality issues before they occur."""
        logger.info(f"PREDICTION: {metric.name} predicted to reach {predicted_value:.2f} in {metric.prediction_horizon}s")
        
        if self.enable_autofix and predicted_value < threshold.critical_threshold:
            autofix_handler = self.autofix_handlers.get(metric.name)
            if autofix_handler:
                try:
                    await autofix_handler(metric, "predicted")
                    logger.info(f"Proactive fix applied for predicted issue: {metric.name}")
                except Exception as e:
                    logger.error(f"Proactive fix failed for {metric.name}: {e}")
    
    async def _update_adaptive_thresholds(self):
        """Update thresholds based on system behavior and performance."""
        for metric_name, threshold in self.thresholds.items():
            if not threshold.adaptive:
                continue
            
            metric = self.metrics.get(metric_name)
            if not metric or len(metric.historical_values) < 10:
                continue
            
            # Calculate statistical properties
            recent_values = [v[1] for v in metric.historical_values[-20:]]
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Adjust thresholds based on system performance
            if metric.trend == "improving":
                # Tighten thresholds for better quality
                new_warning = threshold.warning_threshold * (1 + threshold.adjustment_factor)
                new_critical = threshold.critical_threshold * (1 + threshold.adjustment_factor)
            elif metric.trend == "degrading" and std_value > metric.tolerance:
                # Relax thresholds temporarily to avoid false alarms
                new_warning = threshold.warning_threshold * (1 - threshold.adjustment_factor)
                new_critical = threshold.critical_threshold * (1 - threshold.adjustment_factor)
            else:
                continue  # No adjustment needed
            
            # Apply threshold bounds
            new_warning = min(new_warning, metric.target_value * 0.95)
            new_critical = min(new_critical, metric.target_value * 0.8)
            
            # Update thresholds
            threshold.adjustment_history.append((datetime.now(), threshold.warning_threshold, threshold.critical_threshold))
            threshold.warning_threshold = new_warning
            threshold.critical_threshold = new_critical
            
            logger.debug(f"Adaptive threshold update for {metric_name}: warning={new_warning:.2f}, critical={new_critical:.2f}")
    
    async def _log_quality_report(self):
        """Generate and log comprehensive quality report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "predictions": {},
            "overall_health": "healthy"
        }
        
        critical_issues = 0
        warning_issues = 0
        
        for metric_name, metric in self.metrics.items():
            threshold = self.thresholds.get(metric_name)
            
            status = "healthy"
            if threshold:
                if metric.current_value < threshold.critical_threshold:
                    status = "critical"
                    critical_issues += 1
                elif metric.current_value < threshold.warning_threshold:
                    status = "warning"
                    warning_issues += 1
            
            report["metrics"][metric_name] = {
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "trend": metric.trend,
                "prediction_accuracy": metric.prediction_accuracy,
                "status": status
            }
            
            # Add prediction if available
            prediction = self.prediction_cache.get(metric_name)
            if prediction:
                report["predictions"][metric_name] = {
                    "predicted_value": prediction[1],
                    "prediction_time": prediction[0].isoformat(),
                    "confidence": metric.prediction_accuracy
                }
        
        # Determine overall health
        if critical_issues > 0:
            report["overall_health"] = "critical"
        elif warning_issues > 0:
            report["overall_health"] = "warning"
        
        logger.info(f"Quality Report: {json.dumps(report, indent=2)}")
    
    def register_autofix_handler(self, metric_name: str, handler: Callable):
        """Register an automatic fix handler for a quality metric."""
        self.autofix_handlers[metric_name] = handler
        logger.info(f"Autofix handler registered for {metric_name}")
    
    def add_custom_metric(self, metric: QualityMetric, threshold: Optional[QualityThreshold] = None):
        """Add a custom quality metric to monitor."""
        self.metrics[metric.name] = metric
        
        if threshold:
            self.thresholds[metric.name] = threshold
        else:
            # Create default threshold
            self.thresholds[metric.name] = QualityThreshold(
                metric.name,
                metric.target_value * 0.8,
                metric.target_value * 0.6
            )
        
        logger.info(f"Custom metric added: {metric.name}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current quality summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "total_metrics": len(self.metrics),
            "metrics": {}
        }
        
        for metric_name, metric in self.metrics.items():
            threshold = self.thresholds.get(metric_name)
            prediction = self.prediction_cache.get(metric_name)
            
            summary["metrics"][metric_name] = {
                "current_value": metric.current_value,
                "target_value": metric.target_value,
                "trend": metric.trend,
                "prediction_accuracy": metric.prediction_accuracy,
                "warning_threshold": threshold.warning_threshold if threshold else None,
                "critical_threshold": threshold.critical_threshold if threshold else None,
                "predicted_value": prediction[1] if prediction else None,
                "last_updated": metric.last_updated.isoformat()
            }
        
        return summary