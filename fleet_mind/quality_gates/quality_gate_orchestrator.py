"""Quality Gate Orchestrator - Comprehensive Quality Assurance System.

Master orchestrator that coordinates all quality gates components to provide
comprehensive quality assurance for the Fleet-Mind system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .intelligent_monitor import IntelligentQualityMonitor
from .progressive_testing import ProgressiveTestingFramework
from .performance_optimizer import ContinuousPerformanceOptimizer
from .compliance_automation import ComplianceAutomation, ComplianceFramework
from .reliability_engineering import ProactiveReliabilityEngine, ReliabilityLevel

from ..utils.advanced_logging import get_logger

logger = get_logger(__name__)


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime


class QualityGateOrchestrator:
    """Master orchestrator for all quality gates and quality assurance systems."""
    
    def __init__(self, 
                 enable_all_gates: bool = True,
                 strict_mode: bool = False,
                 target_reliability: ReliabilityLevel = ReliabilityLevel.HIGH):
        """Initialize quality gate orchestrator.
        
        Args:
            enable_all_gates: Enable all quality gates by default
            strict_mode: Enable strict quality enforcement
            target_reliability: Target reliability level
        """
        self.enable_all_gates = enable_all_gates
        self.strict_mode = strict_mode
        self.target_reliability = target_reliability
        
        # Initialize all quality systems
        self.quality_monitor = IntelligentQualityMonitor(
            enable_autofix=True,
            monitoring_interval=30
        )
        
        self.testing_framework = ProgressiveTestingFramework(
            enable_adaptive_generation=True,
            max_parallel_tests=10
        )
        
        self.performance_optimizer = ContinuousPerformanceOptimizer(
            optimization_interval=300,
            enable_auto_optimization=True,
            safety_threshold=0.9
        )
        
        self.compliance_automation = ComplianceAutomation(
            enabled_frameworks=[
                ComplianceFramework.GDPR,
                ComplianceFramework.CCPA,
                ComplianceFramework.ISO_27001,
                ComplianceFramework.SOC2
            ],
            auto_remediation=True
        )
        
        self.reliability_engine = ProactiveReliabilityEngine(
            target_reliability=target_reliability,
            prediction_horizon=3600,
            enable_auto_healing=True
        )
        
        self.quality_gates_active = False
        self.orchestration_task: Optional[asyncio.Task] = None
        self.gate_results_history: List[QualityGateResult] = []
        
        logger.info("Quality Gate Orchestrator initialized")
    
    async def start_quality_gates(self):
        """Start all quality gates and monitoring systems."""
        if self.quality_gates_active:
            logger.warning("Quality gates already active")
            return
        
        logger.info("Starting comprehensive quality gates system...")
        
        try:
            # Start all quality systems
            await self.quality_monitor.start_monitoring()
            await self.performance_optimizer.start_optimization()
            await self.compliance_automation.start_monitoring()
            await self.reliability_engine.start_monitoring()
            
            # Note: Testing framework is started on-demand
            
            self.quality_gates_active = True
            self.orchestration_task = asyncio.create_task(self._orchestration_loop())
            
            logger.info("✅ All quality gates started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start quality gates: {e}")
            await self.stop_quality_gates()
            raise
    
    async def stop_quality_gates(self):
        """Stop all quality gates and monitoring systems."""
        logger.info("Stopping quality gates system...")
        
        self.quality_gates_active = False
        
        if self.orchestration_task:
            self.orchestration_task.cancel()
            try:
                await self.orchestration_task
            except asyncio.CancelledError:
                pass
        
        # Stop all quality systems
        try:
            await self.quality_monitor.stop_monitoring()
            await self.performance_optimizer.stop_optimization()
            await self.compliance_automation.stop_monitoring()
            await self.reliability_engine.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping quality systems: {e}")
        
        logger.info("Quality gates system stopped")
    
    async def _orchestration_loop(self):
        """Main orchestration loop for quality gates."""
        while self.quality_gates_active:
            try:
                # Run comprehensive quality evaluation
                await self._run_quality_evaluation()
                
                # Generate consolidated quality report
                await self._generate_consolidated_report()
                
                # Check for critical quality issues
                await self._check_critical_issues()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in quality gates orchestration: {e}")
                await asyncio.sleep(60)
    
    async def _run_quality_evaluation(self):
        """Run comprehensive quality evaluation across all gates."""
        evaluation_start = datetime.now()
        logger.info("Running comprehensive quality evaluation...")
        
        # Run all quality gates in parallel
        gate_tasks = [
            self._evaluate_quality_monitoring_gate(),
            self._evaluate_performance_gate(),
            self._evaluate_compliance_gate(),
            self._evaluate_reliability_gate()
        ]
        
        gate_results = await asyncio.gather(*gate_tasks, return_exceptions=True)
        
        # Process results
        for result in gate_results:
            if isinstance(result, Exception):
                logger.error(f"Quality gate evaluation failed: {result}")
            elif isinstance(result, QualityGateResult):
                self.gate_results_history.append(result)
                
                if not result.passed:
                    logger.warning(f"Quality gate failed: {result.gate_name} (score: {result.score:.2f})")
                else:
                    logger.info(f"Quality gate passed: {result.gate_name} (score: {result.score:.2f})")
        
        # Limit history size
        if len(self.gate_results_history) > 1000:
            self.gate_results_history = self.gate_results_history[-1000:]
        
        evaluation_time = (datetime.now() - evaluation_start).total_seconds()
        logger.info(f"Quality evaluation completed in {evaluation_time:.2f}s")
    
    async def _evaluate_quality_monitoring_gate(self) -> QualityGateResult:
        """Evaluate quality monitoring gate."""
        start_time = datetime.now()
        
        try:
            # Get quality summary
            quality_summary = self.quality_monitor.get_quality_summary()
            
            # Calculate overall quality score
            total_metrics = quality_summary.get("total_metrics", 1)
            compliant_metrics = 0
            
            for metric_name, metric_data in quality_summary.get("metrics", {}).items():
                current_value = metric_data.get("current_value", 0)
                target_value = metric_data.get("target_value", 100)
                warning_threshold = metric_data.get("warning_threshold", target_value * 0.8)
                
                if current_value >= warning_threshold:
                    compliant_metrics += 1
            
            score = (compliant_metrics / total_metrics * 100) if total_metrics > 0 else 0
            passed = score >= 80.0  # 80% threshold
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="quality_monitoring",
                passed=passed,
                score=score,
                details=quality_summary,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating quality monitoring gate: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="quality_monitoring",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _evaluate_performance_gate(self) -> QualityGateResult:
        """Evaluate performance optimization gate."""
        start_time = datetime.now()
        
        try:
            # Get optimization summary
            optimization_summary = self.performance_optimizer.get_optimization_summary()
            
            # Calculate performance score based on metrics
            current_metrics = optimization_summary.get("current_performance_metrics", {})
            score_components = []
            
            for metric_name, metric_data in current_metrics.items():
                current_value = metric_data.get("current_value", 0)
                target_value = metric_data.get("target_value", 100)
                
                # Calculate metric score based on type
                if metric_name in ["response_time", "error_rate", "cpu_utilization", "memory_utilization"]:
                    # Lower is better
                    metric_score = max(0, min(100, (1 - current_value / target_value) * 100))
                else:
                    # Higher is better (throughput, cache_hit_rate)
                    metric_score = max(0, min(100, (current_value / target_value) * 100))
                
                score_components.append(metric_score)
            
            score = sum(score_components) / len(score_components) if score_components else 0
            passed = score >= 75.0  # 75% threshold for performance
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="performance_optimization",
                passed=passed,
                score=score,
                details=optimization_summary,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating performance gate: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="performance_optimization",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _evaluate_compliance_gate(self) -> QualityGateResult:
        """Evaluate compliance automation gate."""
        start_time = datetime.now()
        
        try:
            # Get compliance summary
            compliance_summary = self.compliance_automation.get_compliance_summary()
            
            # Calculate compliance score
            total_rules = compliance_summary.get("total_rules", 1)
            compliant_rules = compliance_summary.get("compliant_rules", 0)
            
            score = (compliant_rules / total_rules * 100) if total_rules > 0 else 0
            passed = score >= 95.0  # 95% compliance threshold
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="compliance_automation",
                passed=passed,
                score=score,
                details=compliance_summary,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating compliance gate: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="compliance_automation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _evaluate_reliability_gate(self) -> QualityGateResult:
        """Evaluate reliability engineering gate."""
        start_time = datetime.now()
        
        try:
            # Get reliability summary
            reliability_summary = self.reliability_engine.get_reliability_summary()
            
            # Use overall reliability score
            score = reliability_summary.get("overall_reliability_score", 0)
            
            # Determine pass/fail based on target reliability
            if self.target_reliability == ReliabilityLevel.CRITICAL:
                threshold = 99.99
            elif self.target_reliability == ReliabilityLevel.HIGH:
                threshold = 99.9
            elif self.target_reliability == ReliabilityLevel.MEDIUM:
                threshold = 99.5
            else:
                threshold = 99.0
            
            passed = score >= threshold
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="reliability_engineering",
                passed=passed,
                score=score,
                details=reliability_summary,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating reliability gate: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QualityGateResult(
                gate_name="reliability_engineering",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    async def _generate_consolidated_report(self):
        """Generate consolidated quality report across all systems."""
        if not self.gate_results_history:
            return
        
        # Get latest results for each gate
        latest_results = {}
        for result in reversed(self.gate_results_history):
            if result.gate_name not in latest_results:
                latest_results[result.gate_name] = result
        
        # Calculate overall quality score
        if latest_results:
            total_score = sum(result.score for result in latest_results.values())
            overall_score = total_score / len(latest_results)
            
            passed_gates = sum(1 for result in latest_results.values() if result.passed)
            total_gates = len(latest_results)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_quality_score": overall_score,
                "gates_passed": passed_gates,
                "total_gates": total_gates,
                "pass_rate": (passed_gates / total_gates * 100) if total_gates > 0 else 0,
                "strict_mode": self.strict_mode,
                "gate_results": {
                    result.gate_name: {
                        "passed": result.passed,
                        "score": result.score,
                        "execution_time": result.execution_time,
                        "timestamp": result.timestamp.isoformat()
                    } for result in latest_results.values()
                },
                "quality_status": self._determine_quality_status(overall_score, passed_gates, total_gates)
            }
            
            logger.info(f"Quality Gates Report: Overall Score: {overall_score:.2f}%, "
                       f"Passed: {passed_gates}/{total_gates}, Status: {report['quality_status']}")
            
            # Log detailed report in debug mode
            logger.debug(f"Detailed Quality Report: {json.dumps(report, indent=2)}")
    
    def _determine_quality_status(self, overall_score: float, passed_gates: int, total_gates: int) -> str:
        """Determine overall quality status."""
        if self.strict_mode:
            # In strict mode, all gates must pass
            if passed_gates == total_gates and overall_score >= 95.0:
                return "excellent"
            elif passed_gates == total_gates and overall_score >= 85.0:
                return "good"
            elif passed_gates >= total_gates * 0.8:
                return "acceptable"
            else:
                return "poor"
        else:
            # In normal mode, use score-based evaluation
            if overall_score >= 95.0:
                return "excellent"
            elif overall_score >= 85.0:
                return "good"
            elif overall_score >= 70.0:
                return "acceptable"
            else:
                return "poor"
    
    async def _check_critical_issues(self):
        """Check for critical quality issues that require immediate attention."""
        if not self.gate_results_history:
            return
        
        # Get latest results
        latest_results = {}
        for result in reversed(self.gate_results_history):
            if result.gate_name not in latest_results:
                latest_results[result.gate_name] = result
        
        critical_issues = []
        
        for gate_name, result in latest_results.items():
            # Check for critical failures
            if not result.passed:
                if gate_name == "compliance_automation":
                    # Compliance failures are always critical
                    critical_issues.append(f"Compliance gate failed: {gate_name} (score: {result.score:.2f}%)")
                elif gate_name == "reliability_engineering" and result.score < 99.0:
                    # Reliability below 99% is critical
                    critical_issues.append(f"Reliability below critical threshold: {result.score:.2f}%")
                elif result.score < 50.0:
                    # Any gate with score below 50% is critical
                    critical_issues.append(f"Critical quality failure: {gate_name} (score: {result.score:.2f}%)")
        
        # Alert on critical issues
        if critical_issues:
            logger.critical(f"CRITICAL QUALITY ISSUES DETECTED: {len(critical_issues)} issues found")
            for issue in critical_issues:
                logger.critical(f"  - {issue}")
            
            # In strict mode, consider stopping the system
            if self.strict_mode and len(critical_issues) > 2:
                logger.critical("Multiple critical quality issues in strict mode - consider system shutdown")
    
    async def run_comprehensive_test_suite(self, test_filter: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive test suite across all components."""
        logger.info("Running comprehensive test suite...")
        start_time = datetime.now()
        
        try:
            # Define test filter function
            if test_filter:
                filter_func = lambda tc: test_filter.lower() in tc.name.lower() or test_filter in tc.tags
            else:
                filter_func = None
            
            # Run test suite
            test_results = await self.testing_framework.execute_test_suite(
                test_filter=filter_func,
                parallel=True
            )
            
            # Calculate test metrics
            total_tests = len(test_results)
            passed_tests = sum(1 for result in test_results.values() if result.success)
            failed_tests = total_tests - passed_tests
            
            total_time = sum(result.execution_time for result in test_results.values())
            avg_time = total_time / total_tests if total_tests > 0 else 0.0
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_execution_time": total_time,
                "average_test_time": avg_time,
                "suite_execution_time": execution_time,
                "test_filter": test_filter,
                "failed_test_details": [
                    {
                        "test_id": test_id,
                        "error": result.error_message,
                        "execution_time": result.execution_time
                    } for test_id, result in test_results.items() if not result.success
                ]
            }
            
            logger.info(f"Test suite completed: {passed_tests}/{total_tests} passed "
                       f"({summary['success_rate']:.1f}%) in {execution_time:.2f}s")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error running comprehensive test suite: {e}")
            raise
    
    async def generate_quality_certification_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality certification report."""
        logger.info("Generating quality certification report...")
        
        try:
            # Get summaries from all systems
            quality_summary = self.quality_monitor.get_quality_summary()
            performance_summary = self.performance_optimizer.get_optimization_summary()
            compliance_summary = self.compliance_automation.get_compliance_summary()
            reliability_summary = self.reliability_engine.get_reliability_summary()
            testing_summary = self.testing_framework.get_test_summary()
            
            # Calculate certification scores
            quality_score = self._calculate_quality_certification_score(quality_summary)
            performance_score = self._calculate_performance_certification_score(performance_summary)
            compliance_score = compliance_summary.get("overall_compliance_score", 0)
            reliability_score = reliability_summary.get("overall_reliability_score", 0)
            testing_score = testing_summary.get("overall_success_rate", 0)
            
            # Calculate overall certification score
            weights = {
                "quality": 0.25,
                "performance": 0.20,
                "compliance": 0.25,
                "reliability": 0.25,
                "testing": 0.05
            }
            
            overall_score = (
                quality_score * weights["quality"] +
                performance_score * weights["performance"] +
                compliance_score * weights["compliance"] +
                reliability_score * weights["reliability"] +
                testing_score * weights["testing"]
            )
            
            # Determine certification level
            if overall_score >= 99.0:
                certification_level = "PLATINUM"
            elif overall_score >= 95.0:
                certification_level = "GOLD"
            elif overall_score >= 90.0:
                certification_level = "SILVER"
            elif overall_score >= 80.0:
                certification_level = "BRONZE"
            else:
                certification_level = "NOT_CERTIFIED"
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "certification_level": certification_level,
                "overall_score": overall_score,
                "component_scores": {
                    "quality_monitoring": quality_score,
                    "performance_optimization": performance_score,
                    "compliance_automation": compliance_score,
                    "reliability_engineering": reliability_score,
                    "testing_framework": testing_score
                },
                "certification_criteria": {
                    "PLATINUM": "≥99% overall score",
                    "GOLD": "≥95% overall score",
                    "SILVER": "≥90% overall score",
                    "BRONZE": "≥80% overall score"
                },
                "detailed_summaries": {
                    "quality": quality_summary,
                    "performance": performance_summary,
                    "compliance": compliance_summary,
                    "reliability": reliability_summary,
                    "testing": testing_summary
                },
                "recommendations": self._generate_improvement_recommendations(
                    quality_score, performance_score, compliance_score, 
                    reliability_score, testing_score
                )
            }
            
            logger.info(f"Quality Certification: {certification_level} ({overall_score:.2f}%)")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating quality certification report: {e}")
            raise
    
    def _calculate_quality_certification_score(self, quality_summary: Dict[str, Any]) -> float:
        """Calculate quality monitoring certification score."""
        if not quality_summary.get("metrics"):
            return 0.0
        
        total_metrics = len(quality_summary["metrics"])
        compliant_metrics = 0
        
        for metric_data in quality_summary["metrics"].values():
            current_value = metric_data.get("current_value", 0)
            target_value = metric_data.get("target_value", 100)
            
            if current_value >= target_value * 0.9:  # 90% of target
                compliant_metrics += 1
        
        return (compliant_metrics / total_metrics * 100) if total_metrics > 0 else 0
    
    def _calculate_performance_certification_score(self, performance_summary: Dict[str, Any]) -> float:
        """Calculate performance optimization certification score."""
        metrics = performance_summary.get("current_performance_metrics", {})
        if not metrics:
            return 0.0
        
        score_components = []
        for metric_name, metric_data in metrics.items():
            current_value = metric_data.get("current_value", 0)
            target_value = metric_data.get("target_value", 100)
            
            # Calculate performance score
            if metric_name in ["response_time", "error_rate"]:
                # Lower is better
                score = max(0, min(100, (1 - current_value / target_value) * 100))
            else:
                # Higher is better
                score = max(0, min(100, (current_value / target_value) * 100))
            
            score_components.append(score)
        
        return sum(score_components) / len(score_components) if score_components else 0
    
    def _generate_improvement_recommendations(self, quality_score: float, performance_score: float,
                                           compliance_score: float, reliability_score: float,
                                           testing_score: float) -> List[str]:
        """Generate improvement recommendations based on scores."""
        recommendations = []
        
        if quality_score < 90:
            recommendations.append("Improve quality monitoring coverage and metric targets")
        
        if performance_score < 90:
            recommendations.append("Optimize system performance and resource utilization")
        
        if compliance_score < 95:
            recommendations.append("Address compliance violations and strengthen security measures")
        
        if reliability_score < 99:
            recommendations.append("Enhance system reliability and implement additional self-healing")
        
        if testing_score < 85:
            recommendations.append("Expand test coverage and improve test stability")
        
        # Overall recommendations
        overall_avg = (quality_score + performance_score + compliance_score + reliability_score + testing_score) / 5
        
        if overall_avg < 85:
            recommendations.append("Implement comprehensive quality improvement program")
        elif overall_avg < 95:
            recommendations.append("Focus on continuous improvement in identified weak areas")
        
        return recommendations
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        # Get latest gate results
        latest_results = {}
        for result in reversed(self.gate_results_history):
            if result.gate_name not in latest_results:
                latest_results[result.gate_name] = result
        
        # Calculate overall status
        if latest_results:
            overall_score = sum(result.score for result in latest_results.values()) / len(latest_results)
            passed_gates = sum(1 for result in latest_results.values() if result.passed)
            total_gates = len(latest_results)
        else:
            overall_score = 0.0
            passed_gates = 0
            total_gates = 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "quality_gates_active": self.quality_gates_active,
            "strict_mode": self.strict_mode,
            "target_reliability": self.target_reliability.value,
            "overall_quality_score": overall_score,
            "gates_passed": passed_gates,
            "total_gates": total_gates,
            "quality_status": self._determine_quality_status(overall_score, passed_gates, total_gates),
            "system_status": {
                "quality_monitor_active": self.quality_monitor.monitoring_active,
                "performance_optimizer_active": self.performance_optimizer.optimization_active,
                "compliance_automation_active": self.compliance_automation.monitoring_active,
                "reliability_engine_active": self.reliability_engine.monitoring_active
            },
            "recent_evaluations": len(self.gate_results_history),
            "last_evaluation": self.gate_results_history[-1].timestamp.isoformat() if self.gate_results_history else None
        }