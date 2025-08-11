"""Comprehensive system health monitoring and validation for Fleet-Mind."""

import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .logging import get_logger
from ..coordination.swarm_coordinator import SwarmCoordinator
from ..fleet.drone_fleet import DroneFleet
from ..communication.webrtc_streamer import WebRTCStreamer
from ..communication.latent_encoder import LatentEncoder
from ..planning.llm_planner import LLMPlanner


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    status: HealthStatus
    score: float  # 0.0 to 1.0, where 1.0 is perfect health
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)
    check_duration_ms: float = 0.0


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    overall_status: HealthStatus
    overall_score: float
    components: Dict[str, ComponentHealth]
    integration_tests: Dict[str, bool]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


class SystemHealthMonitor:
    """Comprehensive health monitoring for the Fleet-Mind system."""
    
    def __init__(self):
        self.logger = get_logger("system_health", component="health_monitoring")
        
        # Health check functions
        self.health_checkers: Dict[str, Callable] = {}
        self.integration_tests: Dict[str, Callable] = {}
        
        # Health history
        self.health_history: List[SystemHealthReport] = []
        
        # Performance baselines
        self.performance_baselines = {
            'planning_time_ms': 1000.0,  # Expected planning time
            'encoding_time_ms': 100.0,   # Expected encoding time
            'communication_latency_ms': 50.0,  # Expected communication latency
            'fleet_response_time_ms': 200.0,   # Expected fleet response time
        }
        
        self._initialize_health_checkers()
        self._initialize_integration_tests()
    
    def _initialize_health_checkers(self) -> None:
        """Initialize component health checkers."""
        self.health_checkers = {
            'swarm_coordinator': self._check_swarm_coordinator_health,
            'drone_fleet': self._check_drone_fleet_health,
            'webrtc_streamer': self._check_webrtc_streamer_health,
            'latent_encoder': self._check_latent_encoder_health,
            'llm_planner': self._check_llm_planner_health,
        }
    
    def _initialize_integration_tests(self) -> None:
        """Initialize integration tests."""
        self.integration_tests = {
            'coordinator_fleet_integration': self._test_coordinator_fleet_integration,
            'communication_flow': self._test_communication_flow,
            'encoding_pipeline': self._test_encoding_pipeline,
            'planning_to_execution': self._test_planning_to_execution,
            'end_to_end_mission': self._test_end_to_end_mission,
        }
    
    async def perform_full_health_check(
        self,
        coordinator: Optional[SwarmCoordinator] = None,
        fleet: Optional[DroneFleet] = None,
        webrtc: Optional[WebRTCStreamer] = None,
        encoder: Optional[LatentEncoder] = None,
        planner: Optional[LLMPlanner] = None
    ) -> SystemHealthReport:
        """Perform comprehensive health check of all system components."""
        self.logger.info("Starting comprehensive system health check")
        start_time = time.time()
        
        components = {}
        integration_results = {}
        performance_metrics = {}
        overall_recommendations = []
        
        # Component health checks
        if coordinator:
            components['swarm_coordinator'] = await self._check_swarm_coordinator_health(coordinator)
        if fleet:
            components['drone_fleet'] = await self._check_drone_fleet_health(fleet)
        if webrtc:
            components['webrtc_streamer'] = await self._check_webrtc_streamer_health(webrtc)
        if encoder:
            components['latent_encoder'] = await self._check_latent_encoder_health(encoder)
        if planner:
            components['llm_planner'] = await self._check_llm_planner_health(planner)
        
        # Integration tests
        try:
            if coordinator and fleet:
                integration_results['coordinator_fleet_integration'] = await self._test_coordinator_fleet_integration(coordinator, fleet)
            if webrtc and encoder:
                integration_results['communication_flow'] = await self._test_communication_flow(webrtc, encoder)
            if encoder:
                integration_results['encoding_pipeline'] = await self._test_encoding_pipeline(encoder)
            if coordinator and planner:
                integration_results['planning_to_execution'] = await self._test_planning_to_execution(coordinator, planner)
            if all([coordinator, fleet, webrtc, encoder, planner]):
                integration_results['end_to_end_mission'] = await self._test_end_to_end_mission(
                    coordinator, fleet, webrtc, encoder, planner
                )
        except Exception as e:
            self.logger.error(f"Integration tests failed: {e}")
            integration_results['integration_error'] = False
        
        # Performance metrics
        performance_metrics = {
            'total_check_time_ms': (time.time() - start_time) * 1000,
            'components_checked': len(components),
            'integration_tests_run': len(integration_results),
            'integration_success_rate': sum(integration_results.values()) / len(integration_results) if integration_results else 0.0,
        }
        
        # Calculate overall health
        overall_score = self._calculate_overall_health_score(components)
        overall_status = self._determine_overall_status(components, integration_results)
        
        # Generate recommendations
        overall_recommendations = self._generate_system_recommendations(components, integration_results)
        
        report = SystemHealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            components=components,
            integration_tests=integration_results,
            performance_metrics=performance_metrics,
            recommendations=overall_recommendations
        )
        
        # Store in history
        self.health_history.append(report)
        if len(self.health_history) > 50:  # Keep last 50 reports
            self.health_history = self.health_history[-50:]
        
        self.logger.info(
            f"Health check completed: {overall_status.value} "
            f"(score: {overall_score:.2f}, duration: {performance_metrics['total_check_time_ms']:.1f}ms)"
        )
        
        return report
    
    async def _check_swarm_coordinator_health(self, coordinator: SwarmCoordinator) -> ComponentHealth:
        """Check SwarmCoordinator health."""
        start_time = time.time()
        issues = []
        recommendations = []
        
        try:
            # Check if coordinator is properly initialized
            if not hasattr(coordinator, 'llm_planner') or coordinator.llm_planner is None:
                issues.append("LLM planner not initialized")
            
            if not hasattr(coordinator, 'fleet') or coordinator.fleet is None:
                issues.append("No fleet connected")
                recommendations.append("Connect a drone fleet before starting operations")
            
            # Check performance metrics
            stats = coordinator.get_comprehensive_stats()
            system_health = stats.get('system_health', {})
            
            error_rate = system_health.get('error_rate', 0.0)
            if error_rate > 0.05:  # 5% error rate threshold
                issues.append(f"High error rate: {error_rate:.1%}")
                recommendations.append("Investigate and resolve recurring errors")
            
            memory_usage = system_health.get('memory_usage_mb', 0)
            if memory_usage > 500:  # 500MB threshold
                issues.append(f"High memory usage: {memory_usage:.1f}MB")
                recommendations.append("Consider restarting coordinator to free memory")
            
            # Check mission status
            if coordinator.mission_status.value == 'failed':
                issues.append("Last mission failed")
                recommendations.append("Review mission logs and retry with corrected parameters")
            
            # Calculate health score
            score = 1.0
            if issues:
                score -= len(issues) * 0.2
            score = max(0.0, score)
            
            status = HealthStatus.HEALTHY
            if score < 0.3:
                status = HealthStatus.CRITICAL
            elif score < 0.6:
                status = HealthStatus.WARNING
            elif issues:
                status = HealthStatus.WARNING
                
        except Exception as e:
            issues.append(f"Health check failed: {e}")
            recommendations.append("Restart SwarmCoordinator component")
            score = 0.0
            status = HealthStatus.FAILED
        
        return ComponentHealth(
            name="SwarmCoordinator",
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            check_duration_ms=(time.time() - start_time) * 1000
        )
    
    async def _check_drone_fleet_health(self, fleet: DroneFleet) -> ComponentHealth:
        """Check DroneFleet health."""
        start_time = time.time()
        issues = []
        recommendations = []
        
        try:
            # Check drone status distribution
            active_count = len(fleet.get_active_drones())
            failed_count = len(fleet.get_failed_drones())
            total_count = len(fleet.drone_ids)
            
            if active_count == 0:
                issues.append("No active drones")
                recommendations.append("Check drone connectivity and power status")
            elif active_count < total_count * 0.7:  # Less than 70% active
                issues.append(f"Low drone availability: {active_count}/{total_count} active")
                recommendations.append("Investigate failed drones and attempt recovery")
            
            # Check fleet health metrics
            avg_battery = fleet.get_average_battery()
            if avg_battery < 20.0:
                issues.append(f"Low fleet battery: {avg_battery:.1f}%")
                recommendations.append("Schedule battery maintenance or replacement")
            
            avg_health = fleet.get_average_health()
            if avg_health < 0.6:
                issues.append(f"Poor fleet health: {avg_health:.2f}")
                recommendations.append("Perform fleet diagnostics and maintenance")
            
            # Check auto-healing status if available
            if hasattr(fleet, 'get_healing_status'):
                healing_status = fleet.get_healing_status()
                healing_count = healing_status.get('drones_under_healing', 0)
                if healing_count > total_count * 0.3:  # More than 30% under healing
                    issues.append(f"High healing activity: {healing_count} drones")
                    recommendations.append("Investigate recurring drone issues")
            
            # Calculate health score
            score = (active_count / total_count) * 0.5 + (avg_battery / 100) * 0.3 + avg_health * 0.2
            score = max(0.0, min(1.0, score))
            
            status = HealthStatus.HEALTHY
            if score < 0.3:
                status = HealthStatus.CRITICAL
            elif score < 0.6:
                status = HealthStatus.WARNING
            elif issues:
                status = HealthStatus.WARNING
                
        except Exception as e:
            issues.append(f"Health check failed: {e}")
            recommendations.append("Restart DroneFleet component")
            score = 0.0
            status = HealthStatus.FAILED
        
        return ComponentHealth(
            name="DroneFleet",
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            check_duration_ms=(time.time() - start_time) * 1000
        )
    
    async def _check_webrtc_streamer_health(self, webrtc: WebRTCStreamer) -> ComponentHealth:
        """Check WebRTCStreamer health."""
        start_time = time.time()
        issues = []
        recommendations = []
        
        try:
            if not webrtc.is_initialized:
                issues.append("WebRTC streamer not initialized")
                recommendations.append("Initialize WebRTC streamer with drone list")
                score = 0.0
                status = HealthStatus.FAILED
            else:
                # Check connection status
                status_info = webrtc.get_status()
                active_connections = status_info.get('active_connections', 0)
                total_connections = status_info.get('total_connections', 0)
                
                if active_connections == 0:
                    issues.append("No active WebRTC connections")
                    recommendations.append("Check network connectivity and drone status")
                elif active_connections < total_connections * 0.8:  # Less than 80% connected
                    issues.append(f"Connection issues: {active_connections}/{total_connections} connected")
                    recommendations.append("Investigate network issues or drone connectivity")
                
                # Check communication metrics
                avg_latency = status_info.get('average_latency_ms', 0)
                if avg_latency > self.performance_baselines['communication_latency_ms'] * 2:
                    issues.append(f"High communication latency: {avg_latency:.1f}ms")
                    recommendations.append("Check network quality and reduce message size")
                
                packet_loss = status_info.get('average_packet_loss', 0)
                if packet_loss > 0.05:  # 5% packet loss threshold
                    issues.append(f"High packet loss: {packet_loss:.1%}")
                    recommendations.append("Investigate network stability")
                
                # Check failed drones if available
                if hasattr(webrtc, 'failed_drones') and webrtc.failed_drones:
                    failed_count = len(webrtc.failed_drones)
                    issues.append(f"Failed drone connections: {failed_count}")
                    recommendations.append("Attempt reconnection to failed drones")
                
                # Calculate health score
                connection_ratio = active_connections / max(1, total_connections)
                latency_score = max(0, 1 - (avg_latency / 100))  # Score decreases with latency
                packet_score = max(0, 1 - packet_loss * 10)  # Score decreases with packet loss
                
                score = connection_ratio * 0.5 + latency_score * 0.3 + packet_score * 0.2
                score = max(0.0, min(1.0, score))
                
                status = HealthStatus.HEALTHY
                if score < 0.3:
                    status = HealthStatus.CRITICAL
                elif score < 0.6:
                    status = HealthStatus.WARNING
                elif issues:
                    status = HealthStatus.WARNING
                    
        except Exception as e:
            issues.append(f"Health check failed: {e}")
            recommendations.append("Restart WebRTCStreamer component")
            score = 0.0
            status = HealthStatus.FAILED
        
        return ComponentHealth(
            name="WebRTCStreamer",
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            check_duration_ms=(time.time() - start_time) * 1000
        )
    
    async def _check_latent_encoder_health(self, encoder: LatentEncoder) -> ComponentHealth:
        """Check LatentEncoder health."""
        start_time = time.time()
        issues = []
        recommendations = []
        
        try:
            # Check encoder initialization
            if encoder.compression_type is None:
                issues.append("Compression type not set")
                recommendations.append("Initialize encoder with proper compression type")
            
            # Test basic encoding functionality
            test_input = "test encoding input"
            try:
                encoded = encoder.encode(test_input)
                if encoded is None or (hasattr(encoded, '__len__') and len(encoded) == 0):
                    issues.append("Encoder produces empty results")
                    recommendations.append("Check encoder model initialization")
                else:
                    # Test decoding
                    try:
                        decoded = encoder.decode(encoded)
                        if decoded is None:
                            issues.append("Decoder produces empty results")
                    except Exception as e:
                        issues.append(f"Decoding failed: {e}")
                        recommendations.append("Check decoder configuration")
            except Exception as e:
                issues.append(f"Encoding test failed: {e}")
                recommendations.append("Reinitialize encoder or check model files")
            
            # Check performance metrics
            stats = encoder.get_compression_stats()
            avg_encoding_time = stats.get('average_encoding_time_ms', 0)
            if avg_encoding_time > self.performance_baselines['encoding_time_ms'] * 2:
                issues.append(f"Slow encoding: {avg_encoding_time:.1f}ms")
                recommendations.append("Consider using faster compression method")
            
            # Check error rate if available
            if hasattr(encoder, 'encoding_errors'):
                if encoder.encoding_errors > 5:
                    issues.append(f"High encoding error count: {encoder.encoding_errors}")
                    recommendations.append("Investigate encoding issues and restart if needed")
            
            # Calculate health score
            score = 1.0
            if issues:
                score -= len(issues) * 0.25
            score = max(0.0, score)
            
            status = HealthStatus.HEALTHY
            if score < 0.3:
                status = HealthStatus.CRITICAL
            elif score < 0.6:
                status = HealthStatus.WARNING
            elif issues:
                status = HealthStatus.WARNING
                
        except Exception as e:
            issues.append(f"Health check failed: {e}")
            recommendations.append("Restart LatentEncoder component")
            score = 0.0
            status = HealthStatus.FAILED
        
        return ComponentHealth(
            name="LatentEncoder",
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            check_duration_ms=(time.time() - start_time) * 1000
        )
    
    async def _check_llm_planner_health(self, planner: LLMPlanner) -> ComponentHealth:
        """Check LLMPlanner health."""
        start_time = time.time()
        issues = []
        recommendations = []
        
        try:
            # Check planner initialization
            if not planner.model:
                issues.append("LLM model not specified")
                recommendations.append("Configure LLM model name")
            
            # Check performance stats
            stats = planner.get_performance_stats()
            success_rate = stats.get('success_rate', 0)
            if success_rate < 0.8:  # 80% success threshold
                issues.append(f"Low planning success rate: {success_rate:.1%}")
                recommendations.append("Check LLM API connectivity and prompts")
            
            avg_planning_time = stats.get('average_planning_time_ms', 0)
            if avg_planning_time > self.performance_baselines['planning_time_ms'] * 2:
                issues.append(f"Slow planning: {avg_planning_time:.1f}ms")
                recommendations.append("Optimize prompts or use faster LLM model")
            
            # Check consecutive failures if available
            consecutive_failures = stats.get('consecutive_failures', 0)
            if consecutive_failures > 3:
                issues.append(f"Multiple consecutive failures: {consecutive_failures}")
                recommendations.append("Check LLM API status and authentication")
            
            # Test basic planning functionality
            try:
                test_context = {
                    'mission': 'test mission planning',
                    'num_drones': 1,
                    'constraints': {},
                    'drone_capabilities': {},
                    'current_state': {},
                }
                # Note: We don't actually call generate_plan in health check to avoid API costs
                # In production, this might be a lightweight test call
                
            except Exception as e:
                issues.append(f"Planning test setup failed: {e}")
            
            # Calculate health score
            score = success_rate * 0.6 + (1 - consecutive_failures / 10) * 0.4
            score = max(0.0, min(1.0, score))
            
            status = HealthStatus.HEALTHY
            if score < 0.3:
                status = HealthStatus.CRITICAL
            elif score < 0.6:
                status = HealthStatus.WARNING
            elif issues:
                status = HealthStatus.WARNING
                
        except Exception as e:
            issues.append(f"Health check failed: {e}")
            recommendations.append("Restart LLMPlanner component")
            score = 0.0
            status = HealthStatus.FAILED
        
        return ComponentHealth(
            name="LLMPlanner",
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            check_duration_ms=(time.time() - start_time) * 1000
        )
    
    # Integration test methods
    async def _test_coordinator_fleet_integration(self, coordinator: SwarmCoordinator, fleet: DroneFleet) -> bool:
        """Test integration between coordinator and fleet."""
        try:
            # Check if coordinator has fleet reference
            if coordinator.fleet != fleet:
                return False
            
            # Check if fleet monitoring is active
            if not fleet._running:
                return False
            
            # Check if coordinator can get fleet status
            status = await coordinator.get_swarm_status()
            if not status or 'fleet_health' not in status:
                return False
            
            return True
        except Exception:
            return False
    
    async def _test_communication_flow(self, webrtc: WebRTCStreamer, encoder: LatentEncoder) -> bool:
        """Test communication flow between components."""
        try:
            if not webrtc.is_initialized:
                return False
            
            # Test encoding and communication pipeline
            test_data = "test communication"
            encoded = encoder.encode(test_data)
            if encoded is None:
                return False
            
            # Test broadcast (this is a dry run test)
            drone_list = list(webrtc.active_drones)[:1]  # Test with first drone
            if drone_list:
                result = await webrtc.broadcast(encoded, target_drones=drone_list)
                return any(result.values())
            
            return True
        except Exception:
            return False
    
    async def _test_encoding_pipeline(self, encoder: LatentEncoder) -> bool:
        """Test encoding/decoding pipeline."""
        try:
            test_inputs = [
                "simple text",
                ["action1", "action2", "action3"],
                {"type": "complex", "data": [1, 2, 3]}
            ]
            
            for test_input in test_inputs:
                encoded = encoder.encode(test_input)
                if encoded is None:
                    return False
                
                decoded = encoder.decode(encoded)
                if decoded is None:
                    return False
            
            return True
        except Exception:
            return False
    
    async def _test_planning_to_execution(self, coordinator: SwarmCoordinator, planner: LLMPlanner) -> bool:
        """Test planning to execution pipeline."""
        try:
            # Check if coordinator uses the planner
            if coordinator.llm_planner != planner:
                return False
            
            # Check if coordinator can access planner stats
            stats = planner.get_performance_stats()
            if not stats:
                return False
            
            return True
        except Exception:
            return False
    
    async def _test_end_to_end_mission(self, coordinator: SwarmCoordinator, fleet: DroneFleet, 
                                     webrtc: WebRTCStreamer, encoder: LatentEncoder, 
                                     planner: LLMPlanner) -> bool:
        """Test end-to-end mission flow."""
        try:
            # Verify all components are properly connected
            if coordinator.fleet != fleet:
                return False
            if coordinator.llm_planner != planner:
                return False
            if not webrtc.is_initialized:
                return False
            
            # Check that basic mission planning would work
            # (without actually executing to avoid side effects)
            if fleet.get_active_drones() and coordinator.mission_status.value in ['idle', 'completed']:
                return True
            
            return False
        except Exception:
            return False
    
    def _calculate_overall_health_score(self, components: Dict[str, ComponentHealth]) -> float:
        """Calculate overall system health score."""
        if not components:
            return 0.0
        
        total_score = sum(comp.score for comp in components.values())
        return total_score / len(components)
    
    def _determine_overall_status(self, components: Dict[str, ComponentHealth], 
                                integration_results: Dict[str, bool]) -> HealthStatus:
        """Determine overall system status."""
        if not components:
            return HealthStatus.UNKNOWN
        
        # Check for any failed components
        if any(comp.status == HealthStatus.FAILED for comp in components.values()):
            return HealthStatus.FAILED
        
        # Check for any critical components
        if any(comp.status == HealthStatus.CRITICAL for comp in components.values()):
            return HealthStatus.CRITICAL
        
        # Check integration test results
        if integration_results and not all(integration_results.values()):
            return HealthStatus.WARNING
        
        # Check for warning components
        if any(comp.status == HealthStatus.WARNING for comp in components.values()):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _generate_system_recommendations(self, components: Dict[str, ComponentHealth], 
                                       integration_results: Dict[str, bool]) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        # Collect component recommendations
        for comp in components.values():
            recommendations.extend(comp.recommendations)
        
        # Add integration-specific recommendations
        if integration_results:
            failed_tests = [test for test, result in integration_results.items() if not result]
            if failed_tests:
                recommendations.append(f"Failed integration tests: {', '.join(failed_tests)}")
                recommendations.append("Review component connections and initialization order")
        
        # Add system-wide recommendations based on patterns
        critical_components = [comp.name for comp in components.values() 
                             if comp.status == HealthStatus.CRITICAL]
        if len(critical_components) > 1:
            recommendations.append("Multiple critical components detected - consider system restart")
        
        warning_components = [comp.name for comp in components.values() 
                            if comp.status == HealthStatus.WARNING]
        if len(warning_components) > len(components) / 2:
            recommendations.append("System-wide performance issues detected")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health checks."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health checks performed yet"}
        
        latest = self.health_history[-1]
        
        return {
            "latest_status": latest.overall_status.value,
            "latest_score": latest.overall_score,
            "total_checks": len(self.health_history),
            "latest_check_time": latest.timestamp,
            "components_monitored": len(latest.components),
            "active_issues": sum(len(comp.issues) for comp in latest.components.values()),
            "trend": self._calculate_health_trend()
        }
    
    def _calculate_health_trend(self) -> str:
        """Calculate health trend from recent history."""
        if len(self.health_history) < 2:
            return "unknown"
        
        recent_scores = [report.overall_score for report in self.health_history[-5:]]
        
        if len(recent_scores) >= 2:
            if recent_scores[-1] > recent_scores[0] + 0.1:
                return "improving"
            elif recent_scores[-1] < recent_scores[0] - 0.1:
                return "declining"
        
        return "stable"