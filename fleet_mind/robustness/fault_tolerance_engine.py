"""Fault Tolerance Engine for Robust Drone Swarm Operations.

This module provides comprehensive fault tolerance capabilities:
- Distributed system failure detection and recovery
- Byzantine fault tolerance for adversarial environments
- Self-healing mechanisms and automatic recovery
- Redundancy management and failover coordination
- System resilience monitoring and adaptation
"""

import asyncio
import time
import random
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of system failures."""
    COMMUNICATION_LOSS = "communication_loss"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_CRASH = "software_crash"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class RecoveryStrategy(Enum):
    """Failure recovery strategies."""
    RESTART = "restart"
    FAILOVER = "failover"
    REDUNDANCY_SWITCH = "redundancy_switch"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ISOLATION = "isolation"
    RECONFIGURATION = "reconfiguration"
    SELF_HEALING = "self_healing"


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


@dataclass
class SystemNode:
    """Represents a system node (drone, server, etc.)."""
    node_id: str
    node_type: str
    
    # State information
    state: SystemState = SystemState.HEALTHY
    last_heartbeat: float = field(default_factory=time.time)
    health_score: float = 1.0
    
    # Capabilities and resources
    capabilities: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    max_resources: Dict[str, float] = field(default_factory=dict)
    
    # Failure tracking
    failure_count: int = 0
    last_failure: Optional[float] = None
    failure_types: List[FailureType] = field(default_factory=list)
    
    # Recovery information
    recovery_attempts: int = 0
    last_recovery: Optional[float] = None
    recovery_success_rate: float = 1.0
    
    # Connections and dependencies
    connections: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class FailureEvent:
    """Records a system failure event."""
    event_id: str
    node_id: str
    failure_type: FailureType
    timestamp: float = field(default_factory=time.time)
    
    # Failure details
    severity: float = 1.0  # 0.0 (minor) to 1.0 (critical)
    description: str = ""
    symptoms: List[str] = field(default_factory=list)
    
    # Impact assessment
    affected_nodes: Set[str] = field(default_factory=set)
    service_impact: Dict[str, float] = field(default_factory=dict)
    
    # Recovery information
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_started: Optional[float] = None
    recovery_completed: Optional[float] = None
    recovery_successful: bool = False


@dataclass
class RedundancyGroup:
    """Manages redundant system components."""
    group_id: str
    primary_node: str
    backup_nodes: List[str] = field(default_factory=list)
    
    # Redundancy configuration
    min_active_nodes: int = 1
    max_failures_tolerated: int = 1
    
    # State tracking
    active_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    
    # Performance tracking
    load_balancing: bool = True
    performance_weights: Dict[str, float] = field(default_factory=dict)


class FaultToleranceEngine:
    """Comprehensive fault tolerance and recovery system."""
    
    def __init__(self):
        # System topology
        self.nodes: Dict[str, SystemNode] = {}
        self.redundancy_groups: Dict[str, RedundancyGroup] = {}
        
        # Failure tracking
        self.failure_events: deque = deque(maxlen=10000)
        self.failure_patterns: Dict[str, List[FailureEvent]] = defaultdict(list)
        
        # Recovery management
        self.recovery_strategies: Dict[FailureType, List[RecoveryStrategy]] = {}
        self.active_recoveries: Dict[str, FailureEvent] = {}
        
        # Health monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.monitoring_interval: float = 5.0  # seconds
        self.heartbeat_timeout: float = 30.0  # seconds
        
        # Consensus and voting
        self.byzantine_tolerance: int = 1  # Number of Byzantine faults tolerated
        self.consensus_threshold: float = 0.67  # 2/3 majority
        
        # Performance metrics
        self.fault_tolerance_metrics: Dict[str, Any] = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "mean_recovery_time": 0.0,
            "system_availability": 1.0,
            "byzantine_events": 0,
            "false_positives": 0
        }
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        # Start monitoring tasks
        self._start_monitoring()
        
        logger.info("Fault tolerance engine initialized")
    
    def _initialize_recovery_strategies(self) -> None:
        """Initialize default recovery strategies for each failure type."""
        self.recovery_strategies = {
            FailureType.COMMUNICATION_LOSS: [
                RecoveryStrategy.RESTART,
                RecoveryStrategy.REDUNDANCY_SWITCH,
                RecoveryStrategy.RECONFIGURATION
            ],
            FailureType.HARDWARE_FAILURE: [
                RecoveryStrategy.FAILOVER,
                RecoveryStrategy.REDUNDANCY_SWITCH,
                RecoveryStrategy.ISOLATION
            ],
            FailureType.SOFTWARE_CRASH: [
                RecoveryStrategy.RESTART,
                RecoveryStrategy.SELF_HEALING,
                RecoveryStrategy.FAILOVER
            ],
            FailureType.PERFORMANCE_DEGRADATION: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.RECONFIGURATION,
                RecoveryStrategy.REDUNDANCY_SWITCH
            ],
            FailureType.SECURITY_BREACH: [
                RecoveryStrategy.ISOLATION,
                RecoveryStrategy.RESTART,
                RecoveryStrategy.RECONFIGURATION
            ],
            FailureType.BYZANTINE_BEHAVIOR: [
                RecoveryStrategy.ISOLATION,
                RecoveryStrategy.REDUNDANCY_SWITCH,
                RecoveryStrategy.RECONFIGURATION
            ],
            FailureType.NETWORK_PARTITION: [
                RecoveryStrategy.RECONFIGURATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.REDUNDANCY_SWITCH
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.RECONFIGURATION,
                RecoveryStrategy.FAILOVER
            ]
        }
    
    def _start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._failure_detection_loop())
        asyncio.create_task(self._recovery_management_loop())
    
    async def register_node(self, 
                          node_id: str,
                          node_type: str,
                          capabilities: List[str],
                          resources: Dict[str, float]) -> None:
        """Register a new system node."""
        
        node = SystemNode(
            node_id=node_id,
            node_type=node_type,
            capabilities=capabilities,
            resources=resources.copy(),
            max_resources=resources.copy()
        )
        
        self.nodes[node_id] = node
        
        # Register default health check
        self.health_checks[node_id] = self._default_health_check
        
        logger.info(f"Registered node: {node_id} ({node_type})")
    
    async def create_redundancy_group(self,
                                    group_id: str,
                                    primary_node: str,
                                    backup_nodes: List[str],
                                    min_active: int = 1,
                                    max_failures: int = 1) -> None:
        """Create redundancy group for fault tolerance."""
        
        # Validate nodes exist
        all_nodes = [primary_node] + backup_nodes
        for node_id in all_nodes:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not registered")
        
        redundancy_group = RedundancyGroup(
            group_id=group_id,
            primary_node=primary_node,
            backup_nodes=backup_nodes,
            min_active_nodes=min_active,
            max_failures_tolerated=max_failures
        )
        
        # Initialize as all active
        redundancy_group.active_nodes = set(all_nodes)
        
        # Set equal performance weights initially
        for node_id in all_nodes:
            redundancy_group.performance_weights[node_id] = 1.0
        
        self.redundancy_groups[group_id] = redundancy_group
        
        logger.info(f"Created redundancy group: {group_id}")
    
    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered nodes."""
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            # Check heartbeat timeout
            if current_time - node.last_heartbeat > self.heartbeat_timeout:
                await self._handle_node_failure(
                    node_id, 
                    FailureType.COMMUNICATION_LOSS,
                    f"Heartbeat timeout: {current_time - node.last_heartbeat:.1f}s"
                )
                continue
            
            # Run custom health check if available
            if node_id in self.health_checks:
                try:
                    health_result = await self.health_checks[node_id](node)
                    await self._process_health_result(node_id, health_result)
                except Exception as e:
                    logger.error(f"Health check failed for {node_id}: {e}")
                    await self._handle_node_failure(
                        node_id,
                        FailureType.SOFTWARE_CRASH,
                        f"Health check exception: {e}"
                    )
    
    async def _default_health_check(self, node: SystemNode) -> Dict[str, Any]:
        """Default health check implementation."""
        # Simulate health check with some randomness
        health_score = random.uniform(0.8, 1.0)
        
        # Simulate occasional issues
        if random.random() < 0.05:  # 5% chance of issue
            health_score = random.uniform(0.3, 0.7)
        
        return {
            "health_score": health_score,
            "cpu_usage": random.uniform(0.1, 0.8),
            "memory_usage": random.uniform(0.2, 0.9),
            "network_latency": random.uniform(1, 50),
            "error_rate": random.uniform(0.0, 0.05)
        }
    
    async def _process_health_result(self, node_id: str, health_result: Dict[str, Any]) -> None:
        """Process health check results."""
        node = self.nodes[node_id]
        
        # Update health score
        node.health_score = health_result.get("health_score", 1.0)
        
        # Check for performance degradation
        if node.health_score < 0.7:
            await self._handle_node_failure(
                node_id,
                FailureType.PERFORMANCE_DEGRADATION,
                f"Health score degraded to {node.health_score:.2f}"
            )
        
        # Check resource utilization
        cpu_usage = health_result.get("cpu_usage", 0.0)
        memory_usage = health_result.get("memory_usage", 0.0)
        
        if cpu_usage > 0.95 or memory_usage > 0.95:
            await self._handle_node_failure(
                node_id,
                FailureType.RESOURCE_EXHAUSTION,
                f"Resource exhaustion: CPU={cpu_usage:.2f}, Memory={memory_usage:.2f}"
            )
        
        # Update node state based on health
        if node.health_score >= 0.9:
            node.state = SystemState.HEALTHY
        elif node.health_score >= 0.7:
            node.state = SystemState.DEGRADED
        elif node.health_score >= 0.5:
            node.state = SystemState.CRITICAL
        else:
            node.state = SystemState.FAILED
    
    async def _failure_detection_loop(self) -> None:
        """Continuous failure detection and pattern analysis."""
        while True:
            try:
                await self._analyze_failure_patterns()
                await self._detect_byzantine_behavior()
                await asyncio.sleep(10.0)  # Run every 10 seconds
            except Exception as e:
                logger.error(f"Failure detection error: {e}")
                await asyncio.sleep(1.0)
    
    async def _analyze_failure_patterns(self) -> None:
        """Analyze failure patterns and predict potential issues."""
        current_time = time.time()
        
        # Look at recent failures (last hour)
        recent_failures = [
            event for event in self.failure_events
            if current_time - event.timestamp < 3600
        ]
        
        if len(recent_failures) < 3:
            return
        
        # Group failures by type and node
        failure_by_type = defaultdict(list)
        failure_by_node = defaultdict(list)
        
        for event in recent_failures:
            failure_by_type[event.failure_type].append(event)
            failure_by_node[event.node_id].append(event)
        
        # Detect cascade failures
        for failure_type, events in failure_by_type.items():
            if len(events) >= 3:
                logger.warning(f"Potential cascade failure detected: {failure_type.value}")
                await self._handle_cascade_failure(failure_type, events)
        
        # Detect problematic nodes
        for node_id, events in failure_by_node.items():
            if len(events) >= 3:
                logger.warning(f"Problematic node detected: {node_id}")
                await self._handle_problematic_node(node_id, events)
    
    async def _detect_byzantine_behavior(self) -> None:
        """Detect Byzantine (malicious or arbitrary) behavior."""
        
        # Simple Byzantine detection based on consensus disagreement
        for group_id, group in self.redundancy_groups.items():
            if len(group.active_nodes) < 3:
                continue  # Need at least 3 nodes for Byzantine detection
            
            # Simulate consensus check
            consensus_results = {}
            for node_id in group.active_nodes:
                if node_id in self.nodes:
                    # Simulate node consensus vote
                    consensus_results[node_id] = random.choice([True, False])
            
            # Check for Byzantine behavior (minority disagreement)
            true_votes = sum(1 for vote in consensus_results.values() if vote)
            total_votes = len(consensus_results)
            
            if total_votes > 0:
                consensus_ratio = true_votes / total_votes
                
                # If consensus is far from majority, suspect Byzantine behavior
                if abs(consensus_ratio - 0.5) > 0.4:  # Very divided consensus
                    for node_id, vote in consensus_results.items():
                        # Nodes voting in minority might be Byzantine
                        is_minority = (vote and consensus_ratio < 0.5) or (not vote and consensus_ratio > 0.5)
                        
                        if is_minority and random.random() < 0.1:  # 10% chance to flag as Byzantine
                            await self._handle_node_failure(
                                node_id,
                                FailureType.BYZANTINE_BEHAVIOR,
                                f"Byzantine behavior detected: consensus disagreement"
                            )
    
    async def _recovery_management_loop(self) -> None:
        """Manage ongoing recovery processes."""
        while True:
            try:
                await self._check_recovery_progress()
                await asyncio.sleep(5.0)
            except Exception as e:
                logger.error(f"Recovery management error: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_recovery_progress(self) -> None:
        """Check progress of active recovery operations."""
        current_time = time.time()
        completed_recoveries = []
        
        for event_id, failure_event in self.active_recoveries.items():
            if failure_event.recovery_started is None:
                continue
            
            recovery_duration = current_time - failure_event.recovery_started
            
            # Check if recovery has timed out (5 minutes max)
            if recovery_duration > 300:
                logger.warning(f"Recovery timeout for {failure_event.node_id}")
                failure_event.recovery_successful = False
                failure_event.recovery_completed = current_time
                completed_recoveries.append(event_id)
                self.fault_tolerance_metrics["failed_recoveries"] += 1
                continue
            
            # Simulate recovery progress check
            node = self.nodes.get(failure_event.node_id)
            if node and node.state in [SystemState.HEALTHY, SystemState.DEGRADED]:
                # Recovery successful
                failure_event.recovery_successful = True
                failure_event.recovery_completed = current_time
                completed_recoveries.append(event_id)
                
                # Update metrics
                self.fault_tolerance_metrics["successful_recoveries"] += 1
                recovery_time = failure_event.recovery_completed - failure_event.recovery_started
                self._update_mean_recovery_time(recovery_time)
                
                logger.info(f"Recovery successful for {failure_event.node_id} in {recovery_time:.1f}s")
        
        # Clean up completed recoveries
        for event_id in completed_recoveries:
            del self.active_recoveries[event_id]
    
    def _update_mean_recovery_time(self, recovery_time: float) -> None:
        """Update mean recovery time metric."""
        current_mean = self.fault_tolerance_metrics["mean_recovery_time"]
        total_recoveries = self.fault_tolerance_metrics["successful_recoveries"]
        
        # Calculate new mean
        new_mean = ((current_mean * (total_recoveries - 1)) + recovery_time) / total_recoveries
        self.fault_tolerance_metrics["mean_recovery_time"] = new_mean
    
    async def _handle_node_failure(self, 
                                 node_id: str,
                                 failure_type: FailureType,
                                 description: str) -> None:
        """Handle detected node failure."""
        
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Check if this is a duplicate failure event
        if (node.last_failure and 
            time.time() - node.last_failure < 60 and  # Within last minute
            failure_type in node.failure_types[-3:]):  # Same type in recent failures
            return  # Skip duplicate
        
        # Create failure event
        event_id = f"failure_{node_id}_{int(time.time() * 1000)}"
        failure_event = FailureEvent(
            event_id=event_id,
            node_id=node_id,
            failure_type=failure_type,
            description=description,
            severity=self._calculate_failure_severity(failure_type, node)
        )
        
        # Update node failure tracking
        node.failure_count += 1
        node.last_failure = time.time()
        node.failure_types.append(failure_type)
        if len(node.failure_types) > 10:
            node.failure_types = node.failure_types[-10:]  # Keep last 10
        
        # Store failure event
        self.failure_events.append(failure_event)
        self.fault_tolerance_metrics["total_failures"] += 1
        
        # Assess impact on redundancy groups
        affected_groups = self._assess_redundancy_impact(node_id)
        failure_event.affected_nodes = set(affected_groups.keys())
        
        # Select and execute recovery strategy
        recovery_strategy = await self._select_recovery_strategy(failure_event)
        if recovery_strategy:
            await self._execute_recovery_strategy(failure_event, recovery_strategy)
        
        logger.warning(f"Node failure: {node_id} - {failure_type.value} - {description}")
    
    def _calculate_failure_severity(self, failure_type: FailureType, node: SystemNode) -> float:
        """Calculate failure severity based on type and node importance."""
        base_severity = {
            FailureType.COMMUNICATION_LOSS: 0.6,
            FailureType.HARDWARE_FAILURE: 0.9,
            FailureType.SOFTWARE_CRASH: 0.7,
            FailureType.PERFORMANCE_DEGRADATION: 0.4,
            FailureType.SECURITY_BREACH: 1.0,
            FailureType.BYZANTINE_BEHAVIOR: 0.8,
            FailureType.NETWORK_PARTITION: 0.7,
            FailureType.RESOURCE_EXHAUSTION: 0.5
        }.get(failure_type, 0.5)
        
        # Adjust based on node importance (number of dependents)
        importance_factor = min(1.5, 1.0 + len(node.dependents) * 0.1)
        
        return min(1.0, base_severity * importance_factor)
    
    def _assess_redundancy_impact(self, failed_node_id: str) -> Dict[str, float]:
        """Assess impact of node failure on redundancy groups."""
        impact = {}
        
        for group_id, group in self.redundancy_groups.items():
            if failed_node_id in group.active_nodes:
                # Remove from active nodes
                group.active_nodes.discard(failed_node_id)
                group.failed_nodes.add(failed_node_id)
                
                # Calculate impact
                remaining_active = len(group.active_nodes)
                if remaining_active < group.min_active_nodes:
                    impact[group_id] = 1.0  # Critical impact
                elif remaining_active <= group.max_failures_tolerated:
                    impact[group_id] = 0.7  # High impact
                else:
                    impact[group_id] = 0.3  # Moderate impact
        
        return impact
    
    async def _select_recovery_strategy(self, failure_event: FailureEvent) -> Optional[RecoveryStrategy]:
        """Select appropriate recovery strategy for failure."""
        
        strategies = self.recovery_strategies.get(failure_event.failure_type, [])
        if not strategies:
            return None
        
        node = self.nodes[failure_event.node_id]
        
        # Select strategy based on failure history and node state
        for strategy in strategies:
            if await self._is_strategy_applicable(strategy, failure_event, node):
                return strategy
        
        # Fallback to first strategy
        return strategies[0] if strategies else None
    
    async def _is_strategy_applicable(self, 
                                    strategy: RecoveryStrategy,
                                    failure_event: FailureEvent,
                                    node: SystemNode) -> bool:
        """Check if recovery strategy is applicable."""
        
        # Check if strategy has been tried recently
        if (node.last_recovery and 
            time.time() - node.last_recovery < 300 and  # Within 5 minutes
            node.recovery_attempts >= 3):  # Too many recent attempts
            return strategy == RecoveryStrategy.ISOLATION  # Only allow isolation
        
        # Strategy-specific checks
        if strategy == RecoveryStrategy.FAILOVER:
            # Need available backup nodes
            return self._has_available_backups(node.node_id)
        
        elif strategy == RecoveryStrategy.REDUNDANCY_SWITCH:
            # Need redundancy group with available nodes
            return self._has_redundancy_options(node.node_id)
        
        elif strategy == RecoveryStrategy.RESTART:
            # Avoid restart for hardware failures
            return failure_event.failure_type != FailureType.HARDWARE_FAILURE
        
        return True
    
    def _has_available_backups(self, node_id: str) -> bool:
        """Check if node has available backup nodes."""
        for group in self.redundancy_groups.values():
            if node_id == group.primary_node:
                available_backups = [
                    backup for backup in group.backup_nodes
                    if backup in group.active_nodes
                ]
                return len(available_backups) > 0
        return False
    
    def _has_redundancy_options(self, node_id: str) -> bool:
        """Check if node has redundancy options available."""
        for group in self.redundancy_groups.values():
            if node_id in group.active_nodes:
                remaining_nodes = len(group.active_nodes) - 1  # Excluding current node
                return remaining_nodes >= group.min_active_nodes
        return False
    
    async def _execute_recovery_strategy(self, 
                                       failure_event: FailureEvent,
                                       strategy: RecoveryStrategy) -> None:
        """Execute selected recovery strategy."""
        
        failure_event.recovery_strategy = strategy
        failure_event.recovery_started = time.time()
        
        node = self.nodes[failure_event.node_id]
        node.recovery_attempts += 1
        node.last_recovery = time.time()
        
        # Add to active recoveries
        self.active_recoveries[failure_event.event_id] = failure_event
        
        logger.info(f"Executing recovery strategy: {strategy.value} for {failure_event.node_id}")
        
        try:
            if strategy == RecoveryStrategy.RESTART:
                await self._restart_node(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.FAILOVER:
                await self._failover_node(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.REDUNDANCY_SWITCH:
                await self._switch_redundancy(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                await self._graceful_degradation(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.ISOLATION:
                await self._isolate_node(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.RECONFIGURATION:
                await self._reconfigure_system(failure_event.node_id)
            
            elif strategy == RecoveryStrategy.SELF_HEALING:
                await self._self_healing(failure_event.node_id)
            
        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            failure_event.recovery_successful = False
            failure_event.recovery_completed = time.time()
    
    async def _restart_node(self, node_id: str) -> None:
        """Restart failed node."""
        # Simulate restart process
        await asyncio.sleep(2.0)  # Restart delay
        
        node = self.nodes[node_id]
        
        # Simulate restart success/failure
        if random.random() < 0.8:  # 80% success rate
            node.state = SystemState.HEALTHY
            node.health_score = random.uniform(0.8, 1.0)
            node.last_heartbeat = time.time()
            logger.info(f"Node restart successful: {node_id}")
        else:
            logger.warning(f"Node restart failed: {node_id}")
    
    async def _failover_node(self, node_id: str) -> None:
        """Failover to backup node."""
        # Find redundancy group
        for group in self.redundancy_groups.values():
            if node_id == group.primary_node:
                # Find best backup node
                available_backups = [
                    backup for backup in group.backup_nodes
                    if backup in group.active_nodes and backup in self.nodes
                ]
                
                if available_backups:
                    # Select backup with highest performance weight
                    backup_node = max(available_backups, 
                                    key=lambda x: group.performance_weights.get(x, 1.0))
                    
                    # Promote backup to primary
                    group.primary_node = backup_node
                    logger.info(f"Failover: promoted {backup_node} to primary for group {group.group_id}")
                    break
    
    async def _switch_redundancy(self, node_id: str) -> None:
        """Switch to redundant node in group."""
        for group in self.redundancy_groups.values():
            if node_id in group.active_nodes:
                # Remove failed node from active set
                group.active_nodes.discard(node_id)
                group.failed_nodes.add(node_id)
                
                # Rebalance load among remaining nodes
                if group.load_balancing and len(group.active_nodes) > 0:
                    for active_node in group.active_nodes:
                        group.performance_weights[active_node] = 1.0 / len(group.active_nodes)
                
                logger.info(f"Redundancy switch: removed {node_id} from group {group.group_id}")
                break
    
    async def _graceful_degradation(self, node_id: str) -> None:
        """Implement graceful service degradation."""
        node = self.nodes[node_id]
        
        # Reduce node capabilities and resource allocation
        node.resources = {k: v * 0.5 for k, v in node.resources.items()}
        node.capabilities = node.capabilities[:len(node.capabilities)//2]  # Reduce capabilities
        node.state = SystemState.DEGRADED
        
        logger.info(f"Graceful degradation applied to {node_id}")
    
    async def _isolate_node(self, node_id: str) -> None:
        """Isolate problematic node from system."""
        node = self.nodes[node_id]
        
        # Remove from all redundancy groups
        for group in self.redundancy_groups.values():
            group.active_nodes.discard(node_id)
            group.failed_nodes.add(node_id)
        
        # Clear connections
        node.connections.clear()
        node.state = SystemState.FAILED
        
        logger.warning(f"Node isolated: {node_id}")
    
    async def _reconfigure_system(self, node_id: str) -> None:
        """Reconfigure system topology to work around failure."""
        # Simulate system reconfiguration
        await asyncio.sleep(1.0)
        
        # Rebuild connections and dependencies
        for other_node in self.nodes.values():
            if node_id in other_node.connections:
                other_node.connections.discard(node_id)
                
                # Find alternative connections
                for candidate_id, candidate_node in self.nodes.items():
                    if (candidate_id != other_node.node_id and 
                        candidate_node.state == SystemState.HEALTHY and
                        len(other_node.connections) < 3):  # Limit connections
                        other_node.connections.add(candidate_id)
                        break
        
        logger.info(f"System reconfigured around failed node: {node_id}")
    
    async def _self_healing(self, node_id: str) -> None:
        """Attempt self-healing mechanisms."""
        node = self.nodes[node_id]
        
        # Simulate self-healing process
        await asyncio.sleep(3.0)
        
        # Attempt to restore node to healthy state
        if random.random() < 0.6:  # 60% success rate
            node.state = SystemState.HEALTHY
            node.health_score = random.uniform(0.7, 0.9)
            node.last_heartbeat = time.time()
            
            # Clear recent failure types
            node.failure_types = []
            
            logger.info(f"Self-healing successful: {node_id}")
        else:
            logger.warning(f"Self-healing failed: {node_id}")
    
    async def _handle_cascade_failure(self, failure_type: FailureType, events: List[FailureEvent]) -> None:
        """Handle detected cascade failure pattern."""
        logger.critical(f"Cascade failure detected: {failure_type.value}")
        
        # Implement emergency measures
        if failure_type == FailureType.COMMUNICATION_LOSS:
            # Activate emergency communication protocols
            await self._activate_emergency_communication()
        
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            # Emergency resource reallocation
            await self._emergency_resource_reallocation()
        
        # Increase monitoring frequency temporarily
        self.monitoring_interval = 1.0
        asyncio.create_task(self._reset_monitoring_interval())
    
    async def _handle_problematic_node(self, node_id: str, events: List[FailureEvent]) -> None:
        """Handle consistently problematic node."""
        logger.warning(f"Problematic node detected: {node_id}")
        
        node = self.nodes[node_id]
        
        # Reduce node's role in system
        if node.state != SystemState.FAILED:
            await self._graceful_degradation(node_id)
        
        # Consider permanent isolation if too many failures
        if len(events) >= 5:
            await self._isolate_node(node_id)
    
    async def _activate_emergency_communication(self) -> None:
        """Activate emergency communication protocols."""
        # Simulate emergency communication activation
        logger.info("Emergency communication protocols activated")
    
    async def _emergency_resource_reallocation(self) -> None:
        """Perform emergency resource reallocation."""
        # Redistribute resources from healthy nodes
        healthy_nodes = [
            node for node in self.nodes.values()
            if node.state == SystemState.HEALTHY
        ]
        
        if healthy_nodes:
            for node in healthy_nodes:
                # Reduce resource allocation by 20%
                node.resources = {k: v * 0.8 for k, v in node.resources.items()}
        
        logger.info("Emergency resource reallocation completed")
    
    async def _reset_monitoring_interval(self) -> None:
        """Reset monitoring interval after emergency period."""
        await asyncio.sleep(300)  # 5 minutes
        self.monitoring_interval = 5.0
        logger.info("Monitoring interval reset to normal")
    
    async def update_node_heartbeat(self, node_id: str, health_data: Optional[Dict[str, Any]] = None) -> None:
        """Update node heartbeat and health data."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.last_heartbeat = time.time()
        
        if health_data:
            # Update health score if provided
            if "health_score" in health_data:
                node.health_score = health_data["health_score"]
            
            # Update resources if provided
            if "resources" in health_data:
                node.resources.update(health_data["resources"])
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        
        # Count nodes by state
        node_states = defaultdict(int)
        for node in self.nodes.values():
            node_states[node.state.value] += 1
        
        # Calculate system availability
        total_nodes = len(self.nodes)
        healthy_nodes = node_states.get("healthy", 0) + node_states.get("degraded", 0)
        availability = healthy_nodes / total_nodes if total_nodes > 0 else 0.0
        self.fault_tolerance_metrics["system_availability"] = availability
        
        # Recent failure analysis
        recent_failures = [
            event for event in self.failure_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        failure_by_type = defaultdict(int)
        for event in recent_failures:
            failure_by_type[event.failure_type.value] += 1
        
        # Redundancy group status
        redundancy_status = {}
        for group_id, group in self.redundancy_groups.items():
            redundancy_status[group_id] = {
                "primary_node": group.primary_node,
                "active_nodes": len(group.active_nodes),
                "failed_nodes": len(group.failed_nodes),
                "min_required": group.min_active_nodes,
                "status": "healthy" if len(group.active_nodes) >= group.min_active_nodes else "degraded"
            }
        
        return {
            "system_overview": {
                "total_nodes": total_nodes,
                "node_states": dict(node_states),
                "system_availability": availability,
                "active_recoveries": len(self.active_recoveries),
                "recent_failures": len(recent_failures)
            },
            "fault_tolerance_metrics": self.fault_tolerance_metrics.copy(),
            "failure_analysis": {
                "recent_failures_by_type": dict(failure_by_type),
                "failure_patterns_detected": len(self.failure_patterns),
                "byzantine_tolerance": self.byzantine_tolerance
            },
            "redundancy_status": redundancy_status,
            "recovery_status": {
                "active_recoveries": len(self.active_recoveries),
                "mean_recovery_time": self.fault_tolerance_metrics["mean_recovery_time"],
                "success_rate": (
                    self.fault_tolerance_metrics["successful_recoveries"] /
                    max(1, self.fault_tolerance_metrics["successful_recoveries"] + 
                        self.fault_tolerance_metrics["failed_recoveries"])
                )
            }
        }