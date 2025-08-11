"""Comprehensive Performance Monitoring and Benchmarking for Fleet-Mind Generation 3.

This module implements advanced performance monitoring including:
- 1000+ drone scenario benchmarking
- Real-time performance metrics collection
- Scalability analysis and bottleneck detection
- Performance regression testing
- Load testing and stress testing
- Multi-dimensional performance analytics
"""

import asyncio
import time
import json
import statistics
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import concurrent.futures
import random
import math

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..utils.logging import get_logger
from ..coordination.swarm_coordinator import SwarmCoordinator
from ..fleet.drone_fleet import DroneFleet
from ..optimization.ai_performance_optimizer import record_performance_metrics


class BenchmarkType(Enum):
    """Types of performance benchmarks."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SCALABILITY = "scalability"
    STRESS = "stress"
    ENDURANCE = "endurance"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    CACHE = "cache"


class LoadPattern(Enum):
    """Load testing patterns."""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STEP = "step"
    SINE_WAVE = "sine_wave"
    RANDOM = "random"


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks."""
    benchmark_type: BenchmarkType
    duration_seconds: int = 300
    max_drone_count: int = 1000
    load_pattern: LoadPattern = LoadPattern.RAMP_UP
    target_latency_ms: float = 100.0
    target_throughput_rps: float = 1000.0
    ramp_up_duration: int = 60
    steady_state_duration: int = 180
    ramp_down_duration: int = 60
    concurrent_missions: int = 10
    enable_monitoring: bool = True
    save_detailed_results: bool = True


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    timestamp: float
    drone_count: int
    active_missions: int
    cpu_usage_percent: float
    memory_usage_mb: float
    network_throughput_mbps: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    cache_hit_rate: float
    queue_size: int
    connection_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    benchmark_id: str
    config: BenchmarkConfig
    start_time: float
    end_time: float
    snapshots: List[PerformanceSnapshot]
    summary_stats: Dict[str, Any]
    bottlenecks_detected: List[Dict[str, Any]]
    recommendations: List[str]
    success: bool = True
    error_message: Optional[str] = None


class MockDrone:
    """Mock drone for large-scale testing."""
    
    def __init__(self, drone_id: str):
        self.drone_id = drone_id
        self.battery_level = random.uniform(80.0, 100.0)
        self.status = "active"
        self.response_time_ms = random.uniform(10.0, 50.0)
        self.last_update = time.time()
        self.mission_count = 0
        
    def update(self):
        """Update drone state."""
        self.battery_level = max(0, self.battery_level - random.uniform(0.1, 0.3))
        self.response_time_ms = max(5.0, self.response_time_ms + random.uniform(-2.0, 2.0))
        self.last_update = time.time()
        
        if self.battery_level < 20:
            self.status = "low_battery"
        elif self.battery_level < 5:
            self.status = "critical"


class MockFleet:
    """Mock fleet for large-scale testing."""
    
    def __init__(self, initial_size: int = 100):
        self.drones = {}
        self.total_missions = 0
        self.active_missions = 0
        
        # Create initial fleet
        for i in range(initial_size):
            drone_id = f"drone_{i:04d}"
            self.drones[drone_id] = MockDrone(drone_id)
    
    def scale_to_size(self, target_size: int):
        """Scale fleet to target size."""
        current_size = len(self.drones)
        
        if target_size > current_size:
            # Add drones
            for i in range(current_size, target_size):
                drone_id = f"drone_{i:04d}"
                self.drones[drone_id] = MockDrone(drone_id)
        elif target_size < current_size:
            # Remove drones
            drone_ids = list(self.drones.keys())
            for i in range(target_size, current_size):
                if drone_ids[i] in self.drones:
                    del self.drones[drone_ids[i]]
    
    def get_active_drones(self) -> List[str]:
        """Get list of active drone IDs."""
        return [drone_id for drone_id, drone in self.drones.items() 
                if drone.status == "active"]
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get overall fleet status."""
        active_count = len(self.get_active_drones())
        total_count = len(self.drones)
        
        avg_battery = statistics.mean([drone.battery_level for drone in self.drones.values()])
        avg_response_time = statistics.mean([drone.response_time_ms for drone in self.drones.values()])
        
        return {
            "total_drones": total_count,
            "active_drones": active_count,
            "availability": active_count / max(1, total_count),
            "average_battery": avg_battery,
            "average_response_time": avg_response_time,
            "total_missions": self.total_missions,
            "active_missions": self.active_missions,
        }
    
    def simulate_mission(self) -> Dict[str, Any]:
        """Simulate a mission execution."""
        start_time = time.time()
        
        # Select random drones for mission
        active_drones = self.get_active_drones()
        if not active_drones:
            return {"success": False, "error": "No active drones"}
        
        mission_size = min(random.randint(1, 10), len(active_drones))
        selected_drones = random.sample(active_drones, mission_size)
        
        # Simulate mission execution time
        execution_time = random.uniform(0.05, 0.2)  # 50-200ms
        await asyncio.sleep(execution_time) if asyncio.iscoroutinefunction(asyncio.sleep) else time.sleep(execution_time)
        
        # Update drone states
        for drone_id in selected_drones:
            if drone_id in self.drones:
                self.drones[drone_id].mission_count += 1
                self.drones[drone_id].update()
        
        self.total_missions += 1
        
        return {
            "success": True,
            "mission_id": f"mission_{self.total_missions}",
            "drones_involved": len(selected_drones),
            "execution_time_ms": execution_time * 1000,
            "latency_ms": (time.time() - start_time) * 1000,
        }


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self, coordinator: Optional[SwarmCoordinator] = None):
        """Initialize performance benchmark system.
        
        Args:
            coordinator: SwarmCoordinator instance to test (optional)
        """
        self.coordinator = coordinator
        self.mock_fleet = MockFleet()
        
        # Benchmark state
        self.current_benchmark: Optional[BenchmarkResult] = None
        self.benchmark_history: List[BenchmarkResult] = []
        self.running = False
        
        # Performance monitoring
        self.performance_snapshots: deque = deque(maxlen=10000)
        self.real_time_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.benchmark_task: Optional[asyncio.Task] = None
        
        # Threading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        
        # Logging
        self.logger = get_logger("performance_benchmark")
        
    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run comprehensive performance benchmark."""
        benchmark_id = f"benchmark_{int(time.time())}_{config.benchmark_type.value}"
        self.logger.info(f"Starting benchmark: {benchmark_id}")
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            config=config,
            start_time=time.time(),
            end_time=0,
            snapshots=[],
            summary_stats={},
            bottlenecks_detected=[],
            recommendations=[],
        )
        
        self.current_benchmark = result
        
        try:
            # Start monitoring
            if config.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._continuous_monitoring())
            
            # Execute benchmark based on type
            if config.benchmark_type == BenchmarkType.SCALABILITY:
                await self._run_scalability_benchmark(config)
            elif config.benchmark_type == BenchmarkType.STRESS:
                await self._run_stress_benchmark(config)
            elif config.benchmark_type == BenchmarkType.LATENCY:
                await self._run_latency_benchmark(config)
            elif config.benchmark_type == BenchmarkType.THROUGHPUT:
                await self._run_throughput_benchmark(config)
            elif config.benchmark_type == BenchmarkType.ENDURANCE:
                await self._run_endurance_benchmark(config)
            else:
                await self._run_general_benchmark(config)
            
            result.end_time = time.time()
            result.snapshots = list(self.performance_snapshots)
            result.summary_stats = self._calculate_summary_stats(result.snapshots)
            result.bottlenecks_detected = self._detect_bottlenecks(result.snapshots)
            result.recommendations = self._generate_recommendations(result)
            
            self.logger.info(f"Benchmark completed: {benchmark_id}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = time.time()
            self.logger.error(f"Benchmark failed: {e}")
        
        finally:
            # Stop monitoring
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.current_benchmark = None
        
        self.benchmark_history.append(result)
        return result
    
    async def _run_scalability_benchmark(self, config: BenchmarkConfig):
        """Run scalability benchmark with increasing drone counts."""
        self.logger.info("Running scalability benchmark")
        
        # Test different fleet sizes
        test_sizes = [10, 50, 100, 250, 500, 750, 1000]
        test_sizes = [size for size in test_sizes if size <= config.max_drone_count]
        
        for drone_count in test_sizes:
            self.logger.info(f"Testing with {drone_count} drones")
            
            # Scale mock fleet
            self.mock_fleet.scale_to_size(drone_count)
            
            # Run test scenarios
            await self._run_test_scenarios(drone_count, config.concurrent_missions, 30)
            
            # Wait between tests
            await asyncio.sleep(5)
    
    async def _run_stress_benchmark(self, config: BenchmarkConfig):
        """Run stress test with maximum load."""
        self.logger.info("Running stress benchmark")
        
        # Scale to maximum
        self.mock_fleet.scale_to_size(config.max_drone_count)
        
        # Generate high load
        stress_duration = config.duration_seconds
        concurrent_tasks = []
        
        # Create multiple concurrent mission streams
        for i in range(config.concurrent_missions * 2):  # Double the normal load
            task = asyncio.create_task(
                self._generate_continuous_load(stress_duration)
            )
            concurrent_tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    async def _run_latency_benchmark(self, config: BenchmarkConfig):
        """Run latency-focused benchmark."""
        self.logger.info("Running latency benchmark")
        
        # Test with moderate fleet size for latency measurement
        self.mock_fleet.scale_to_size(min(100, config.max_drone_count))
        
        # Single-threaded sequential requests for accurate latency measurement
        for _ in range(100):  # 100 sequential requests
            start_time = time.time()
            result = self.mock_fleet.simulate_mission()
            latency = (time.time() - start_time) * 1000
            
            # Record latency
            self.real_time_metrics['latency'].append(latency)
            
            # Small delay between requests
            await asyncio.sleep(0.01)
    
    async def _run_throughput_benchmark(self, config: BenchmarkConfig):
        """Run throughput-focused benchmark."""
        self.logger.info("Running throughput benchmark")
        
        # Scale to full capacity
        self.mock_fleet.scale_to_size(config.max_drone_count)
        
        # Maximum concurrent load
        concurrent_tasks = []
        tasks_per_second = config.target_throughput_rps / 10  # Distribute across task workers
        
        for i in range(10):  # 10 worker tasks
            task = asyncio.create_task(
                self._throughput_worker(tasks_per_second, config.duration_seconds)
            )
            concurrent_tasks.append(task)
        
        await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    async def _run_endurance_benchmark(self, config: BenchmarkConfig):
        """Run long-duration endurance test."""
        self.logger.info("Running endurance benchmark")
        
        # Moderate fleet size for stability
        self.mock_fleet.scale_to_size(min(500, config.max_drone_count))
        
        # Continuous moderate load
        endurance_duration = config.duration_seconds
        
        # Create steady background load
        background_tasks = []
        for i in range(5):  # 5 background workers
            task = asyncio.create_task(
                self._generate_continuous_load(endurance_duration, load_factor=0.5)
            )
            background_tasks.append(task)
        
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    async def _run_general_benchmark(self, config: BenchmarkConfig):
        """Run general benchmark with load pattern."""
        self.logger.info("Running general benchmark")
        
        duration = config.duration_seconds
        
        if config.load_pattern == LoadPattern.RAMP_UP:
            await self._run_ramp_up_pattern(config)
        elif config.load_pattern == LoadPattern.SPIKE:
            await self._run_spike_pattern(config)
        elif config.load_pattern == LoadPattern.STEP:
            await self._run_step_pattern(config)
        elif config.load_pattern == LoadPattern.SINE_WAVE:
            await self._run_sine_wave_pattern(config)
        elif config.load_pattern == LoadPattern.RANDOM:
            await self._run_random_pattern(config)
        else:  # CONSTANT
            await self._run_constant_load(config)
    
    async def _run_ramp_up_pattern(self, config: BenchmarkConfig):
        """Run ramp-up load pattern."""
        total_duration = config.duration_seconds
        ramp_duration = config.ramp_up_duration
        steady_duration = config.steady_state_duration
        ramp_down_duration = config.ramp_down_duration
        
        # Ramp up phase
        await self._ramp_load(0.1, 1.0, ramp_duration, config.max_drone_count)
        
        # Steady state phase
        await self._constant_load(1.0, steady_duration, config.max_drone_count, config.concurrent_missions)
        
        # Ramp down phase
        await self._ramp_load(1.0, 0.1, ramp_down_duration, config.max_drone_count)
    
    async def _ramp_load(self, start_factor: float, end_factor: float, duration: int, max_drones: int):
        """Gradually ramp load from start to end factor."""
        steps = duration // 5  # Change every 5 seconds
        step_duration = duration / steps
        
        for step in range(steps):
            # Calculate current load factor
            progress = step / (steps - 1) if steps > 1 else 1
            current_factor = start_factor + (end_factor - start_factor) * progress
            
            # Set fleet size
            drone_count = int(max_drones * current_factor)
            self.mock_fleet.scale_to_size(max(10, drone_count))
            
            # Generate load for this step
            await self._generate_continuous_load(step_duration, current_factor)
    
    async def _constant_load(self, load_factor: float, duration: int, max_drones: int, concurrent_missions: int):
        """Generate constant load."""
        drone_count = int(max_drones * load_factor)
        self.mock_fleet.scale_to_size(max(10, drone_count))
        
        # Run concurrent mission tasks
        concurrent_tasks = []
        for i in range(concurrent_missions):
            task = asyncio.create_task(
                self._generate_continuous_load(duration, load_factor)
            )
            concurrent_tasks.append(task)
        
        await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    
    async def _run_spike_pattern(self, config: BenchmarkConfig):
        """Run spike load pattern."""
        normal_load = 0.3
        spike_load = 1.0
        spike_duration = 30  # 30 second spikes
        
        remaining_duration = config.duration_seconds
        
        while remaining_duration > 0:
            # Normal load period
            normal_period = min(60, remaining_duration // 2)
            if normal_period > 0:
                await self._constant_load(normal_load, normal_period, config.max_drone_count, config.concurrent_missions)
                remaining_duration -= normal_period
            
            # Spike period
            if remaining_duration > 0:
                spike_period = min(spike_duration, remaining_duration)
                await self._constant_load(spike_load, spike_period, config.max_drone_count, config.concurrent_missions * 2)
                remaining_duration -= spike_period
    
    async def _run_test_scenarios(self, drone_count: int, concurrent_missions: int, duration: int):
        """Run test scenarios with specific parameters."""
        # Set fleet size
        self.mock_fleet.scale_to_size(drone_count)
        
        # Generate test load
        start_time = time.time()
        mission_count = 0
        
        while time.time() - start_time < duration:
            # Execute missions
            tasks = []
            for _ in range(concurrent_missions):
                task = asyncio.create_task(self._execute_test_mission())
                tasks.append(task)
            
            # Wait for completion
            await asyncio.gather(*tasks, return_exceptions=True)
            mission_count += len(tasks)
            
            # Small delay
            await asyncio.sleep(0.1)
        
        self.logger.debug(f"Completed {mission_count} missions with {drone_count} drones")
    
    async def _execute_test_mission(self) -> Dict[str, Any]:
        """Execute a single test mission."""
        start_time = time.time()
        
        # Simulate mission planning and execution
        result = self.mock_fleet.simulate_mission()
        
        # Record metrics
        latency = (time.time() - start_time) * 1000
        self.real_time_metrics['latency'].append(latency)
        self.real_time_metrics['throughput'].append(1.0)
        
        if result["success"]:
            self.real_time_metrics['success_rate'].append(1.0)
        else:
            self.real_time_metrics['success_rate'].append(0.0)
            self.real_time_metrics['error_rate'].append(1.0)
        
        return result
    
    async def _generate_continuous_load(self, duration: float, load_factor: float = 1.0):
        """Generate continuous load for specified duration."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Execute mission
            await self._execute_test_mission()
            
            # Adjust delay based on load factor
            base_delay = 0.1  # 100ms base delay
            delay = base_delay / max(0.1, load_factor)
            await asyncio.sleep(delay)
    
    async def _throughput_worker(self, target_rps: float, duration: int):
        """Worker task for throughput testing."""
        interval = 1.0 / target_rps
        end_time = time.time() + duration
        
        while time.time() < end_time:
            await self._execute_test_mission()
            await asyncio.sleep(interval)
    
    async def _continuous_monitoring(self):
        """Continuous performance monitoring during benchmarks."""
        while self.current_benchmark:
            try:
                # Collect performance snapshot
                snapshot = await self._collect_performance_snapshot()
                self.performance_snapshots.append(snapshot)
                
                # Record for AI optimization
                record_performance_metrics(
                    latency_ms=snapshot.latency_ms,
                    throughput_rps=snapshot.throughput_rps,
                    cpu_usage_percent=snapshot.cpu_usage_percent,
                    memory_usage_mb=snapshot.memory_usage_mb,
                    error_rate=snapshot.error_rate,
                )
                
                await asyncio.sleep(1.0)  # Sample every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot."""
        current_time = time.time()
        fleet_status = self.mock_fleet.get_fleet_status()
        
        # System metrics
        cpu_usage = 0.0
        memory_usage = 0.0
        
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.used / (1024 * 1024)  # MB
        
        # Calculate recent metrics
        recent_latencies = list(self.real_time_metrics['latency'])[-10:]
        avg_latency = statistics.mean(recent_latencies) if recent_latencies else 0.0
        
        recent_throughput = list(self.real_time_metrics['throughput'])[-10:]
        throughput_rps = len(recent_throughput) if recent_throughput else 0.0
        
        recent_success = list(self.real_time_metrics['success_rate'])[-10:]
        error_rate = 1.0 - (statistics.mean(recent_success) if recent_success else 1.0)
        
        return PerformanceSnapshot(
            timestamp=current_time,
            drone_count=fleet_status["total_drones"],
            active_missions=fleet_status["active_missions"],
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            network_throughput_mbps=0.0,  # Mock
            latency_ms=avg_latency,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            cache_hit_rate=0.85,  # Mock
            queue_size=0,  # Mock
            connection_count=fleet_status["active_drones"],
        )
    
    def _calculate_summary_stats(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Calculate summary statistics from snapshots."""
        if not snapshots:
            return {}
        
        latencies = [s.latency_ms for s in snapshots]
        throughputs = [s.throughput_rps for s in snapshots]
        error_rates = [s.error_rate for s in snapshots]
        cpu_usages = [s.cpu_usage_percent for s in snapshots]
        memory_usages = [s.memory_usage_mb for s in snapshots]
        
        return {
            "duration_seconds": snapshots[-1].timestamp - snapshots[0].timestamp,
            "total_snapshots": len(snapshots),
            "latency": {
                "min_ms": min(latencies) if latencies else 0,
                "max_ms": max(latencies) if latencies else 0,
                "avg_ms": statistics.mean(latencies) if latencies else 0,
                "p50_ms": statistics.median(latencies) if latencies else 0,
                "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
                "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
            },
            "throughput": {
                "min_rps": min(throughputs) if throughputs else 0,
                "max_rps": max(throughputs) if throughputs else 0,
                "avg_rps": statistics.mean(throughputs) if throughputs else 0,
            },
            "error_rate": {
                "min": min(error_rates) if error_rates else 0,
                "max": max(error_rates) if error_rates else 0,
                "avg": statistics.mean(error_rates) if error_rates else 0,
            },
            "system_resources": {
                "max_cpu_percent": max(cpu_usages) if cpu_usages else 0,
                "avg_cpu_percent": statistics.mean(cpu_usages) if cpu_usages else 0,
                "max_memory_mb": max(memory_usages) if memory_usages else 0,
                "avg_memory_mb": statistics.mean(memory_usages) if memory_usages else 0,
            },
            "fleet_stats": {
                "max_drones": max(s.drone_count for s in snapshots),
                "avg_drones": statistics.mean([s.drone_count for s in snapshots]),
                "max_active_missions": max(s.active_missions for s in snapshots),
            }
        }
    
    def _detect_bottlenecks(self, snapshots: List[PerformanceSnapshot]) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks from snapshots."""
        bottlenecks = []
        
        if not snapshots:
            return bottlenecks
        
        # CPU bottleneck
        high_cpu_snapshots = [s for s in snapshots if s.cpu_usage_percent > 80]
        if len(high_cpu_snapshots) > len(snapshots) * 0.1:  # >10% of time
            bottlenecks.append({
                "type": "cpu",
                "severity": "high",
                "description": f"High CPU usage detected in {len(high_cpu_snapshots)} snapshots",
                "max_value": max(s.cpu_usage_percent for s in high_cpu_snapshots),
                "recommendation": "Consider scaling CPU resources or optimizing algorithms"
            })
        
        # Memory bottleneck
        high_memory_snapshots = [s for s in snapshots if s.memory_usage_mb > 8192]  # >8GB
        if len(high_memory_snapshots) > len(snapshots) * 0.1:
            bottlenecks.append({
                "type": "memory",
                "severity": "high",
                "description": f"High memory usage detected in {len(high_memory_snapshots)} snapshots",
                "max_value": max(s.memory_usage_mb for s in high_memory_snapshots),
                "recommendation": "Consider increasing memory limits or optimizing data structures"
            })
        
        # Latency bottleneck
        high_latency_snapshots = [s for s in snapshots if s.latency_ms > 100]
        if len(high_latency_snapshots) > len(snapshots) * 0.2:  # >20% of time
            bottlenecks.append({
                "type": "latency",
                "severity": "medium",
                "description": f"High latency detected in {len(high_latency_snapshots)} snapshots",
                "max_value": max(s.latency_ms for s in high_latency_snapshots),
                "recommendation": "Optimize processing algorithms or increase parallelism"
            })
        
        # Error rate bottleneck
        high_error_snapshots = [s for s in snapshots if s.error_rate > 0.05]  # >5% error rate
        if len(high_error_snapshots) > 0:
            bottlenecks.append({
                "type": "errors",
                "severity": "critical",
                "description": f"High error rate detected in {len(high_error_snapshots)} snapshots",
                "max_value": max(s.error_rate for s in high_error_snapshots),
                "recommendation": "Investigate error causes and improve error handling"
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, result: BenchmarkResult) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        if not result.summary_stats:
            return recommendations
        
        # Latency recommendations
        avg_latency = result.summary_stats.get("latency", {}).get("avg_ms", 0)
        if avg_latency > 100:
            recommendations.append(
                f"Average latency ({avg_latency:.1f}ms) exceeds target (100ms). "
                "Consider implementing caching, optimizing algorithms, or increasing parallelism."
            )
        
        # Throughput recommendations
        avg_throughput = result.summary_stats.get("throughput", {}).get("avg_rps", 0)
        if avg_throughput < result.config.target_throughput_rps:
            recommendations.append(
                f"Throughput ({avg_throughput:.1f} RPS) below target ({result.config.target_throughput_rps} RPS). "
                "Consider horizontal scaling or performance optimization."
            )
        
        # Resource recommendations
        max_cpu = result.summary_stats.get("system_resources", {}).get("max_cpu_percent", 0)
        if max_cpu > 90:
            recommendations.append(
                "CPU usage reached critical levels (>90%). Consider CPU scaling or workload optimization."
            )
        
        max_memory = result.summary_stats.get("system_resources", {}).get("max_memory_mb", 0)
        if max_memory > 12288:  # >12GB
            recommendations.append(
                "Memory usage is high. Consider memory optimization or increasing memory limits."
            )
        
        # Error rate recommendations
        avg_error_rate = result.summary_stats.get("error_rate", {}).get("avg", 0)
        if avg_error_rate > 0.01:  # >1% error rate
            recommendations.append(
                f"Error rate ({avg_error_rate:.2%}) is elevated. Investigate and resolve error causes."
            )
        
        # Bottleneck-specific recommendations
        for bottleneck in result.bottlenecks_detected:
            if bottleneck not in recommendations:
                recommendations.append(bottleneck.get("recommendation", ""))
        
        return [r for r in recommendations if r]  # Remove empty strings
    
    def save_benchmark_results(self, result: BenchmarkResult, filepath: str):
        """Save benchmark results to file."""
        try:
            # Convert result to serializable format
            result_data = {
                "benchmark_id": result.benchmark_id,
                "config": {
                    "benchmark_type": result.config.benchmark_type.value,
                    "duration_seconds": result.config.duration_seconds,
                    "max_drone_count": result.config.max_drone_count,
                    "load_pattern": result.config.load_pattern.value,
                    "target_latency_ms": result.config.target_latency_ms,
                    "target_throughput_rps": result.config.target_throughput_rps,
                    "concurrent_missions": result.config.concurrent_missions,
                },
                "start_time": result.start_time,
                "end_time": result.end_time,
                "summary_stats": result.summary_stats,
                "bottlenecks_detected": result.bottlenecks_detected,
                "recommendations": result.recommendations,
                "success": result.success,
                "error_message": result.error_message,
                "snapshot_count": len(result.snapshots),
            }
            
            # Save detailed snapshots if requested
            if result.config.save_detailed_results:
                result_data["snapshots"] = [
                    {
                        "timestamp": s.timestamp,
                        "drone_count": s.drone_count,
                        "latency_ms": s.latency_ms,
                        "throughput_rps": s.throughput_rps,
                        "cpu_usage_percent": s.cpu_usage_percent,
                        "memory_usage_mb": s.memory_usage_mb,
                        "error_rate": s.error_rate,
                    }
                    for s in result.snapshots
                ]
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            self.logger.info(f"Saved benchmark results to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {e}")
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> str:
        """Generate comprehensive performance report."""
        if not results:
            return "No benchmark results available."
        
        report = []
        report.append("# Fleet-Mind Generation 3 Performance Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive summary
        total_benchmarks = len(results)
        successful_benchmarks = sum(1 for r in results if r.success)
        
        report.append("## Executive Summary")
        report.append(f"- Total Benchmarks: {total_benchmarks}")
        report.append(f"- Successful Benchmarks: {successful_benchmarks}")
        report.append(f"- Success Rate: {successful_benchmarks / total_benchmarks:.1%}")
        report.append("")
        
        # Performance highlights
        if successful_benchmarks > 0:
            successful_results = [r for r in results if r.success]
            
            best_latency = min(
                r.summary_stats.get("latency", {}).get("avg_ms", float('inf'))
                for r in successful_results
            )
            
            best_throughput = max(
                r.summary_stats.get("throughput", {}).get("avg_rps", 0)
                for r in successful_results
            )
            
            max_drones_tested = max(
                r.summary_stats.get("fleet_stats", {}).get("max_drones", 0)
                for r in successful_results
            )
            
            report.append("## Performance Highlights")
            report.append(f"- Best Average Latency: {best_latency:.1f}ms")
            report.append(f"- Best Throughput: {best_throughput:.1f} RPS")
            report.append(f"- Maximum Drones Tested: {max_drones_tested}")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for i, result in enumerate(results):
            report.append(f"### Benchmark {i+1}: {result.config.benchmark_type.value.title()}")
            report.append(f"- Duration: {result.end_time - result.start_time:.1f}s")
            report.append(f"- Success: {result.success}")
            
            if result.success and result.summary_stats:
                stats = result.summary_stats
                
                if "latency" in stats:
                    lat = stats["latency"]
                    report.append(f"- Latency: avg={lat.get('avg_ms', 0):.1f}ms, "
                                f"p95={lat.get('p95_ms', 0):.1f}ms, "
                                f"p99={lat.get('p99_ms', 0):.1f}ms")
                
                if "throughput" in stats:
                    thr = stats["throughput"]
                    report.append(f"- Throughput: avg={thr.get('avg_rps', 0):.1f} RPS")
                
                if "error_rate" in stats:
                    err = stats["error_rate"]
                    report.append(f"- Error Rate: {err.get('avg', 0):.2%}")
            
            if result.bottlenecks_detected:
                report.append(f"- Bottlenecks: {len(result.bottlenecks_detected)} detected")
            
            if result.recommendations:
                report.append(f"- Recommendations: {len(result.recommendations)} provided")
            
            report.append("")
        
        # Overall recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        unique_recommendations = list(set(all_recommendations))
        
        if unique_recommendations:
            report.append("## Overall Recommendations")
            for i, rec in enumerate(unique_recommendations[:10], 1):  # Top 10
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)


# Global benchmark instance
_performance_benchmark: Optional[PerformanceBenchmark] = None

def get_performance_benchmark() -> PerformanceBenchmark:
    """Get or create global performance benchmark instance."""
    global _performance_benchmark
    if _performance_benchmark is None:
        _performance_benchmark = PerformanceBenchmark()
    return _performance_benchmark

async def run_generation3_benchmark(
    benchmark_type: BenchmarkType = BenchmarkType.SCALABILITY,
    max_drone_count: int = 1000,
    duration_seconds: int = 300,
) -> BenchmarkResult:
    """Run Generation 3 performance benchmark."""
    benchmark = get_performance_benchmark()
    
    config = BenchmarkConfig(
        benchmark_type=benchmark_type,
        duration_seconds=duration_seconds,
        max_drone_count=max_drone_count,
        target_latency_ms=100.0,
        target_throughput_rps=1000.0,
        concurrent_missions=20,
    )
    
    return await benchmark.run_benchmark(config)