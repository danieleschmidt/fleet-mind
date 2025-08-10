"""Scalability and performance benchmarks for Fleet-Mind."""

import asyncio
import time
import statistics
import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import concurrent.futures

from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator
from fleet_mind.fleet.drone_fleet import DroneFleet
from fleet_mind.communication.webrtc_streamer import WebRTCStreamer
from fleet_mind.optimization.distributed_computing import DistributedCoordinator, ComputeNode, NodeStatus
from fleet_mind.optimization.performance_optimizer import PerformanceOptimizer, AdaptiveCache
from fleet_mind.monitoring.health_monitor import HealthMonitor
from fleet_mind.utils.error_handling import ErrorHandler


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        
    def record_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Record performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({"value": value, "unit": unit, "timestamp": time.time()})
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of metric."""
        if metric_name not in self.metrics:
            return {}
        
        values = [m["value"] for m in self.metrics[metric_name]]
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": sorted(values)[int(len(values) * 0.95)] if values else 0.0,
            "p99": sorted(values)[int(len(values) * 0.99)] if values else 0.0
        }
    
    def print_results(self) -> None:
        """Print benchmark results."""
        print(f"\n=== {self.name} Benchmark Results ===")
        for metric_name in sorted(self.metrics.keys()):
            summary = self.get_metric_summary(metric_name)
            unit = self.metrics[metric_name][0]["unit"]
            print(f"{metric_name}: mean={summary['mean']:.3f}{unit}, "
                  f"p95={summary['p95']:.3f}{unit}, max={summary['max']:.3f}{unit}")


class SwarmCoordinationBenchmark(PerformanceBenchmark):
    """Benchmark swarm coordination performance."""
    
    def __init__(self):
        super().__init__("Swarm Coordination")
    
    @pytest.mark.asyncio
    async def test_coordination_scaling(self):
        """Test coordination performance with increasing fleet sizes."""
        fleet_sizes = [10, 50, 100, 250, 500, 1000]
        
        for fleet_size in fleet_sizes:
            print(f"Testing coordination with {fleet_size} drones...")
            
            # Create coordinator
            coordinator = SwarmCoordinator(llm_model="mock", enable_health_monitoring=False)
            
            # Create fleet
            drone_ids = [f"drone_{i}" for i in range(fleet_size)]
            fleet = DroneFleet(drone_ids)
            
            await coordinator.connect_fleet(fleet)
            
            # Benchmark mission planning
            start_time = time.time()
            plan = await coordinator.generate_plan("Survey the designated area efficiently")
            planning_time = (time.time() - start_time) * 1000
            
            self.record_metric(f"planning_time_{fleet_size}_drones", planning_time, "ms")
            
            # Benchmark status retrieval
            start_time = time.time()
            status = await coordinator.get_swarm_status()
            status_time = (time.time() - start_time) * 1000
            
            self.record_metric(f"status_time_{fleet_size}_drones", status_time, "ms")
            
            # Benchmark formation commands
            start_time = time.time()
            await coordinator.execute_formation_change("circular")
            formation_time = (time.time() - start_time) * 1000
            
            self.record_metric(f"formation_time_{fleet_size}_drones", formation_time, "ms")
            
            # Memory usage estimation
            import sys
            memory_mb = sys.getsizeof(coordinator) / 1024 / 1024
            self.record_metric(f"memory_usage_{fleet_size}_drones", memory_mb, "MB")
            
            # Cleanup
            await coordinator.cleanup()
            del coordinator, fleet
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test performance under concurrent operations."""
        coordinator = SwarmCoordinator(llm_model="mock")
        fleet = DroneFleet([f"drone_{i}" for i in range(100)])
        await coordinator.connect_fleet(fleet)
        
        # Test concurrent mission planning
        concurrent_tasks = 20
        tasks = []
        
        start_time = time.time()
        
        for i in range(concurrent_tasks):
            task = coordinator.generate_plan(f"Mission {i}: patrol sector {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        successful = len([r for r in results if not isinstance(r, Exception)])
        
        self.record_metric("concurrent_planning_time", total_time, "ms")
        self.record_metric("concurrent_planning_success_rate", successful / concurrent_tasks * 100, "%")
        self.record_metric("concurrent_planning_throughput", concurrent_tasks / (total_time / 1000), "ops/sec")
        
        await coordinator.cleanup()


class CommunicationBenchmark(PerformanceBenchmark):
    """Benchmark communication performance."""
    
    def __init__(self):
        super().__init__("Communication")
    
    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test message throughput with various payload sizes."""
        streamer = WebRTCStreamer("test_coordinator")
        await streamer.initialize()
        
        payload_sizes = [100, 1000, 10000, 100000]  # bytes
        message_counts = [100, 500, 1000, 2000]
        
        for payload_size in payload_sizes:
            payload = "x" * payload_size
            
            for message_count in message_counts:
                print(f"Testing {message_count} messages of {payload_size} bytes...")
                
                start_time = time.time()
                
                tasks = []
                for i in range(message_count):
                    task = streamer.send_message_to_all({
                        "type": "benchmark",
                        "sequence": i,
                        "payload": payload
                    })
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                total_time = time.time() - start_time
                throughput = message_count / total_time
                bandwidth = (payload_size * message_count) / total_time / 1024 / 1024  # MB/s
                
                self.record_metric(f"throughput_{payload_size}b_{message_count}msg", throughput, "msg/sec")
                self.record_metric(f"bandwidth_{payload_size}b_{message_count}msg", bandwidth, "MB/s")
                self.record_metric(f"latency_{payload_size}b_{message_count}msg", (total_time / message_count) * 1000, "ms")
        
        await streamer.cleanup()
    
    @pytest.mark.asyncio
    async def test_connection_scaling(self):
        """Test performance with increasing number of connections."""
        coordinator = WebRTCStreamer("coordinator")
        await coordinator.initialize()
        
        connection_counts = [10, 25, 50, 100, 200]
        
        for conn_count in connection_counts:
            print(f"Testing with {conn_count} connections...")
            
            # Simulate connections
            for i in range(conn_count):
                await coordinator.handle_new_connection(f"drone_{i}")
            
            # Test broadcasting performance
            message = {"type": "formation", "command": "circular_formation"}
            
            start_time = time.time()
            await coordinator.send_message_to_all(message)
            broadcast_time = (time.time() - start_time) * 1000
            
            self.record_metric(f"broadcast_time_{conn_count}_connections", broadcast_time, "ms")
            self.record_metric(f"per_connection_time_{conn_count}", broadcast_time / conn_count, "ms/conn")
            
            # Memory usage
            import sys
            memory_mb = sys.getsizeof(coordinator) / 1024 / 1024
            self.record_metric(f"memory_usage_{conn_count}_connections", memory_mb, "MB")
        
        await coordinator.cleanup()


class DistributedComputingBenchmark(PerformanceBenchmark):
    """Benchmark distributed computing performance."""
    
    def __init__(self):
        super().__init__("Distributed Computing")
    
    @pytest.mark.asyncio
    async def test_task_distribution_performance(self):
        """Test task distribution and execution performance."""
        coordinator = DistributedCoordinator()
        
        # Add multiple compute nodes
        node_configs = [
            ("node1", "localhost", 8001, 4, 8.0, {"general", "llm_processing"}),
            ("node2", "localhost", 8002, 8, 16.0, {"general", "image_processing"}),
            ("node3", "localhost", 8003, 6, 12.0, {"general", "path_planning"}),
            ("node4", "localhost", 8004, 4, 8.0, {"general"}),
        ]
        
        for node_id, host, port, cpu, memory, capabilities in node_configs:
            coordinator.add_compute_node(host, port, cpu, memory, capabilities)
        
        await coordinator.start()
        
        # Test different task loads
        task_counts = [10, 50, 100, 250, 500]
        
        for task_count in task_counts:
            print(f"Testing distribution of {task_count} tasks...")
            
            start_time = time.time()
            
            # Submit tasks
            task_futures = []
            for i in range(task_count):
                future = coordinator.execute_distributed_planning({
                    "mission": f"Task {i}",
                    "complexity": i % 5
                })
                task_futures.append(future)
            
            # Wait for completion
            results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            total_time = time.time() - start_time
            successful = len([r for r in results if not isinstance(r, Exception)])
            
            self.record_metric(f"distribution_time_{task_count}_tasks", total_time * 1000, "ms")
            self.record_metric(f"distribution_throughput_{task_count}", task_count / total_time, "tasks/sec")
            self.record_metric(f"distribution_success_rate_{task_count}", successful / task_count * 100, "%")
        
        await coordinator.stop()
    
    def test_load_balancing_efficiency(self):
        """Test load balancing algorithm efficiency."""
        coordinator = DistributedCoordinator()
        
        # Create nodes with different capabilities
        nodes = [
            ComputeNode("fast_node", "host1", 8001, 8, 16.0, current_load=0.1, capabilities={"general", "gpu"}),
            ComputeNode("medium_node", "host2", 8002, 4, 8.0, current_load=0.5, capabilities={"general"}),
            ComputeNode("slow_node", "host3", 8003, 2, 4.0, current_load=0.8, capabilities={"general"}),
            ComputeNode("busy_node", "host4", 8004, 8, 16.0, current_load=1.2, capabilities={"general", "llm"})
        ]
        
        for node in nodes:
            coordinator.load_balancer.register_node(node)
        
        # Test node selection performance
        from fleet_mind.optimization.distributed_computing import DistributedTask, TaskPriority
        
        # Test 1000 selections
        start_time = time.time()
        
        for i in range(1000):
            task = DistributedTask(
                task_id=f"task_{i}",
                task_type="general",
                payload={"data": i},
                priority=TaskPriority.NORMAL
            )
            
            selected_node = coordinator.load_balancer.select_node(task)
        
        selection_time = (time.time() - start_time) * 1000
        
        self.record_metric("node_selection_time", selection_time / 1000, "ms/selection")
        self.record_metric("selection_throughput", 1000 / (selection_time / 1000), "selections/sec")


class CachingBenchmark(PerformanceBenchmark):
    """Benchmark caching system performance."""
    
    def __init__(self):
        super().__init__("Caching")
    
    def test_cache_performance(self):
        """Test cache performance with various strategies and sizes."""
        from fleet_mind.optimization.performance_optimizer import AdaptiveCache, CacheStrategy
        
        strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.ADAPTIVE]
        cache_sizes = [1000, 5000, 10000]
        
        for strategy in strategies:
            for cache_size in cache_sizes:
                print(f"Testing {strategy.value} cache with {cache_size} entries...")
                
                cache = AdaptiveCache(max_size=cache_size, strategy=strategy)
                
                # Populate cache
                start_time = time.time()
                for i in range(cache_size):
                    cache.put(f"key_{i}", f"value_{i}")
                
                populate_time = (time.time() - start_time) * 1000
                
                # Test read performance
                start_time = time.time()
                hits = 0
                for i in range(cache_size * 2):  # 50% hit rate
                    key = f"key_{i % cache_size}"
                    value = cache.get(key)
                    if value is not None:
                        hits += 1
                
                read_time = (time.time() - start_time) * 1000
                
                # Test eviction performance
                start_time = time.time()
                for i in range(cache_size, cache_size * 2):
                    cache.put(f"key_new_{i}", f"value_new_{i}")
                
                eviction_time = (time.time() - start_time) * 1000
                
                cache_stats = cache.get_stats()
                
                self.record_metric(f"populate_time_{strategy.value}_{cache_size}", populate_time, "ms")
                self.record_metric(f"read_time_{strategy.value}_{cache_size}", read_time, "ms")
                self.record_metric(f"eviction_time_{strategy.value}_{cache_size}", eviction_time, "ms")
                self.record_metric(f"hit_rate_{strategy.value}_{cache_size}", cache_stats["hit_rate"], "%")
    
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency."""
        import sys
        from fleet_mind.optimization.performance_optimizer import AdaptiveCache
        
        cache = AdaptiveCache(max_size=10000)
        
        # Test with different value sizes
        value_sizes = [100, 1000, 10000, 100000]  # bytes
        
        for value_size in value_sizes:
            cache.clear()
            
            value = "x" * value_size
            memory_before = sys.getsizeof(cache.cache)
            
            # Add 1000 entries
            for i in range(1000):
                cache.put(f"key_{i}", value)
            
            memory_after = sys.getsizeof(cache.cache)
            memory_per_entry = (memory_after - memory_before) / 1000
            
            self.record_metric(f"memory_per_entry_{value_size}b", memory_per_entry, "bytes/entry")


class IntegratedSystemBenchmark(PerformanceBenchmark):
    """Benchmark complete integrated system performance."""
    
    def __init__(self):
        super().__init__("Integrated System")
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test complete end-to-end system performance."""
        # Create full system
        coordinator = SwarmCoordinator(
            llm_model="mock",
            enable_health_monitoring=True,
            enable_optimization=True
        )
        
        fleet = DroneFleet([f"drone_{i}" for i in range(100)])
        await coordinator.connect_fleet(fleet)
        await fleet.start_monitoring()
        
        # Run comprehensive test scenarios
        scenarios = [
            ("simple_mission", "Survey the area"),
            ("complex_mission", "Conduct search and rescue operation with obstacle avoidance"),
            ("formation_change", "Change to V-formation for efficient travel"),
            ("emergency_response", "Execute emergency landing protocol"),
        ]
        
        for scenario_name, mission_description in scenarios:
            print(f"Testing scenario: {scenario_name}")
            
            # Full mission lifecycle
            start_time = time.time()
            
            # 1. Mission planning
            plan_start = time.time()
            plan = await coordinator.generate_plan(mission_description)
            planning_time = (time.time() - plan_start) * 1000
            
            # 2. Status check
            status_start = time.time()
            status = await coordinator.get_swarm_status()
            status_time = (time.time() - status_start) * 1000
            
            # 3. Formation changes
            formation_start = time.time()
            await coordinator.execute_formation_change("circular")
            formation_time = (time.time() - formation_start) * 1000
            
            # 4. Health monitoring
            health_start = time.time()
            health_stats = coordinator.get_comprehensive_stats()
            health_time = (time.time() - health_start) * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Record metrics
            self.record_metric(f"{scenario_name}_planning_time", planning_time, "ms")
            self.record_metric(f"{scenario_name}_status_time", status_time, "ms")
            self.record_metric(f"{scenario_name}_formation_time", formation_time, "ms")
            self.record_metric(f"{scenario_name}_health_time", health_time, "ms")
            self.record_metric(f"{scenario_name}_total_time", total_time, "ms")
            
            # System resource usage
            import psutil
            if hasattr(psutil, 'Process'):
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.record_metric(f"{scenario_name}_cpu_usage", cpu_percent, "%")
                self.record_metric(f"{scenario_name}_memory_usage", memory_mb, "MB")
        
        await fleet.stop_monitoring()
        await coordinator.cleanup()
    
    @pytest.mark.asyncio
    async def test_system_stress(self):
        """Test system under stress conditions."""
        coordinator = SwarmCoordinator(llm_model="mock")
        fleet = DroneFleet([f"drone_{i}" for i in range(500)])  # Large fleet
        
        await coordinator.connect_fleet(fleet)
        
        # Stress test: concurrent operations
        concurrent_operations = 50
        operation_types = [
            "Survey area Alpha",
            "Patrol perimeter",
            "Search and rescue",
            "Formation change",
            "Status update"
        ]
        
        start_time = time.time()
        tasks = []
        
        for i in range(concurrent_operations):
            operation = operation_types[i % len(operation_types)]
            if "Formation" in operation:
                task = coordinator.execute_formation_change("circular")
            else:
                task = coordinator.generate_plan(f"{operation} - Operation {i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        successful = len([r for r in results if not isinstance(r, Exception)])
        errors = [r for r in results if isinstance(r, Exception)]
        
        self.record_metric("stress_test_total_time", total_time, "ms")
        self.record_metric("stress_test_success_rate", successful / concurrent_operations * 100, "%")
        self.record_metric("stress_test_throughput", concurrent_operations / (total_time / 1000), "ops/sec")
        self.record_metric("stress_test_error_count", len(errors), "errors")
        
        await coordinator.cleanup()


class BenchmarkSuite:
    """Complete benchmark suite for Fleet-Mind."""
    
    def __init__(self):
        self.benchmarks = [
            SwarmCoordinationBenchmark(),
            CommunicationBenchmark(),
            DistributedComputingBenchmark(),
            CachingBenchmark(),
            IntegratedSystemBenchmark()
        ]
        self.results = {}
    
    @pytest.mark.asyncio
    async def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("Starting Fleet-Mind Performance Benchmark Suite")
        print("=" * 60)
        
        for benchmark in self.benchmarks:
            print(f"\nRunning {benchmark.name} benchmarks...")
            
            try:
                # Run benchmark methods
                for method_name in dir(benchmark):
                    if method_name.startswith('test_'):
                        method = getattr(benchmark, method_name)
                        print(f"  Running {method_name}...")
                        
                        if asyncio.iscoroutinefunction(method):
                            await method()
                        else:
                            method()
                
                # Store results
                self.results[benchmark.name] = benchmark.metrics
                benchmark.print_results()
                
            except Exception as e:
                print(f"Benchmark {benchmark.name} failed: {e}")
    
    def generate_report(self, filename: str = "benchmark_report.txt"):
        """Generate comprehensive benchmark report."""
        with open(filename, 'w') as f:
            f.write("Fleet-Mind Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.ctime()}\n\n")
            
            for benchmark_name, metrics in self.results.items():
                f.write(f"\n{benchmark_name} Results:\n")
                f.write("-" * 30 + "\n")
                
                for metric_name, values in metrics.items():
                    if values:
                        value_list = [v["value"] for v in values]
                        unit = values[0]["unit"]
                        
                        f.write(f"  {metric_name}:\n")
                        f.write(f"    Mean: {statistics.mean(value_list):.3f} {unit}\n")
                        f.write(f"    Min:  {min(value_list):.3f} {unit}\n")
                        f.write(f"    Max:  {max(value_list):.3f} {unit}\n")
                        if len(value_list) > 1:
                            f.write(f"    StDev: {statistics.stdev(value_list):.3f} {unit}\n")
                        f.write("\n")
        
        print(f"Benchmark report saved to {filename}")


# Test runner
@pytest.mark.asyncio
async def test_run_benchmark_suite():
    """Run the complete benchmark suite."""
    suite = BenchmarkSuite()
    await suite.run_all_benchmarks()
    suite.generate_report()


if __name__ == "__main__":
    # Run benchmarks directly
    asyncio.run(test_run_benchmark_suite())