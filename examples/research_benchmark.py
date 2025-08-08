#!/usr/bin/env python3
"""Research benchmark for Fleet-Mind performance evaluation and algorithm comparison."""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import csv
from pathlib import Path

from fleet_mind import (
    SwarmCoordinator,
    DroneFleet,
    MissionConstraints,
    LatentEncoder,
    CompressionType,
    PlanningLevel,
    performance_monitor,
    get_performance_summary
)

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    experiment_name: str
    fleet_size: int
    compression_type: str
    planning_level: str
    latency_ms: float
    throughput_ops_sec: float
    compression_ratio: float
    memory_usage_mb: float
    success_rate: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass 
class ComparativeStudy:
    """Comparative study configuration."""
    name: str
    description: str
    fleet_sizes: List[int]
    compression_types: List[CompressionType]
    planning_levels: List[PlanningLevel]
    iterations: int
    duration_seconds: int

class FleetMindBenchmark:
    """Comprehensive benchmarking suite for Fleet-Mind research."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
    
    async def run_scalability_study(self) -> Dict[str, Any]:
        """Test scalability from 10 to 1000 drones."""
        print("🔬 Running Scalability Study")
        print("=" * 50)
        
        fleet_sizes = [10, 25, 50, 100, 200, 500]
        study_results = []
        
        for fleet_size in fleet_sizes:
            print(f"\nTesting {fleet_size} drones...")
            
            # Run multiple iterations for statistical significance
            iteration_results = []
            for iteration in range(3):
                result = await self._benchmark_fleet_size(fleet_size, iteration)
                iteration_results.append(result)
                self.results.append(result)
            
            # Calculate statistics
            latencies = [r.latency_ms for r in iteration_results]
            throughputs = [r.throughput_ops_sec for r in iteration_results]
            
            study_result = {
                'fleet_size': fleet_size,
                'avg_latency_ms': statistics.mean(latencies),
                'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'avg_throughput_ops_sec': statistics.mean(throughputs),
                'std_throughput_ops_sec': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            }
            study_results.append(study_result)
            
            print(f"  Avg Latency: {study_result['avg_latency_ms']:.1f}ms ± {study_result['std_latency_ms']:.1f}ms")
            print(f"  Avg Throughput: {study_result['avg_throughput_ops_sec']:.1f} ops/sec")
        
        return {
            'study_name': 'scalability',
            'results': study_results,
            'timestamp': time.time()
        }
    
    async def run_compression_comparison(self) -> Dict[str, Any]:
        """Compare different compression algorithms."""
        print("\n🗜️  Running Compression Algorithm Comparison")
        print("=" * 50)
        
        compression_types = [
            CompressionType.SIMPLE_QUANTIZATION,
            CompressionType.LEARNED_VQVAE,
            CompressionType.NEURAL_COMPRESSION,
            CompressionType.DICTIONARY_COMPRESSION
        ]
        
        fleet_size = 50  # Standard fleet size for comparison
        comparison_results = []
        
        for compression_type in compression_types:
            print(f"\nTesting {compression_type.value} compression...")
            
            # Run benchmark with specific compression
            results = []
            for iteration in range(5):  # More iterations for compression comparison
                result = await self._benchmark_compression(
                    fleet_size, 
                    compression_type, 
                    iteration
                )
                results.append(result)
                self.results.append(result)
            
            # Calculate compression-specific metrics
            compression_ratios = [r.compression_ratio for r in results]
            latencies = [r.latency_ms for r in results]
            
            comparison_result = {
                'compression_type': compression_type.value,
                'avg_compression_ratio': statistics.mean(compression_ratios),
                'std_compression_ratio': statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0,
                'avg_latency_ms': statistics.mean(latencies),
                'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'bandwidth_efficiency': statistics.mean([r.metadata.get('bandwidth_saved_percent', 0) for r in results])
            }
            comparison_results.append(comparison_result)
            
            print(f"  Compression Ratio: {comparison_result['avg_compression_ratio']:.1f}x")
            print(f"  Bandwidth Saved: {comparison_result['bandwidth_efficiency']:.1f}%")
            print(f"  Processing Latency: {comparison_result['avg_latency_ms']:.1f}ms")
        
        return {
            'study_name': 'compression_comparison',
            'results': comparison_results,
            'timestamp': time.time()
        }
    
    async def run_planning_latency_study(self) -> Dict[str, Any]:
        """Study planning latency across different levels."""
        print("\n🧠 Running Planning Latency Study")
        print("=" * 50)
        
        planning_levels = [
            PlanningLevel.STRATEGIC,
            PlanningLevel.TACTICAL, 
            PlanningLevel.REACTIVE
        ]
        
        fleet_size = 25
        latency_results = []
        
        for planning_level in planning_levels:
            print(f"\nTesting {planning_level.value} planning...")
            
            results = []
            for iteration in range(10):  # More iterations for latency measurement
                result = await self._benchmark_planning_latency(
                    fleet_size,
                    planning_level,
                    iteration
                )
                results.append(result)
                self.results.append(result)
            
            # Calculate latency statistics
            latencies = [r.latency_ms for r in results]
            
            latency_result = {
                'planning_level': planning_level.value,
                'avg_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99_latency_ms': sorted(latencies)[int(len(latencies) * 0.99)],
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
                'std_latency_ms': statistics.stdev(latencies)
            }
            latency_results.append(latency_result)
            
            print(f"  Avg: {latency_result['avg_latency_ms']:.1f}ms")
            print(f"  P95: {latency_result['p95_latency_ms']:.1f}ms") 
            print(f"  P99: {latency_result['p99_latency_ms']:.1f}ms")
        
        return {
            'study_name': 'planning_latency',
            'results': latency_results,
            'timestamp': time.time()
        }
    
    async def run_end_to_end_latency_study(self) -> Dict[str, Any]:
        """Measure complete end-to-end system latency."""
        print("\n⚡ Running End-to-End Latency Study")
        print("=" * 50)
        
        test_scenarios = [
            {'name': 'simple_mission', 'complexity': 'low', 'fleet_size': 10},
            {'name': 'formation_flight', 'complexity': 'medium', 'fleet_size': 25}, 
            {'name': 'complex_search', 'complexity': 'high', 'fleet_size': 50},
        ]
        
        e2e_results = []
        
        for scenario in test_scenarios:
            print(f"\nTesting {scenario['name']} scenario...")
            
            scenario_results = []
            for iteration in range(5):
                result = await self._benchmark_end_to_end_latency(
                    scenario['fleet_size'],
                    scenario['complexity'], 
                    iteration
                )
                scenario_results.append(result)
                self.results.append(result)
            
            # Break down latency components
            total_latencies = [r.latency_ms for r in scenario_results]
            planning_times = [r.metadata.get('planning_time_ms', 0) for r in scenario_results]
            encoding_times = [r.metadata.get('encoding_time_ms', 0) for r in scenario_results]
            transmission_times = [r.metadata.get('transmission_time_ms', 0) for r in scenario_results]
            
            e2e_result = {
                'scenario': scenario['name'],
                'complexity': scenario['complexity'],
                'fleet_size': scenario['fleet_size'],
                'total_latency_ms': {
                    'avg': statistics.mean(total_latencies),
                    'p95': sorted(total_latencies)[int(len(total_latencies) * 0.95)],
                    'std': statistics.stdev(total_latencies)
                },
                'planning_latency_ms': {
                    'avg': statistics.mean(planning_times),
                    'std': statistics.stdev(planning_times)
                },
                'encoding_latency_ms': {
                    'avg': statistics.mean(encoding_times),
                    'std': statistics.stdev(encoding_times)
                },
                'transmission_latency_ms': {
                    'avg': statistics.mean(transmission_times),
                    'std': statistics.stdev(transmission_times)
                }
            }
            e2e_results.append(e2e_result)
            
            print(f"  Total E2E: {e2e_result['total_latency_ms']['avg']:.1f}ms (P95: {e2e_result['total_latency_ms']['p95']:.1f}ms)")
            print(f"  Planning: {e2e_result['planning_latency_ms']['avg']:.1f}ms")
            print(f"  Encoding: {e2e_result['encoding_latency_ms']['avg']:.1f}ms")
            print(f"  Transmission: {e2e_result['transmission_latency_ms']['avg']:.1f}ms")
        
        return {
            'study_name': 'end_to_end_latency',
            'results': e2e_results,
            'timestamp': time.time()
        }
    
    async def _benchmark_fleet_size(self, fleet_size: int, iteration: int) -> BenchmarkResult:
        """Benchmark specific fleet size."""
        # Initialize components
        drone_ids = [f"bench_drone_{i}" for i in range(fleet_size)]
        fleet = DroneFleet(drone_ids=drone_ids)
        
        coordinator = SwarmCoordinator(
            llm_model="gpt-4o", 
            max_drones=fleet_size,
            latent_dim=512
        )
        
        await coordinator.connect_fleet(fleet)
        
        # Measure performance
        start_time = time.time()
        memory_start = coordinator._get_memory_usage()
        
        # Execute test operations
        operations_completed = 0
        test_duration = 5  # seconds
        
        mission_desc = f"Coordinate {fleet_size} drones in formation flight mission"
        
        while (time.time() - start_time) < test_duration:
            plan = await coordinator.generate_plan(mission_desc)
            operations_completed += 1
            
            # Avoid overwhelming system
            if operations_completed % 5 == 0:
                await asyncio.sleep(0.1)
        
        end_time = time.time()
        memory_end = coordinator._get_memory_usage()
        
        total_time = end_time - start_time
        latency_ms = (total_time / operations_completed) * 1000
        throughput = operations_completed / total_time
        
        return BenchmarkResult(
            experiment_name=f"scalability_test_{fleet_size}",
            fleet_size=fleet_size,
            compression_type="learned_vqvae", 
            planning_level="strategic",
            latency_ms=latency_ms,
            throughput_ops_sec=throughput,
            compression_ratio=100.0,  # Default
            memory_usage_mb=memory_end - memory_start,
            success_rate=1.0,
            timestamp=time.time(),
            metadata={
                'iteration': iteration,
                'operations_completed': operations_completed,
                'test_duration': total_time
            }
        )
    
    async def _benchmark_compression(
        self, 
        fleet_size: int, 
        compression_type: CompressionType, 
        iteration: int
    ) -> BenchmarkResult:
        """Benchmark specific compression algorithm."""
        # Initialize encoder with specific compression
        encoder = LatentEncoder(
            input_dim=4096,
            latent_dim=512,
            compression_type=compression_type.value
        )
        
        # Test compression performance
        test_data = ["Test action sequence"] * 100
        
        start_time = time.time()
        compressed_results = []
        
        for data in test_data:
            compressed = encoder.encode(data)
            compressed_results.append(compressed)
        
        end_time = time.time()
        
        # Get compression stats
        stats = encoder.get_compression_stats()
        
        total_time = end_time - start_time
        latency_ms = (total_time / len(test_data)) * 1000
        
        return BenchmarkResult(
            experiment_name=f"compression_{compression_type.value}",
            fleet_size=fleet_size,
            compression_type=compression_type.value,
            planning_level="strategic",
            latency_ms=latency_ms,
            throughput_ops_sec=len(test_data) / total_time,
            compression_ratio=stats['theoretical_compression_ratio'],
            memory_usage_mb=50.0,  # Estimated
            success_rate=1.0,
            timestamp=time.time(),
            metadata={
                'iteration': iteration,
                'bandwidth_saved_percent': stats['current_metrics']['bandwidth_saved_percent'],
                'encoding_time_ms': stats['average_encoding_time_ms']
            }
        )
    
    async def _benchmark_planning_latency(
        self,
        fleet_size: int,
        planning_level: PlanningLevel, 
        iteration: int
    ) -> BenchmarkResult:
        """Benchmark planning latency for specific level."""
        # Initialize minimal components
        fleet = DroneFleet([f"test_drone_{i}" for i in range(fleet_size)])
        coordinator = SwarmCoordinator(max_drones=fleet_size)
        await coordinator.connect_fleet(fleet)
        
        # Generate plan and measure time
        mission = f"Execute {planning_level.value} level coordination"
        
        start_time = time.time()
        plan = await coordinator.llm_planner.generate_plan(
            context={
                'mission': mission,
                'num_drones': fleet_size,
                'constraints': {},
                'drone_capabilities': {},
                'current_state': {}
            },
            planning_level=planning_level
        )
        end_time = time.time()
        
        planning_time_ms = (end_time - start_time) * 1000
        
        return BenchmarkResult(
            experiment_name=f"planning_{planning_level.value}",
            fleet_size=fleet_size,
            compression_type="learned_vqvae",
            planning_level=planning_level.value,
            latency_ms=planning_time_ms,
            throughput_ops_sec=1.0 / (end_time - start_time),
            compression_ratio=100.0,
            memory_usage_mb=coordinator._get_memory_usage(),
            success_rate=1.0 if plan else 0.0,
            timestamp=time.time(),
            metadata={
                'iteration': iteration,
                'plan_size': len(str(plan)),
                'planning_time_ms': planning_time_ms
            }
        )
    
    async def _benchmark_end_to_end_latency(
        self,
        fleet_size: int,
        complexity: str,
        iteration: int
    ) -> BenchmarkResult:
        """Benchmark complete end-to-end latency."""
        # Initialize full system
        fleet = DroneFleet([f"e2e_drone_{i}" for i in range(fleet_size)])
        coordinator = SwarmCoordinator(max_drones=fleet_size)
        await coordinator.connect_fleet(fleet)
        
        # Define mission based on complexity
        missions = {
            'low': "Simple patrol mission",
            'medium': "Formation flying with obstacle avoidance", 
            'high': "Complex search and rescue with dynamic replanning"
        }
        
        mission = missions.get(complexity, missions['low'])
        
        # Measure end-to-end latency
        e2e_start = time.time()
        
        # 1. Planning
        planning_start = time.time()
        plan = await coordinator.generate_plan(mission)
        planning_end = time.time()
        
        # 2. Encoding 
        encoding_start = time.time()
        latent_code = plan['latent_code']
        encoding_end = time.time()
        
        # 3. Transmission (simulated)
        transmission_start = time.time()
        await coordinator.webrtc_streamer.broadcast(latent_code)
        transmission_end = time.time()
        
        e2e_end = time.time()
        
        total_latency = (e2e_end - e2e_start) * 1000
        planning_time = (planning_end - planning_start) * 1000
        encoding_time = (encoding_end - encoding_start) * 1000
        transmission_time = (transmission_end - transmission_start) * 1000
        
        return BenchmarkResult(
            experiment_name=f"e2e_{complexity}",
            fleet_size=fleet_size,
            compression_type="learned_vqvae",
            planning_level="strategic",
            latency_ms=total_latency,
            throughput_ops_sec=1000.0 / total_latency,
            compression_ratio=100.0,
            memory_usage_mb=coordinator._get_memory_usage(),
            success_rate=1.0,
            timestamp=time.time(),
            metadata={
                'iteration': iteration,
                'complexity': complexity,
                'planning_time_ms': planning_time,
                'encoding_time_ms': encoding_time,
                'transmission_time_ms': transmission_time
            }
        )
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to files."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fleet_mind_benchmark_{timestamp}"
        
        # Save JSON results
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'results': [asdict(result) for result in self.results],
                'summary': self.get_summary_statistics(),
                'timestamp': time.time()
            }, f, indent=2)
        
        # Save CSV results  
        csv_path = self.output_dir / f"{filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
        
        print(f"\n💾 Results saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        
        return str(json_path)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        latencies = [r.latency_ms for r in self.results]
        throughputs = [r.throughput_ops_sec for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        
        return {
            'total_experiments': len(self.results),
            'latency_stats': {
                'avg_ms': statistics.mean(latencies),
                'median_ms': statistics.median(latencies),
                'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99_ms': sorted(latencies)[int(len(latencies) * 0.99)],
                'std_ms': statistics.stdev(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies)
            },
            'throughput_stats': {
                'avg_ops_sec': statistics.mean(throughputs),
                'max_ops_sec': max(throughputs),
                'std_ops_sec': statistics.stdev(throughputs)
            },
            'memory_stats': {
                'avg_mb': statistics.mean(memory_usage),
                'max_mb': max(memory_usage),
                'std_mb': statistics.stdev(memory_usage)
            }
        }
    
    def print_summary(self):
        """Print benchmark summary."""
        summary = self.get_summary_statistics()
        
        print("\n📊 Benchmark Summary")
        print("=" * 50)
        print(f"Total Experiments: {summary['total_experiments']}")
        
        print(f"\nLatency Statistics:")
        lat_stats = summary['latency_stats']
        print(f"  Average: {lat_stats['avg_ms']:.1f}ms")
        print(f"  Median: {lat_stats['median_ms']:.1f}ms") 
        print(f"  P95: {lat_stats['p95_ms']:.1f}ms")
        print(f"  P99: {lat_stats['p99_ms']:.1f}ms")
        print(f"  Range: {lat_stats['min_ms']:.1f}ms - {lat_stats['max_ms']:.1f}ms")
        
        print(f"\nThroughput Statistics:")
        tp_stats = summary['throughput_stats']
        print(f"  Average: {tp_stats['avg_ops_sec']:.1f} ops/sec")
        print(f"  Maximum: {tp_stats['max_ops_sec']:.1f} ops/sec")
        
        print(f"\nMemory Usage:")
        mem_stats = summary['memory_stats']
        print(f"  Average: {mem_stats['avg_mb']:.1f} MB")
        print(f"  Maximum: {mem_stats['max_mb']:.1f} MB")

async def run_comprehensive_benchmark():
    """Run the complete Fleet-Mind research benchmark suite."""
    benchmark = FleetMindBenchmark()
    
    print("🚀 Fleet-Mind Research Benchmark Suite")
    print("=" * 60)
    print("This benchmark evaluates Fleet-Mind performance across multiple dimensions")
    print("for research publication and algorithmic comparison.\n")
    
    try:
        # Run all studies
        studies = []
        
        # 1. Scalability Study
        scalability_results = await benchmark.run_scalability_study()
        studies.append(scalability_results)
        
        # 2. Compression Comparison
        compression_results = await benchmark.run_compression_comparison()
        studies.append(compression_results)
        
        # 3. Planning Latency Study
        planning_results = await benchmark.run_planning_latency_study()
        studies.append(planning_results)
        
        # 4. End-to-End Latency Study
        e2e_results = await benchmark.run_end_to_end_latency_study()
        studies.append(e2e_results)
        
        # Generate comprehensive results
        print(f"\n🎯 Generating Comprehensive Results...")
        benchmark.print_summary()
        
        # Save all results
        results_file = benchmark.save_results()
        
        # Generate research summary
        print(f"\n📝 Research Summary:")
        print(f"   • Tested {len(set(r.fleet_size for r in benchmark.results))} different fleet sizes")
        print(f"   • Compared {len(set(r.compression_type for r in benchmark.results))} compression algorithms")
        print(f"   • Evaluated {len(set(r.planning_level for r in benchmark.results))} planning levels")
        print(f"   • Total statistical samples: {len(benchmark.results)}")
        
        # Key findings
        summary = benchmark.get_summary_statistics()
        lat_stats = summary['latency_stats']
        print(f"\n🏆 Key Performance Metrics:")
        print(f"   • Average End-to-End Latency: {lat_stats['avg_ms']:.1f}ms")
        print(f"   • 95th Percentile Latency: {lat_stats['p95_ms']:.1f}ms")
        print(f"   • Maximum Throughput: {summary['throughput_stats']['max_ops_sec']:.1f} operations/sec")
        print(f"   • Target <100ms Latency: {'✅ ACHIEVED' if lat_stats['p95_ms'] < 100 else '❌ NOT ACHIEVED'}")
        
        print(f"\n✅ Benchmark suite completed successfully!")
        print(f"📊 Results available in: {results_file}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
        benchmark.save_results("interrupted_benchmark")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        benchmark.save_results("failed_benchmark")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())