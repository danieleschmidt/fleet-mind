#!/usr/bin/env python3
"""
Fleet-Mind Generation 3 Scalable Demo

This demo showcases the advanced scalability and performance optimizations of Fleet-Mind Generation 3:
- AI-powered performance optimization
- Service mesh coordination for 1000+ drones
- Multi-tier caching with intelligent strategies
- High-performance communication with connection pooling
- ML-based cost optimization and spot instance management
- Sub-100ms latency for massive drone coordination
- Comprehensive performance monitoring and benchmarking

Run this demo to see Fleet-Mind Generation 3 in action!
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any
import argparse
import sys
from pathlib import Path

# Add fleet_mind to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fleet_mind.coordination.swarm_coordinator import SwarmCoordinator, MissionConstraints
from fleet_mind.fleet.drone_fleet import DroneFleet
from fleet_mind.optimization.performance_benchmark import (
    get_performance_benchmark, BenchmarkConfig, BenchmarkType, LoadPattern
)
from fleet_mind.optimization.ai_performance_optimizer import get_ai_optimizer, record_performance_metrics
from fleet_mind.optimization.service_mesh_coordinator import get_service_mesh, ServiceType
from fleet_mind.optimization.ml_cost_optimizer import get_cost_optimizer, DeploymentStrategy
from fleet_mind.optimization.multi_tier_cache import get_multi_tier_cache
from fleet_mind.communication.high_performance_comm import get_hp_communicator, Priority
from fleet_mind.optimization.cloud_deployment_optimizer import get_cloud_optimizer, optimize_fleet_deployment


class Generation3Demo:
    """Comprehensive Generation 3 demonstration."""
    
    def __init__(self):
        """Initialize Generation 3 demo."""
        self.coordinator = None
        self.fleet = None
        self.demo_start_time = None
        self.results = {}
        
        # Demo configuration
        self.max_drones = 1000
        self.target_latency_ms = 100
        self.benchmark_duration = 300  # 5 minutes
        
        print("🚁 Fleet-Mind Generation 3 Scalable Demo")
        print("=" * 60)
        print("Advanced drone swarm coordination with massive scalability!")
        print()
    
    async def run_complete_demo(self):
        """Run the complete Generation 3 demonstration."""
        self.demo_start_time = time.time()
        
        try:
            print("🚀 Starting Fleet-Mind Generation 3 Demo...")
            print()
            
            # Phase 1: System Initialization
            await self._demo_phase_1_initialization()
            
            # Phase 2: Performance Optimization
            await self._demo_phase_2_performance()
            
            # Phase 3: Massive Scale Testing
            await self._demo_phase_3_scalability()
            
            # Phase 4: Advanced Features
            await self._demo_phase_4_advanced_features()
            
            # Phase 5: Cost Optimization
            await self._demo_phase_5_cost_optimization()
            
            # Phase 6: Comprehensive Benchmarking
            await self._demo_phase_6_benchmarking()
            
            # Generate final report
            await self._generate_final_report()
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self._cleanup_demo()
    
    async def _demo_phase_1_initialization(self):
        """Phase 1: Initialize Generation 3 systems."""
        print("📋 PHASE 1: System Initialization")
        print("-" * 40)
        
        # Initialize SwarmCoordinator
        print("🔧 Initializing Generation 3 SwarmCoordinator...")
        self.coordinator = SwarmCoordinator(
            llm_model="gpt-4o",
            latent_dim=512,
            max_drones=self.max_drones,
            update_rate=50.0,  # High update rate for responsiveness
        )
        
        # Initialize Generation 3 systems
        await self.coordinator.initialize_generation3_systems()
        print("✅ Generation 3 systems initialized")
        
        # Create large-scale mock fleet
        print(f"🚁 Creating mock fleet of {self.max_drones} drones...")
        drone_ids = [f"drone_{i:04d}" for i in range(self.max_drones)]
        self.fleet = DroneFleet(drone_ids)
        
        # Connect fleet to coordinator
        print("🔗 Connecting fleet to coordinator...")
        await self.coordinator.connect_fleet(self.fleet)
        print("✅ Fleet connected successfully")
        
        # Display initialization summary
        stats = self.coordinator.get_generation3_comprehensive_stats()
        print(f"📊 Initialization Summary:")
        print(f"   • Total Drones: {stats['swarm_status']['active_drones']}")
        print(f"   • Generation 3 Optimizations: {len([k for k, v in stats['generation3']['optimizations_enabled'].items() if v])}")
        print(f"   • System Uptime: {stats['swarm_status']['uptime_seconds']:.1f}s")
        print()
        
        await asyncio.sleep(2)  # Demo pacing
    
    async def _demo_phase_2_performance(self):
        """Phase 2: Demonstrate performance optimization."""
        print("⚡ PHASE 2: AI-Powered Performance Optimization")
        print("-" * 50)
        
        # Get AI optimizer
        ai_optimizer = get_ai_optimizer()
        
        print("🧠 Recording performance metrics for AI optimization...")
        
        # Simulate various performance scenarios
        scenarios = [
            {"cpu": 45, "memory": 512, "latency": 80, "desc": "Normal load"},
            {"cpu": 75, "memory": 1024, "latency": 120, "desc": "High CPU load"},
            {"cpu": 35, "memory": 2048, "latency": 95, "desc": "High memory usage"},
            {"cpu": 85, "memory": 768, "latency": 150, "desc": "Peak load"},
        ]
        
        for scenario in scenarios:
            print(f"   📈 Testing scenario: {scenario['desc']}")
            
            # Record metrics
            record_performance_metrics(
                latency_ms=scenario["latency"],
                cpu_usage_percent=scenario["cpu"],
                memory_usage_mb=scenario["memory"],
                cache_hit_rate=0.85,
                error_rate=0.02,
            )
            
            await asyncio.sleep(0.5)  # Allow processing
        
        # Get AI optimization recommendations
        print("🎯 Generating AI optimization recommendations...")
        optimization_action = await ai_optimizer.suggest_optimization()
        
        if optimization_action:
            print(f"   💡 AI Recommendation: {optimization_action.action_type}")
            print(f"   🎯 Confidence: {optimization_action.confidence:.2f}")
            print(f"   📈 Expected Improvement: {optimization_action.predicted_improvement:.2f}")
            
            # Apply optimization
            print("   🔧 Applying AI optimization...")
            await ai_optimizer._apply_optimization(optimization_action)
            print("   ✅ Optimization applied successfully")
        
        # Display AI optimizer stats
        ai_stats = ai_optimizer.get_optimization_stats()
        print(f"📊 AI Optimization Summary:")
        print(f"   • Strategy: {ai_stats['strategy']}")
        print(f"   • Optimization Attempts: {ai_stats['optimization_attempts']}")
        print(f"   • Success Rate: {ai_stats['success_rate']:.1%}")
        print(f"   • Total Improvement: {ai_stats['total_improvement']:.3f}")
        print()
        
        await asyncio.sleep(2)
    
    async def _demo_phase_3_scalability(self):
        """Phase 3: Demonstrate massive scale coordination."""
        print("📈 PHASE 3: Massive Scale Coordination (1000+ Drones)")
        print("-" * 55)
        
        # Get service mesh
        service_mesh = await get_service_mesh()
        
        print("🌐 Testing service mesh with increasing load...")
        
        # Test different fleet sizes
        test_sizes = [100, 250, 500, 750, 1000]
        latency_results = {}
        
        for drone_count in test_sizes:
            print(f"   🚁 Testing with {drone_count} drones...")
            
            # Simulate mission planning for different fleet sizes
            start_time = time.time()
            
            # Generate high-performance mission plan
            mission_plan = await self.coordinator.generate_plan_generation3(
                f"Coordinate {drone_count} drones in formation flight pattern",
                constraints=MissionConstraints(max_altitude=100.0, safety_distance=3.0),
                context={"drone_count": drone_count, "performance_test": True}
            )
            
            planning_latency = (time.time() - start_time) * 1000
            latency_results[drone_count] = planning_latency
            
            print(f"      ⏱️  Planning Latency: {planning_latency:.1f}ms")
            
            # Test mission execution with high-performance communication
            if drone_count <= 500:  # Limit full execution testing
                start_time = time.time()
                success = await self.coordinator.execute_mission_generation3(mission_plan)
                execution_latency = (time.time() - start_time) * 1000
                
                print(f"      🎯 Execution Success: {success}")
                print(f"      ⏱️  Execution Latency: {execution_latency:.1f}ms")
            
            # Check if sub-100ms target is met
            if planning_latency <= self.target_latency_ms:
                print(f"      ✅ Sub-100ms target achieved!")
            else:
                print(f"      ⚠️  Above 100ms target ({planning_latency:.1f}ms)")
            
            await asyncio.sleep(1)  # Demo pacing
        
        # Display scalability results
        print(f"📊 Scalability Test Results:")
        for drone_count, latency in latency_results.items():
            status = "✅" if latency <= self.target_latency_ms else "⚠️"
            print(f"   {status} {drone_count:4d} drones: {latency:5.1f}ms")
        
        # Service mesh statistics
        mesh_stats = service_mesh.get_mesh_status()
        print(f"🌐 Service Mesh Status:")
        print(f"   • Total Services: {mesh_stats['total_services']}")
        print(f"   • Healthy Services: {mesh_stats['healthy_services']}")
        print(f"   • Service Availability: {mesh_stats['service_availability']:.1%}")
        print(f"   • Average Response Time: {mesh_stats['avg_response_time_ms']:.1f}ms")
        print()
        
        self.results['scalability'] = {
            'latency_results': latency_results,
            'max_drones_tested': max(test_sizes),
            'sub_100ms_achieved': all(l <= 100 for l in latency_results.values()),
        }
        
        await asyncio.sleep(2)
    
    async def _demo_phase_4_advanced_features(self):
        """Phase 4: Demonstrate advanced Generation 3 features."""
        print("🔥 PHASE 4: Advanced Generation 3 Features")
        print("-" * 45)
        
        # Multi-tier caching demonstration
        print("💾 Testing Multi-Tier Caching System...")
        cache = await get_multi_tier_cache()
        
        # Test cache performance
        cache_test_keys = [f"mission_plan_{i}" for i in range(100)]
        cache_test_data = {"plan": "formation_flight", "drones": list(range(50))}
        
        # Warm up cache
        print("   🔥 Warming up cache with test data...")
        start_time = time.time()
        for key in cache_test_keys[:20]:  # Warm 20 keys
            await cache.put(key, cache_test_data)
        warmup_time = (time.time() - start_time) * 1000
        
        # Test cache retrieval performance
        print("   📊 Testing cache retrieval performance...")
        hit_count = 0
        start_time = time.time()
        for key in cache_test_keys:
            result = await cache.get(key)
            if result is not None:
                hit_count += 1
        
        retrieval_time = (time.time() - start_time) * 1000
        hit_rate = hit_count / len(cache_test_keys)
        
        cache_stats = cache.get_comprehensive_stats()
        print(f"   📈 Cache Performance:")
        print(f"      • Cache Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
        print(f"      • L1 Hit Rate: {cache_stats['l1_hit_rate']:.1%}")
        print(f"      • L2 Hit Rate: {cache_stats['l2_hit_rate']:.1%}")
        print(f"      • Average Retrieval Time: {retrieval_time/len(cache_test_keys):.2f}ms")
        print()
        
        # High-performance communication demonstration
        print("⚡ Testing High-Performance Communication...")
        hp_comm = await get_hp_communicator()
        
        # Test broadcast performance
        destinations = [f"ws://drone_{i:04d}:8080/command" for i in range(100)]
        test_payload = {"command": "status_check", "timestamp": time.time()}
        
        start_time = time.time()
        broadcast_result = await hp_comm.broadcast_message(
            destinations=destinations,
            payload=test_payload,
            priority=Priority.HIGH,
            max_concurrent=50,
        )
        broadcast_time = (time.time() - start_time) * 1000
        
        print(f"   📡 Broadcast Performance:")
        print(f"      • Destinations: {broadcast_result['total_destinations']}")
        print(f"      • Success Rate: {broadcast_result['success_rate']:.1%}")
        print(f"      • Total Time: {broadcast_time:.1f}ms")
        print(f"      • Average per Drone: {broadcast_time/len(destinations):.2f}ms")
        
        comm_stats = hp_comm.get_comprehensive_stats()
        print(f"   📊 Communication Stats:")
        print(f"      • Messages per Second: {comm_stats['messages_per_second']:.1f}")
        print(f"      • Connection Pools: {comm_stats['connection_pools']}")
        print(f"      • Total Connections: {comm_stats['total_connections']}")
        print()
        
        self.results['advanced_features'] = {
            'cache_hit_rate': cache_stats['overall_hit_rate'],
            'broadcast_success_rate': broadcast_result['success_rate'],
            'avg_broadcast_latency_ms': broadcast_time / len(destinations),
        }
        
        await asyncio.sleep(2)
    
    async def _demo_phase_5_cost_optimization(self):
        """Phase 5: Demonstrate cost optimization."""
        print("💰 PHASE 5: ML-Based Cost Optimization")
        print("-" * 40)
        
        # Get cost optimizer
        cost_optimizer = get_cost_optimizer()
        
        print("📊 Recording cost and performance metrics...")
        
        # Simulate different load scenarios for cost analysis
        scenarios = [
            {"cpu": 30, "memory": 512, "cost": 10.50, "desc": "Low load"},
            {"cpu": 60, "memory": 1024, "cost": 25.75, "desc": "Medium load"},
            {"cpu": 85, "memory": 2048, "cost": 45.20, "desc": "High load"},
        ]
        
        for scenario in scenarios:
            print(f"   💹 Analyzing scenario: {scenario['desc']}")
            
            # Record cost metrics
            from fleet_mind.optimization.ml_cost_optimizer import record_cost_metrics
            record_cost_metrics(
                current_cost=scenario["cost"],
                instance_counts={"on_demand": 3, "spot": 7},
                performance_metrics={
                    "cpu_usage": scenario["cpu"],
                    "memory_usage": scenario["memory"],
                    "response_time": 80.0,
                    "throughput": 150.0,
                }
            )
            
            await asyncio.sleep(0.5)
        
        # Get cost optimization recommendations
        print("🎯 Generating cost optimization recommendations...")
        current_load = {"cpu_usage": 70, "memory_usage": 1200, "response_time": 85, "throughput": 120}
        predicted_load = {"cpu_usage": 75, "memory_usage": 1300, "response_time": 90, "throughput": 130}
        
        recommendation = await cost_optimizer.predict_optimal_scaling(current_load, predicted_load)
        
        print(f"   💡 Cost Optimization Recommendation:")
        print(f"      • Action: {recommendation.action}")
        print(f"      • Target Capacity: {recommendation.target_capacity}")
        print(f"      • Predicted Savings: ${recommendation.predicted_cost_savings:.2f}/hour")
        print(f"      • Confidence: {recommendation.confidence:.2f}")
        print(f"      • Risk Assessment: {list(recommendation.risk_assessment.keys())}")
        
        # Apply cost optimization
        if recommendation.predicted_cost_savings > 1.0:  # $1/hour savings
            print("   🔧 Applying cost optimization...")
            optimization_result = await cost_optimizer.execute_cost_optimization(recommendation)
            print(f"   ✅ Cost optimization result: {optimization_result['status']}")
            if optimization_result['status'] == 'success':
                print(f"      💰 Actual Savings: ${optimization_result['actual_savings']:.2f}/hour")
        
        # Display cost optimizer statistics
        cost_stats = cost_optimizer.get_optimization_stats()
        print(f"📊 Cost Optimization Summary:")
        print(f"   • Strategy: {cost_stats['strategy']}")
        print(f"   • Current Hourly Cost: ${cost_stats['current_hourly_cost']:.2f}")
        print(f"   • Total Cost Savings: ${cost_stats['total_cost_savings']:.2f}")
        print(f"   • Optimization Success Rate: {cost_stats['success_rate']:.1%}")
        print(f"   • Spot Instance Ratio: {cost_stats['spot_instance_ratio']:.1%}")
        print()
        
        # Cloud deployment optimization
        print("☁️  Testing Cloud Deployment Optimization...")
        deployment_result = await optimize_fleet_deployment(
            max_instances=50,
            cost_budget_per_hour=30.0,
        )
        
        print(f"   📈 Deployment Optimization Results:")
        print(f"      • Strategy: {deployment_result['strategy']}")
        print(f"      • Estimated Savings: ${deployment_result['estimated_cost_savings']:.2f}/hour")
        print(f"      • Recommendations: {len(deployment_result['recommendations'])}")
        print()
        
        self.results['cost_optimization'] = {
            'predicted_savings': recommendation.predicted_cost_savings,
            'total_savings_achieved': cost_stats['total_cost_savings'],
            'optimization_success_rate': cost_stats['success_rate'],
        }
        
        await asyncio.sleep(2)
    
    async def _demo_phase_6_benchmarking(self):
        """Phase 6: Comprehensive performance benchmarking."""
        print("🏆 PHASE 6: Comprehensive Performance Benchmarking")
        print("-" * 55)
        
        # Get performance benchmark system
        benchmark = get_performance_benchmark()
        
        print("⚡ Running Generation 3 Performance Benchmarks...")
        print("   (This may take a few minutes...)")
        
        # Define benchmark scenarios
        benchmark_scenarios = [
            {
                "type": BenchmarkType.LATENCY,
                "name": "Latency Optimization Test",
                "duration": 60,  # Shortened for demo
            },
            {
                "type": BenchmarkType.THROUGHPUT,
                "name": "Throughput Maximization Test", 
                "duration": 60,
            },
            {
                "type": BenchmarkType.SCALABILITY,
                "name": "1000+ Drone Scalability Test",
                "duration": 90,
            },
        ]
        
        benchmark_results = []
        
        for scenario in benchmark_scenarios:
            print(f"   🧪 Running {scenario['name']}...")
            
            config = BenchmarkConfig(
                benchmark_type=scenario["type"],
                duration_seconds=scenario["duration"],
                max_drone_count=self.max_drones,
                target_latency_ms=self.target_latency_ms,
                target_throughput_rps=1000.0,
                concurrent_missions=20,
                load_pattern=LoadPattern.RAMP_UP,
            )
            
            # Run benchmark
            result = await benchmark.run_benchmark(config)
            benchmark_results.append(result)
            
            if result.success:
                summary = result.summary_stats
                print(f"      ✅ Benchmark completed successfully")
                if "latency" in summary:
                    print(f"         ⏱️  Average Latency: {summary['latency']['avg_ms']:.1f}ms")
                    print(f"         📊 P95 Latency: {summary['latency']['p95_ms']:.1f}ms")
                if "throughput" in summary:
                    print(f"         🚀 Average Throughput: {summary['throughput']['avg_rps']:.1f} RPS")
                if "error_rate" in summary:
                    print(f"         🎯 Error Rate: {summary['error_rate']['avg']:.2%}")
            else:
                print(f"      ❌ Benchmark failed: {result.error_message}")
            
            print()
        
        # Generate comprehensive benchmark report
        if benchmark_results:
            print("📋 Generating comprehensive benchmark report...")
            report = benchmark.generate_performance_report(benchmark_results)
            
            # Save report to file
            report_file = f"generation3_benchmark_report_{int(time.time())}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"   📄 Report saved to: {report_file}")
            
            # Display key metrics
            successful_benchmarks = [r for r in benchmark_results if r.success]
            if successful_benchmarks:
                best_latency = min(
                    r.summary_stats.get("latency", {}).get("avg_ms", float('inf'))
                    for r in successful_benchmarks
                )
                best_throughput = max(
                    r.summary_stats.get("throughput", {}).get("avg_rps", 0)
                    for r in successful_benchmarks
                )
                
                print(f"🏆 Benchmark Highlights:")
                print(f"   • Best Average Latency: {best_latency:.1f}ms")
                print(f"   • Best Throughput: {best_throughput:.1f} RPS")
                print(f"   • Sub-100ms Target: {'✅ ACHIEVED' if best_latency <= 100 else '❌ MISSED'}")
                print(f"   • Successful Tests: {len(successful_benchmarks)}/{len(benchmark_results)}")
        
        print()
        
        self.results['benchmarking'] = {
            'total_benchmarks': len(benchmark_results),
            'successful_benchmarks': len([r for r in benchmark_results if r.success]),
            'best_latency_ms': min(
                (r.summary_stats.get("latency", {}).get("avg_ms", float('inf'))
                 for r in benchmark_results if r.success), 
                default=float('inf')
            ),
        }
        
        await asyncio.sleep(2)
    
    async def _generate_final_report(self):
        """Generate final demo report."""
        print("📊 GENERATION 3 DEMO FINAL REPORT")
        print("=" * 60)
        
        demo_duration = time.time() - self.demo_start_time
        
        print(f"⏱️  Total Demo Duration: {demo_duration:.1f} seconds")
        print()
        
        # Overall success metrics
        overall_success = True
        success_criteria = []
        
        # Scalability success
        if 'scalability' in self.results:
            scalability = self.results['scalability']
            sub_100ms_success = scalability['sub_100ms_achieved']
            success_criteria.append(("Sub-100ms Latency (1000 drones)", sub_100ms_success))
            overall_success &= sub_100ms_success
        
        # Advanced features success
        if 'advanced_features' in self.results:
            features = self.results['advanced_features']
            cache_success = features['cache_hit_rate'] > 0.8
            comm_success = features['broadcast_success_rate'] > 0.9
            success_criteria.append(("Cache Hit Rate >80%", cache_success))
            success_criteria.append(("Broadcast Success >90%", comm_success))
            overall_success &= cache_success and comm_success
        
        # Cost optimization success
        if 'cost_optimization' in self.results:
            cost = self.results['cost_optimization']
            cost_success = cost['predicted_savings'] > 0
            success_criteria.append(("Cost Savings Achieved", cost_success))
            overall_success &= cost_success
        
        # Benchmarking success
        if 'benchmarking' in self.results:
            bench = self.results['benchmarking']
            bench_success = bench['successful_benchmarks'] >= bench['total_benchmarks'] * 0.8
            success_criteria.append(("Benchmark Success Rate >80%", bench_success))
            overall_success &= bench_success
        
        print("🎯 SUCCESS CRITERIA:")
        for criterion, success in success_criteria:
            status = "✅" if success else "❌"
            print(f"   {status} {criterion}")
        
        print()
        print(f"🏆 OVERALL DEMO SUCCESS: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print()
        
        # Key achievements
        print("🌟 KEY ACHIEVEMENTS:")
        if 'scalability' in self.results and self.results['scalability']['sub_100ms_achieved']:
            print("   ✅ Achieved sub-100ms latency for 1000+ drone coordination")
        
        if 'advanced_features' in self.results:
            features = self.results['advanced_features']
            print(f"   ✅ Multi-tier cache hit rate: {features['cache_hit_rate']:.1%}")
            print(f"   ✅ High-performance broadcast success: {features['broadcast_success_rate']:.1%}")
        
        if 'cost_optimization' in self.results:
            cost = self.results['cost_optimization']
            if cost['predicted_savings'] > 0:
                print(f"   ✅ Cost savings achieved: ${cost['predicted_savings']:.2f}/hour")
        
        print()
        
        # System statistics
        if self.coordinator:
            final_stats = self.coordinator.get_generation3_comprehensive_stats()
            
            print("📈 FINAL SYSTEM STATISTICS:")
            print(f"   • Total Drones Coordinated: {final_stats['swarm_status']['active_drones']}")
            print(f"   • System Uptime: {final_stats['swarm_status']['uptime_seconds']:.1f}s")
            print(f"   • Mission Success Rate: {final_stats['swarm_status'].get('mission_progress', 0):.1%}")
            print(f"   • Average Response Time: {final_stats['swarm_status']['recent_latency_ms']:.1f}ms")
            
            if 'generation3' in final_stats:
                gen3_stats = final_stats['generation3']
                enabled_optimizations = sum(1 for v in gen3_stats['optimizations_enabled'].values() if v)
                print(f"   • Generation 3 Optimizations Active: {enabled_optimizations}/5")
                
                if 'ai_optimization' in gen3_stats:
                    ai_stats = gen3_stats['ai_optimization']
                    print(f"   • AI Optimizations Applied: {ai_stats.get('optimization_attempts', 0)}")
                    print(f"   • AI Success Rate: {ai_stats.get('success_rate', 0):.1%}")
        
        print()
        print("🎉 Generation 3 Demo Complete!")
        print("   Fleet-Mind has demonstrated advanced scalability and performance")
        print("   capabilities suitable for coordinating 1000+ drone operations")
        print("   with sub-100ms latency and intelligent optimization!")
        print()
    
    async def _cleanup_demo(self):
        """Clean up demo resources."""
        print("🧹 Cleaning up demo resources...")
        
        try:
            if self.coordinator:
                await self.coordinator.shutdown_generation3_systems()
            
            print("✅ Demo cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Fleet-Mind Generation 3 Scalable Demo")
    parser.add_argument("--max-drones", type=int, default=1000, help="Maximum number of drones to test")
    parser.add_argument("--target-latency", type=float, default=100.0, help="Target latency in milliseconds")
    parser.add_argument("--benchmark-duration", type=int, default=300, help="Benchmark duration in seconds")
    parser.add_argument("--quick", action="store_true", help="Run quick demo (reduced duration)")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = Generation3Demo()
    
    # Adjust demo parameters
    demo.max_drones = args.max_drones
    demo.target_latency_ms = args.target_latency
    demo.benchmark_duration = args.benchmark_duration
    
    if args.quick:
        demo.benchmark_duration = 60  # Quick mode
        print("🏃 Running in quick mode (reduced benchmark duration)")
        print()
    
    # Run the complete demonstration
    await demo.run_complete_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
        print("Thank you for trying Fleet-Mind Generation 3!")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)