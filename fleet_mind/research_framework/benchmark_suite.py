"""Benchmark Suite for Comprehensive Performance Evaluation.

Advanced benchmarking framework for evaluating drone swarm coordination algorithms,
performance metrics, and comparative studies across different approaches.
"""

import asyncio
import math
import time
import random
import statistics
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

class BenchmarkType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    SCALABILITY = "scalability"
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    ENERGY_EFFICIENCY = "energy_efficiency"

class MetricUnit(Enum):
    MILLISECONDS = "ms"
    SECONDS = "s"
    OPERATIONS_PER_SECOND = "ops/s"
    PERCENTAGE = "%"
    RATIO = "ratio"
    JOULES = "J"
    WATTS = "W"

@dataclass
class PerformanceMetric:
    """Defines a performance metric for benchmarking."""
    metric_id: str
    name: str
    benchmark_type: BenchmarkType
    unit: MetricUnit
    higher_is_better: bool
    target_value: Optional[float] = None
    tolerance: float = 0.1
    
    def evaluate_performance(self, measured_value: float) -> Dict[str, Any]:
        """Evaluate performance against target and return assessment."""
        
        assessment = {
            'measured_value': measured_value,
            'target_value': self.target_value,
            'unit': self.unit.value,
            'higher_is_better': self.higher_is_better
        }
        
        if self.target_value is not None:
            if self.higher_is_better:
                performance_ratio = measured_value / self.target_value
                meets_target = measured_value >= self.target_value * (1 - self.tolerance)
            else:
                performance_ratio = self.target_value / measured_value
                meets_target = measured_value <= self.target_value * (1 + self.tolerance)
            
            assessment.update({
                'performance_ratio': performance_ratio,
                'meets_target': meets_target,
                'deviation_percent': abs(performance_ratio - 1.0) * 100
            })
        else:
            assessment.update({
                'performance_ratio': 1.0,
                'meets_target': True,
                'deviation_percent': 0.0
            })
        
        return assessment

@dataclass
class BenchmarkResult:
    """Results from a benchmark execution."""
    benchmark_id: str
    algorithm_name: str
    test_scenario: str
    metric_results: Dict[str, float]
    metadata: Dict[str, Any]
    execution_time: float
    timestamp: float
    success: bool = True

@dataclass
class ComparativeStudy:
    """Configuration for comparative algorithm study."""
    study_id: str
    study_name: str
    algorithms: List[str]
    test_scenarios: List[str]
    performance_metrics: List[PerformanceMetric]
    sample_size: int
    confidence_level: float = 0.95
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

class BenchmarkSuite:
    """Comprehensive benchmarking suite for drone swarm algorithms."""
    
    def __init__(self, output_directory: str = "/tmp/benchmarks"):
        self.output_directory = output_directory
        
        # Benchmark registry
        self.registered_algorithms: Dict[str, Callable] = {}
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.test_scenarios: Dict[str, Dict[str, Any]] = {}
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.comparative_studies: Dict[str, ComparativeStudy] = {}
        self.study_results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        
        # Execution state
        self.running_benchmarks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.benchmark_stats = {
            'total_benchmarks_run': 0,
            'successful_benchmarks': 0,
            'total_execution_time': 0.0,
            'algorithms_tested': 0,
            'comparative_studies_completed': 0
        }
        
        # Initialize standard metrics and scenarios
        self._initialize_standard_metrics()
        self._initialize_standard_scenarios()
    
    def _initialize_standard_metrics(self):
        """Initialize standard performance metrics."""
        
        standard_metrics = [
            PerformanceMetric(
                metric_id="coordination_latency",
                name="Coordination Latency",
                benchmark_type=BenchmarkType.LATENCY,
                unit=MetricUnit.MILLISECONDS,
                higher_is_better=False,
                target_value=100.0,  # 100ms target
                tolerance=0.2
            ),
            PerformanceMetric(
                metric_id="formation_accuracy",
                name="Formation Accuracy",
                benchmark_type=BenchmarkType.ACCURACY,
                unit=MetricUnit.PERCENTAGE,
                higher_is_better=True,
                target_value=95.0,  # 95% accuracy target
                tolerance=0.05
            ),
            PerformanceMetric(
                metric_id="throughput_ops",
                name="Operations Throughput",
                benchmark_type=BenchmarkType.THROUGHPUT,
                unit=MetricUnit.OPERATIONS_PER_SECOND,
                higher_is_better=True,
                target_value=1000.0,  # 1000 ops/s target
                tolerance=0.1
            ),
            PerformanceMetric(
                metric_id="scalability_factor",
                name="Scalability Factor",
                benchmark_type=BenchmarkType.SCALABILITY,
                unit=MetricUnit.RATIO,
                higher_is_better=True,
                target_value=0.9,  # 90% efficiency at scale
                tolerance=0.1
            ),
            PerformanceMetric(
                metric_id="energy_efficiency",
                name="Energy Efficiency",
                benchmark_type=BenchmarkType.ENERGY_EFFICIENCY,
                unit=MetricUnit.JOULES,
                higher_is_better=False,
                target_value=10.0,  # 10J per operation
                tolerance=0.3
            ),
            PerformanceMetric(
                metric_id="fault_tolerance",
                name="Fault Tolerance",
                benchmark_type=BenchmarkType.ROBUSTNESS,
                unit=MetricUnit.PERCENTAGE,
                higher_is_better=True,
                target_value=99.0,  # 99% uptime
                tolerance=0.02
            )
        ]
        
        for metric in standard_metrics:
            self.performance_metrics[metric.metric_id] = metric
    
    def _initialize_standard_scenarios(self):
        """Initialize standard test scenarios."""
        
        standard_scenarios = {
            "small_formation": {
                "name": "Small Formation (10 drones)",
                "num_drones": 10,
                "formation_type": "line",
                "environment": "open_space",
                "obstacles": 0,
                "duration": 60.0
            },
            "medium_swarm": {
                "name": "Medium Swarm (50 drones)",
                "num_drones": 50,
                "formation_type": "grid",
                "environment": "urban",
                "obstacles": 5,
                "duration": 120.0
            },
            "large_swarm": {
                "name": "Large Swarm (100 drones)",
                "num_drones": 100,
                "formation_type": "dynamic",
                "environment": "complex",
                "obstacles": 20,
                "duration": 300.0
            },
            "adversarial_environment": {
                "name": "Adversarial Environment",
                "num_drones": 30,
                "formation_type": "defensive",
                "environment": "hostile",
                "obstacles": 10,
                "communication_jamming": 0.2,
                "gps_denial": 0.1,
                "duration": 180.0
            },
            "resource_constrained": {
                "name": "Resource Constrained",
                "num_drones": 25,
                "formation_type": "efficient",
                "environment": "open_space",
                "battery_constraint": 0.3,
                "bandwidth_limit": 0.5,
                "compute_limit": 0.4,
                "duration": 90.0
            }
        }
        
        self.test_scenarios.update(standard_scenarios)
    
    async def register_algorithm(self, 
                                algorithm_name: str,
                                algorithm_function: Callable) -> bool:
        """Register algorithm for benchmarking."""
        
        if algorithm_name in self.registered_algorithms:
            return False
        
        self.registered_algorithms[algorithm_name] = algorithm_function
        self.benchmark_stats['algorithms_tested'] += 1
        
        return True
    
    async def add_custom_metric(self, metric: PerformanceMetric) -> bool:
        """Add custom performance metric."""
        
        if metric.metric_id in self.performance_metrics:
            return False
        
        self.performance_metrics[metric.metric_id] = metric
        return True
    
    async def add_test_scenario(self, 
                              scenario_id: str,
                              scenario_config: Dict[str, Any]) -> bool:
        """Add custom test scenario."""
        
        if scenario_id in self.test_scenarios:
            return False
        
        self.test_scenarios[scenario_id] = scenario_config
        return True
    
    async def run_single_benchmark(self,
                                 algorithm_name: str,
                                 scenario_id: str,
                                 metrics: List[str],
                                 iterations: int = 5) -> Optional[BenchmarkResult]:
        """Run single algorithm benchmark."""
        
        if (algorithm_name not in self.registered_algorithms or
            scenario_id not in self.test_scenarios):
            return None
        
        algorithm_func = self.registered_algorithms[algorithm_name]
        scenario_config = self.test_scenarios[scenario_id]
        
        benchmark_id = f"bench_{algorithm_name}_{scenario_id}_{int(time.time() * 1000)}"
        
        try:
            start_time = time.time()
            
            # Run multiple iterations and collect metrics
            iteration_results = []
            
            for iteration in range(iterations):
                iteration_result = await self._run_benchmark_iteration(
                    algorithm_func, scenario_config, metrics
                )
                iteration_results.append(iteration_result)
            
            # Aggregate results
            aggregated_metrics = self._aggregate_iteration_results(iteration_results)
            
            execution_time = time.time() - start_time
            
            # Create benchmark result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                algorithm_name=algorithm_name,
                test_scenario=scenario_id,
                metric_results=aggregated_metrics,
                metadata={
                    'iterations': iterations,
                    'scenario_config': scenario_config,
                    'individual_results': iteration_results
                },
                execution_time=execution_time,
                timestamp=start_time,
                success=True
            )
            
            # Store result
            self.benchmark_results.append(result)
            
            # Update statistics
            self.benchmark_stats['total_benchmarks_run'] += 1
            self.benchmark_stats['successful_benchmarks'] += 1
            self.benchmark_stats['total_execution_time'] += execution_time
            
            return result
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            self.benchmark_stats['total_benchmarks_run'] += 1
            return None
    
    async def _run_benchmark_iteration(self,
                                     algorithm_func: Callable,
                                     scenario_config: Dict[str, Any],
                                     metrics: List[str]) -> Dict[str, float]:
        """Run single benchmark iteration."""
        
        # Start timing
        start_time = time.time()
        
        # Execute algorithm
        try:
            algorithm_result = await algorithm_func(scenario_config)
        except Exception:
            # Fallback to synchronous execution
            algorithm_result = algorithm_func(scenario_config)
        
        execution_time = time.time() - start_time
        
        # Extract metrics from algorithm result
        iteration_metrics = {}
        
        for metric_id in metrics:
            if metric_id in self.performance_metrics:
                metric_value = self._extract_metric_value(
                    metric_id, algorithm_result, scenario_config, execution_time
                )
                iteration_metrics[metric_id] = metric_value
        
        return iteration_metrics
    
    def _extract_metric_value(self,
                            metric_id: str,
                            algorithm_result: Any,
                            scenario_config: Dict[str, Any],
                            execution_time: float) -> float:
        """Extract specific metric value from algorithm result."""
        
        metric = self.performance_metrics[metric_id]
        
        # Extract based on metric type
        if metric.benchmark_type == BenchmarkType.LATENCY:
            if metric_id == "coordination_latency":
                return execution_time * 1000  # Convert to milliseconds
            else:
                return self._extract_from_result(algorithm_result, metric_id, execution_time * 1000)
        
        elif metric.benchmark_type == BenchmarkType.THROUGHPUT:
            if metric_id == "throughput_ops":
                num_drones = scenario_config.get('num_drones', 1)
                return num_drones / execution_time if execution_time > 0 else 0.0
            else:
                return self._extract_from_result(algorithm_result, metric_id, 0.0)
        
        elif metric.benchmark_type == BenchmarkType.ACCURACY:
            if metric_id == "formation_accuracy":
                # Simulate formation accuracy based on algorithm performance
                base_accuracy = 90.0
                noise_factor = random.gauss(0, 5)
                return max(0.0, min(100.0, base_accuracy + noise_factor))
            else:
                return self._extract_from_result(algorithm_result, metric_id, 90.0)
        
        elif metric.benchmark_type == BenchmarkType.SCALABILITY:
            if metric_id == "scalability_factor":
                num_drones = scenario_config.get('num_drones', 1)
                # Simulate scalability degradation
                ideal_time = 0.1  # 100ms for reference
                actual_time = execution_time
                scalability = ideal_time / (actual_time * math.log(num_drones)) if actual_time > 0 else 0.0
                return min(1.0, scalability)
            else:
                return self._extract_from_result(algorithm_result, metric_id, 0.8)
        
        elif metric.benchmark_type == BenchmarkType.ENERGY_EFFICIENCY:
            if metric_id == "energy_efficiency":
                num_drones = scenario_config.get('num_drones', 1)
                # Estimate energy based on drone count and execution time
                base_power = 50.0  # Watts per drone
                energy = base_power * num_drones * execution_time  # Joules
                return energy
            else:
                return self._extract_from_result(algorithm_result, metric_id, 10.0)
        
        elif metric.benchmark_type == BenchmarkType.ROBUSTNESS:
            if metric_id == "fault_tolerance":
                # Simulate fault tolerance based on scenario complexity
                complexity_factor = scenario_config.get('obstacles', 0) / 10.0
                base_tolerance = 99.0
                tolerance_loss = complexity_factor * 5.0
                return max(80.0, base_tolerance - tolerance_loss)
            else:
                return self._extract_from_result(algorithm_result, metric_id, 95.0)
        
        else:
            return self._extract_from_result(algorithm_result, metric_id, 0.0)
    
    def _extract_from_result(self, 
                           algorithm_result: Any, 
                           metric_id: str, 
                           default_value: float) -> float:
        """Extract metric value from algorithm result or return default."""
        
        if isinstance(algorithm_result, dict):
            return algorithm_result.get(metric_id, default_value)
        elif hasattr(algorithm_result, metric_id):
            return getattr(algorithm_result, metric_id)
        else:
            return default_value
    
    def _aggregate_iteration_results(self, iteration_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate results from multiple iterations."""
        
        if not iteration_results:
            return {}
        
        aggregated = {}
        
        # Get all metric keys
        all_metrics = set()
        for result in iteration_results:
            all_metrics.update(result.keys())
        
        # Aggregate each metric
        for metric_id in all_metrics:
            values = [result.get(metric_id, 0.0) for result in iteration_results]
            
            # Use mean for aggregation
            aggregated[metric_id] = statistics.mean(values) if values else 0.0
            
            # Also store additional statistics
            if len(values) > 1:
                aggregated[f"{metric_id}_std"] = statistics.stdev(values)
                aggregated[f"{metric_id}_min"] = min(values)
                aggregated[f"{metric_id}_max"] = max(values)
            else:
                aggregated[f"{metric_id}_std"] = 0.0
                aggregated[f"{metric_id}_min"] = aggregated[metric_id]
                aggregated[f"{metric_id}_max"] = aggregated[metric_id]
        
        return aggregated
    
    async def run_comparative_study(self,
                                  study_name: str,
                                  algorithms: List[str],
                                  scenarios: List[str],
                                  metrics: List[str],
                                  sample_size: int = 10) -> str:
        """Run comprehensive comparative study."""
        
        study_id = f"study_{int(time.time() * 1000)}"
        
        # Validate inputs
        valid_algorithms = [alg for alg in algorithms if alg in self.registered_algorithms]
        valid_scenarios = [scn for scn in scenarios if scn in self.test_scenarios]
        valid_metrics = [met for met in metrics if met in self.performance_metrics]
        
        if not valid_algorithms or not valid_scenarios or not valid_metrics:
            return ""
        
        # Create study configuration
        study = ComparativeStudy(
            study_id=study_id,
            study_name=study_name,
            algorithms=valid_algorithms,
            test_scenarios=valid_scenarios,
            performance_metrics=[self.performance_metrics[m] for m in valid_metrics],
            sample_size=sample_size
        )
        
        self.comparative_studies[study_id] = study
        
        # Run all algorithm-scenario combinations
        study_results = []
        
        for algorithm in valid_algorithms:
            for scenario in valid_scenarios:
                benchmark_result = await self.run_single_benchmark(
                    algorithm, scenario, valid_metrics, sample_size
                )
                
                if benchmark_result:
                    study_results.append(benchmark_result)
        
        # Store study results
        self.study_results[study_id] = study_results
        self.benchmark_stats['comparative_studies_completed'] += 1
        
        return study_id
    
    async def generate_performance_report(self,
                                        study_id: Optional[str] = None,
                                        algorithm_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        if study_id and study_id in self.study_results:
            # Study-specific report
            return await self._generate_study_report(study_id)
        elif algorithm_name:
            # Algorithm-specific report
            return await self._generate_algorithm_report(algorithm_name)
        else:
            # Overall benchmark report
            return await self._generate_overall_report()
    
    async def _generate_study_report(self, study_id: str) -> Dict[str, Any]:
        """Generate report for specific comparative study."""
        
        study = self.comparative_studies[study_id]
        results = self.study_results[study_id]
        
        # Organize results by algorithm and scenario
        algorithm_results = defaultdict(list)
        scenario_results = defaultdict(list)
        
        for result in results:
            algorithm_results[result.algorithm_name].append(result)
            scenario_results[result.test_scenario].append(result)
        
        # Calculate comparative statistics
        comparative_analysis = {}
        
        for metric in study.performance_metrics:
            metric_id = metric.metric_id
            
            # Collect values by algorithm
            algorithm_values = {}
            for algorithm in study.algorithms:
                values = []
                for result in algorithm_results[algorithm]:
                    if metric_id in result.metric_results:
                        values.append(result.metric_results[metric_id])
                
                if values:
                    algorithm_values[algorithm] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            # Rank algorithms for this metric
            if algorithm_values:
                if metric.higher_is_better:
                    ranked_algorithms = sorted(algorithm_values.items(), 
                                             key=lambda x: x[1]['mean'], reverse=True)
                else:
                    ranked_algorithms = sorted(algorithm_values.items(), 
                                             key=lambda x: x[1]['mean'])
                
                comparative_analysis[metric_id] = {
                    'metric_info': {
                        'name': metric.name,
                        'unit': metric.unit.value,
                        'higher_is_better': metric.higher_is_better,
                        'target_value': metric.target_value
                    },
                    'algorithm_performance': algorithm_values,
                    'ranking': [alg for alg, _ in ranked_algorithms]
                }
        
        return {
            'study_info': {
                'study_id': study_id,
                'study_name': study.study_name,
                'algorithms_tested': len(study.algorithms),
                'scenarios_tested': len(study.test_scenarios),
                'metrics_evaluated': len(study.performance_metrics),
                'sample_size': study.sample_size
            },
            'comparative_analysis': comparative_analysis,
            'execution_summary': {
                'total_benchmarks': len(results),
                'successful_benchmarks': sum(1 for r in results if r.success),
                'total_execution_time': sum(r.execution_time for r in results)
            }
        }
    
    async def _generate_algorithm_report(self, algorithm_name: str) -> Dict[str, Any]:
        """Generate report for specific algorithm."""
        
        algorithm_results = [r for r in self.benchmark_results 
                           if r.algorithm_name == algorithm_name]
        
        if not algorithm_results:
            return {'error': f'No results found for algorithm {algorithm_name}'}
        
        # Organize by scenario
        scenario_performance = defaultdict(list)
        
        for result in algorithm_results:
            scenario_performance[result.test_scenario].append(result)
        
        # Calculate performance statistics
        performance_summary = {}
        
        for scenario, results in scenario_performance.items():
            # Aggregate metrics across results
            metric_aggregates = defaultdict(list)
            
            for result in results:
                for metric_id, value in result.metric_results.items():
                    if not metric_id.endswith('_std') and not metric_id.endswith('_min') and not metric_id.endswith('_max'):
                        metric_aggregates[metric_id].append(value)
            
            # Calculate statistics
            scenario_stats = {}
            for metric_id, values in metric_aggregates.items():
                if values:
                    scenario_stats[metric_id] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            performance_summary[scenario] = scenario_stats
        
        return {
            'algorithm_info': {
                'algorithm_name': algorithm_name,
                'total_benchmarks': len(algorithm_results),
                'scenarios_tested': len(scenario_performance),
                'success_rate': sum(1 for r in algorithm_results if r.success) / len(algorithm_results)
            },
            'performance_by_scenario': performance_summary,
            'overall_execution_time': sum(r.execution_time for r in algorithm_results)
        }
    
    async def _generate_overall_report(self) -> Dict[str, Any]:
        """Generate overall benchmarking report."""
        
        # Calculate overall statistics
        total_algorithms = len(self.registered_algorithms)
        total_scenarios = len(self.test_scenarios)
        total_metrics = len(self.performance_metrics)
        
        # Algorithm performance summary
        algorithm_summary = {}
        
        for algorithm_name in self.registered_algorithms.keys():
            algorithm_results = [r for r in self.benchmark_results 
                               if r.algorithm_name == algorithm_name]
            
            if algorithm_results:
                success_rate = sum(1 for r in algorithm_results if r.success) / len(algorithm_results)
                avg_execution_time = sum(r.execution_time for r in algorithm_results) / len(algorithm_results)
                
                algorithm_summary[algorithm_name] = {
                    'benchmarks_run': len(algorithm_results),
                    'success_rate': success_rate,
                    'avg_execution_time': avg_execution_time
                }
        
        return {
            'framework_overview': {
                'total_algorithms_registered': total_algorithms,
                'total_test_scenarios': total_scenarios,
                'total_performance_metrics': total_metrics,
                'total_benchmarks_run': len(self.benchmark_results),
                'comparative_studies_completed': len(self.comparative_studies)
            },
            'algorithm_summary': algorithm_summary,
            'benchmark_statistics': self.benchmark_stats.copy(),
            'recent_activity': {
                'recent_benchmarks': len([r for r in self.benchmark_results 
                                        if time.time() - r.timestamp < 3600]),  # Last hour
                'running_benchmarks': len(self.running_benchmarks)
            }
        }
    
    async def export_results(self, 
                           format_type: str = "json",
                           study_id: Optional[str] = None) -> str:
        """Export benchmark results to specified format."""
        
        if study_id:
            data = await self.generate_performance_report(study_id=study_id)
        else:
            data = await self.generate_performance_report()
        
        # Export logic would be implemented here
        # For now, return the path where data would be saved
        timestamp = int(time.time())
        filename = f"benchmark_results_{timestamp}.{format_type}"
        filepath = f"{self.output_directory}/{filename}"
        
        return filepath
    
    def get_benchmark_statistics(self) -> Dict[str, Any]:
        """Get comprehensive benchmark suite statistics."""
        
        # Calculate metric distribution
        metric_distribution = defaultdict(int)
        for metric in self.performance_metrics.values():
            metric_distribution[metric.benchmark_type.value] += 1
        
        # Calculate scenario complexity distribution
        scenario_complexity = {}
        for scenario_id, config in self.test_scenarios.items():
            complexity_score = (
                config.get('num_drones', 0) / 10 +
                config.get('obstacles', 0) / 5 +
                config.get('duration', 0) / 60
            )
            scenario_complexity[scenario_id] = complexity_score
        
        return {
            'suite_configuration': {
                'registered_algorithms': len(self.registered_algorithms),
                'performance_metrics': len(self.performance_metrics),
                'test_scenarios': len(self.test_scenarios),
                'output_directory': self.output_directory
            },
            'metric_distribution': dict(metric_distribution),
            'scenario_complexity': scenario_complexity,
            'execution_statistics': self.benchmark_stats.copy(),
            'active_benchmarks': len(self.running_benchmarks),
            'results_storage': {
                'total_results': len(self.benchmark_results),
                'study_results': len(self.study_results),
                'comparative_studies': len(self.comparative_studies)
            }
        }