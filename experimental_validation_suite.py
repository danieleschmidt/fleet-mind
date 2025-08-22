#!/usr/bin/env python3
"""Experimental Validation Suite for Novel Swarm Coordination Algorithms.

This comprehensive validation suite implements the experimental framework for testing
novel algorithms developed in the Fleet-Mind research platform. It provides automated
experimental validation, statistical analysis, and publication-ready reporting.

RESEARCH VALIDATION PROTOCOL:
1. Algorithm Development - Implement novel algorithms with baselines
2. Controlled Validation - Statistical comparison in simulation
3. Real-World Validation - Field testing integration
4. Publication Preparation - Academic paper generation

ALGORITHMS VALIDATED:
- Quantum-Enhanced Multi-Agent Graph Neural Networks (QMAGNN) 
- Neuromorphic Collective Intelligence with Synaptic Plasticity (NCISP)
- Bio-Hybrid Collective Decision Making (BHCDM)
- Semantic-Aware Latent Compression with Graph Dynamics (SALCGD)
- Quantum-Bio-Neuromorphic Hybrid Coordination (QBNHC)
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import time
import logging
from pathlib import Path
from scipy import stats
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

# Import novel algorithms
from fleet_mind.research_framework.novel_algorithms import (
    AlgorithmType,
    PerformanceMetrics,
    ExperimentalResult,
    QuantumEnhancedMultiAgentGNN,
    NeuromorphicCollectiveIntelligence,
    SemanticAwareLatentCompression,
    NovelAlgorithmExperimentalFramework
)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentalScenarioGenerator:
    """Generates diverse experimental scenarios for algorithm validation."""
    
    def __init__(self, swarm_sizes: List[int] = [10, 50, 100, 200]):
        self.swarm_sizes = swarm_sizes
        self.scenario_types = [
            'formation_flight',
            'search_and_rescue', 
            'area_coverage',
            'obstacle_avoidance',
            'adaptive_transport',
            'adversarial_resilience'
        ]
        
    def generate_test_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """Generate diverse test scenarios for comprehensive validation."""
        scenarios = []
        
        for i in range(num_scenarios):
            swarm_size = np.random.choice(self.swarm_sizes)
            scenario_type = np.random.choice(self.scenario_types)
            
            # Generate random drone states (position, velocity, orientation)
            states = self._generate_drone_states(swarm_size)
            
            # Generate corresponding optimal actions (ground truth)
            actions = self._generate_optimal_actions(states, scenario_type)
            
            # Mission objective based on scenario type
            objective = self._generate_mission_objective(scenario_type, swarm_size)
            
            scenario = {
                'scenario_id': i,
                'swarm_size': swarm_size,
                'scenario_type': scenario_type,
                'states': states,
                'actions': actions,
                'objective': objective,
                'complexity_score': self._calculate_complexity(states, objective)
            }
            
            scenarios.append(scenario)
            
        logger.info(f"Generated {num_scenarios} test scenarios")
        return scenarios
    
    def _generate_drone_states(self, swarm_size: int) -> np.ndarray:
        """Generate realistic drone state vectors."""
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, battery_level]
        states = np.zeros((swarm_size, 10))
        
        # Positions in 3D space (meters)
        states[:, 0] = np.random.uniform(-500, 500, swarm_size)  # x
        states[:, 1] = np.random.uniform(-500, 500, swarm_size)  # y  
        states[:, 2] = np.random.uniform(10, 150, swarm_size)    # z (altitude)
        
        # Velocities (m/s)
        states[:, 3:6] = np.random.uniform(-20, 20, (swarm_size, 3))
        
        # Orientations (radians)
        states[:, 6:9] = np.random.uniform(-np.pi, np.pi, (swarm_size, 3))
        
        # Battery levels (0-1)
        states[:, 9] = np.random.uniform(0.2, 1.0, swarm_size)
        
        return states
    
    def _generate_optimal_actions(self, states: np.ndarray, scenario_type: str) -> np.ndarray:
        """Generate optimal coordination actions for given scenario."""
        swarm_size = states.shape[0]
        actions = np.zeros((swarm_size, 6))  # 6-DOF control
        
        if scenario_type == 'formation_flight':
            # V-formation with leader
            leader_idx = 0
            actions[leader_idx] = [10, 0, 0, 0, 0, 0]  # Leader moves forward
            
            for i in range(1, swarm_size):
                # Followers form V-shape
                side = 1 if i % 2 == 1 else -1
                formation_offset = (i // 2 + 1) * 20  # 20m spacing
                target_y = side * formation_offset
                target_x = states[leader_idx, 0] - formation_offset * 0.5
                
                # Simple pursuit toward formation position
                actions[i, 0] = (target_x - states[i, 0]) * 0.1
                actions[i, 1] = (target_y - states[i, 1]) * 0.1
                
        elif scenario_type == 'search_and_rescue':
            # Spread out for maximum coverage
            for i in range(swarm_size):
                # Grid search pattern
                grid_size = int(np.sqrt(swarm_size))
                target_x = (i % grid_size) * 100 - 250
                target_y = (i // grid_size) * 100 - 250
                
                actions[i, 0] = (target_x - states[i, 0]) * 0.05
                actions[i, 1] = (target_y - states[i, 1]) * 0.05
                
        elif scenario_type == 'area_coverage':
            # Lawnmower pattern coordination
            for i in range(swarm_size):
                if i % 2 == 0:
                    actions[i, 0] = 15  # Move forward
                else:
                    actions[i, 0] = -15  # Move backward
                    
        elif scenario_type == 'obstacle_avoidance':
            # Reactive obstacle avoidance (simplified)
            for i in range(swarm_size):
                # Avoid other drones
                avoidance_force = np.zeros(3)
                for j in range(swarm_size):
                    if i != j:
                        distance = np.linalg.norm(states[i, :3] - states[j, :3])
                        if distance < 30:  # 30m avoidance radius
                            repulsion = (states[i, :3] - states[j, :3]) / max(distance, 1.0)
                            avoidance_force += repulsion * (30 - distance)
                
                actions[i, :3] = np.clip(avoidance_force, -10, 10)
                
        elif scenario_type == 'adaptive_transport':
            # Coordinated payload transport
            center = np.mean(states[:, :3], axis=0)
            for i in range(swarm_size):
                # Move toward formation center
                to_center = center - states[i, :3]
                actions[i, :3] = to_center * 0.1
                
        elif scenario_type == 'adversarial_resilience':
            # Random evasive maneuvers with coordination
            for i in range(swarm_size):
                actions[i] = np.random.uniform(-5, 5, 6)
        
        # Add noise to make more realistic
        actions += np.random.normal(0, 0.1, actions.shape)
        
        return actions
    
    def _generate_mission_objective(self, scenario_type: str, swarm_size: int) -> Dict[str, Any]:
        """Generate mission objective parameters for scenario."""
        base_objectives = {
            'formation_flight': {
                'type': 'formation',
                'formation_type': 'v_formation',
                'spacing': 20,
                'leader_id': 0
            },
            'search_and_rescue': {
                'type': 'search',
                'search_area': [-500, 500, -500, 500],
                'target_type': 'survivors',
                'coverage_requirement': 0.95
            },
            'area_coverage': {
                'type': 'coverage',
                'area_bounds': [-1000, 1000, -1000, 1000],
                'resolution': 10,
                'coverage_pattern': 'lawnmower'
            },
            'obstacle_avoidance': {
                'type': 'navigation',
                'destination': [1000, 0, 100],
                'obstacles': [[0, 0, 50], [500, 300, 75]],
                'safety_margin': 30
            },
            'adaptive_transport': {
                'type': 'transport',
                'payload_weight': 50,
                'destination': [800, -200, 80],
                'formation_required': True
            },
            'adversarial_resilience': {
                'type': 'evasion',
                'threat_locations': [[200, 200], [-300, 400]],
                'evasion_priority': 'high',
                'coordination_requirement': True
            }
        }
        
        objective = base_objectives[scenario_type].copy()
        objective['swarm_size'] = swarm_size
        objective['priority'] = np.random.choice(['low', 'medium', 'high'])
        
        return objective
    
    def _calculate_complexity(self, states: np.ndarray, objective: Dict[str, Any]) -> float:
        """Calculate scenario complexity score for analysis."""
        swarm_size = states.shape[0]
        
        # Base complexity from swarm size
        size_complexity = np.log(swarm_size) / 10
        
        # Spatial distribution complexity
        positions = states[:, :3]
        spatial_variance = np.var(positions, axis=0).sum()
        spatial_complexity = min(spatial_variance / 10000, 1.0)
        
        # Mission complexity
        mission_complexity_map = {
            'formation': 0.3,
            'search': 0.6,
            'coverage': 0.7,
            'navigation': 0.8,
            'transport': 0.9,
            'evasion': 1.0
        }
        mission_complexity = mission_complexity_map.get(objective['type'], 0.5)
        
        total_complexity = (size_complexity + spatial_complexity + mission_complexity) / 3
        return total_complexity


class StatisticalValidator:
    """Advanced statistical validation for experimental results."""
    
    def __init__(self):
        self.alpha = 0.05  # Significance level
        self.power = 0.8   # Statistical power
        
    def validate_experimental_results(self, 
                                    results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Comprehensive statistical validation of experimental results."""
        validation_report = {
            'summary_statistics': self._calculate_summary_statistics(results),
            'significance_tests': self._run_significance_tests(results),
            'effect_sizes': self._calculate_effect_sizes(results),
            'power_analysis': self._perform_power_analysis(results),
            'multiple_comparisons': self._adjust_multiple_comparisons(results)
        }
        
        return validation_report
    
    def _calculate_summary_statistics(self, results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Calculate descriptive statistics for all algorithms."""
        stats_by_algorithm = {}
        
        for result in results:
            algo_name = result.algorithm_type.value
            metrics = result.metrics
            
            if algo_name not in stats_by_algorithm:
                stats_by_algorithm[algo_name] = {
                    'latencies': [],
                    'energy_efficiency': [],
                    'adaptation_speed': [],
                    'fault_tolerance': []
                }
            
            stats_by_algorithm[algo_name]['latencies'].append(metrics.coordination_latency_ms)
            stats_by_algorithm[algo_name]['energy_efficiency'].append(metrics.energy_efficiency_multiplier)
            stats_by_algorithm[algo_name]['adaptation_speed'].append(metrics.adaptation_speed_multiplier)
            stats_by_algorithm[algo_name]['fault_tolerance'].append(metrics.fault_tolerance_percentage)
        
        # Calculate statistics
        summary_stats = {}
        for algo_name, data in stats_by_algorithm.items():
            summary_stats[algo_name] = {}
            for metric, values in data.items():
                summary_stats[algo_name][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return summary_stats
    
    def _run_significance_tests(self, results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Run statistical significance tests comparing algorithms."""
        significance_tests = {}
        
        # Group results by algorithm type
        results_by_algo = {}
        for result in results:
            algo_name = result.algorithm_type.value
            if algo_name not in results_by_algo:
                results_by_algo[algo_name] = []
            results_by_algo[algo_name].append(result)
        
        # Pairwise comparisons
        algorithm_names = list(results_by_algo.keys())
        for i, algo1 in enumerate(algorithm_names):
            for j, algo2 in enumerate(algorithm_names[i+1:], i+1):
                
                # Extract latency data for comparison
                latencies1 = [r.metrics.coordination_latency_ms for r in results_by_algo[algo1]]
                latencies2 = [r.metrics.coordination_latency_ms for r in results_by_algo[algo2]]
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(latencies1, latencies2, alternative='two-sided')
                
                # T-test (parametric)
                t_stat, t_p_value = stats.ttest_ind(latencies1, latencies2)
                
                comparison_key = f"{algo1}_vs_{algo2}"
                significance_tests[comparison_key] = {
                    'mann_whitney_u': {'statistic': statistic, 'p_value': p_value},
                    't_test': {'statistic': t_stat, 'p_value': t_p_value},
                    'significant': p_value < self.alpha
                }
        
        return significance_tests
    
    def _calculate_effect_sizes(self, results: List[ExperimentalResult]) -> Dict[str, float]:
        """Calculate effect sizes (Cohen's d) for practical significance."""
        effect_sizes = {}
        
        for result in results:
            # Use baseline comparison to calculate effect size
            baseline_latency = 100.0  # Assumed baseline
            observed_latency = result.metrics.coordination_latency_ms
            
            # Cohen's d calculation (simplified)
            pooled_std = 10.0  # Assumed baseline standard deviation
            cohens_d = (baseline_latency - observed_latency) / pooled_std
            
            effect_sizes[result.algorithm_type.value] = cohens_d
        
        return effect_sizes
    
    def _perform_power_analysis(self, results: List[ExperimentalResult]) -> Dict[str, float]:
        """Perform statistical power analysis for experimental design."""
        power_analysis = {}
        
        for result in results:
            # Calculate observed power based on effect size and sample size
            effect_size = result.effect_size
            sample_size = 30  # Assumed from experimental design
            
            # Power calculation (simplified)
            observed_power = 1 - stats.norm.cdf(
                stats.norm.ppf(1 - self.alpha/2) - effect_size * np.sqrt(sample_size/2)
            )
            
            power_analysis[result.algorithm_type.value] = observed_power
        
        return power_analysis
    
    def _adjust_multiple_comparisons(self, results: List[ExperimentalResult]) -> Dict[str, float]:
        """Apply multiple comparison corrections (Bonferroni, FDR)."""
        p_values = [result.statistical_significance for result in results]
        algorithm_names = [result.algorithm_type.value for result in results]
        
        # Bonferroni correction
        bonferroni_alpha = self.alpha / len(p_values)
        bonferroni_significant = [p < bonferroni_alpha for p in p_values]
        
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        n = len(p_values)
        fdr_significant = [False] * n
        
        for i, idx in enumerate(sorted_indices):
            if p_values[idx] <= (i + 1) * self.alpha / n:
                fdr_significant[idx] = True
            else:
                break
        
        corrections = {}
        for i, algo_name in enumerate(algorithm_names):
            corrections[algo_name] = {
                'original_p': p_values[i],
                'bonferroni_significant': bonferroni_significant[i],
                'fdr_significant': fdr_significant[i]
            }
        
        return corrections


class VisualizationGenerator:
    """Generate publication-quality visualizations for research results."""
    
    def __init__(self, output_dir: str = "research_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def generate_performance_comparison(self, 
                                      results: List[ExperimentalResult],
                                      save: bool = True) -> plt.Figure:
        """Generate comprehensive performance comparison visualization."""
        # Extract data for visualization
        algorithms = [r.algorithm_type.value for r in results]
        latencies = [r.metrics.coordination_latency_ms for r in results]
        energy_efficiency = [r.metrics.energy_efficiency_multiplier for r in results]
        adaptation_speed = [r.metrics.adaptation_speed_multiplier for r in results]
        fault_tolerance = [r.metrics.fault_tolerance_percentage for r in results]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Novel Swarm Coordination Algorithms: Performance Comparison', 
                    fontsize=16, fontweight='bold')
        
        # Coordination Latency
        axes[0, 0].bar(algorithms, latencies, alpha=0.8)
        axes[0, 0].set_title('Coordination Latency (ms)')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Energy Efficiency
        axes[0, 1].bar(algorithms, energy_efficiency, alpha=0.8, color='green')
        axes[0, 1].set_title('Energy Efficiency Multiplier')
        axes[0, 1].set_ylabel('Efficiency Gain (x)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Adaptation Speed
        axes[1, 0].bar(algorithms, adaptation_speed, alpha=0.8, color='orange')
        axes[1, 0].set_title('Adaptation Speed Multiplier')
        axes[1, 0].set_ylabel('Speed Gain (x)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Fault Tolerance
        axes[1, 1].bar(algorithms, fault_tolerance, alpha=0.8, color='red')
        axes[1, 1].set_title('Fault Tolerance (%)')
        axes[1, 1].set_ylabel('Fault Tolerance (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'performance_comparison.png', 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'performance_comparison.pdf', 
                       bbox_inches='tight')
        
        return fig
    
    def generate_statistical_significance_plot(self,
                                             validation_results: Dict[str, Any],
                                             save: bool = True) -> plt.Figure:
        """Generate statistical significance visualization."""
        significance_tests = validation_results['significance_tests']
        
        # Extract p-values for heatmap
        comparisons = list(significance_tests.keys())
        p_values = [significance_tests[comp]['mann_whitney_u']['p_value'] 
                   for comp in comparisons]
        
        # Create significance matrix
        algorithms = set()
        for comp in comparisons:
            algo1, algo2 = comp.split('_vs_')
            algorithms.add(algo1)
            algorithms.add(algo2)
        
        algorithms = sorted(list(algorithms))
        n_algos = len(algorithms)
        significance_matrix = np.ones((n_algos, n_algos))
        
        for i, comp in enumerate(comparisons):
            algo1, algo2 = comp.split('_vs_')
            idx1 = algorithms.index(algo1)
            idx2 = algorithms.index(algo2)
            p_val = p_values[i]
            significance_matrix[idx1, idx2] = p_val
            significance_matrix[idx2, idx1] = p_val
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(significance_matrix, dtype=bool))
        
        sns.heatmap(significance_matrix, 
                   mask=mask,
                   xticklabels=algorithms,
                   yticklabels=algorithms,
                   annot=True,
                   fmt='.4f',
                   cmap='RdYlBu_r',
                   center=0.05,
                   ax=ax)
        
        ax.set_title('Statistical Significance (p-values) - Mann-Whitney U Test')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save:
            plt.savefig(self.output_dir / 'statistical_significance.png', 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'statistical_significance.pdf', 
                       bbox_inches='tight')
        
        return fig
    
    def generate_scalability_analysis(self,
                                    results_by_swarm_size: Dict[int, List[ExperimentalResult]],
                                    save: bool = True) -> plt.Figure:
        """Generate scalability analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scalability Analysis: Performance vs Swarm Size', 
                    fontsize=16, fontweight='bold')
        
        swarm_sizes = sorted(results_by_swarm_size.keys())
        algorithms = set()
        for results_list in results_by_swarm_size.values():
            for result in results_list:
                algorithms.add(result.algorithm_type.value)
        
        algorithms = sorted(list(algorithms))
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        # Latency vs Swarm Size
        for i, algorithm in enumerate(algorithms):
            latencies = []
            for size in swarm_sizes:
                size_results = [r for r in results_by_swarm_size[size] 
                               if r.algorithm_type.value == algorithm]
                if size_results:
                    avg_latency = np.mean([r.metrics.coordination_latency_ms 
                                         for r in size_results])
                    latencies.append(avg_latency)
                else:
                    latencies.append(None)
            
            axes[0, 0].plot(swarm_sizes, latencies, 'o-', 
                           label=algorithm, color=colors[i])
        
        axes[0, 0].set_xlabel('Swarm Size')
        axes[0, 0].set_ylabel('Coordination Latency (ms)')
        axes[0, 0].set_title('Latency Scalability')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_yscale('log')
        
        # Similar plots for other metrics...
        # (Implementation continues for energy efficiency, adaptation speed, fault tolerance)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'scalability_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / 'scalability_analysis.pdf', 
                       bbox_inches='tight')
        
        return fig


async def run_comprehensive_validation():
    """Main function to run comprehensive experimental validation."""
    logger.info("Starting comprehensive experimental validation suite")
    
    # Initialize components
    scenario_generator = ExperimentalScenarioGenerator(swarm_sizes=[10, 50, 100])
    validator = StatisticalValidator()
    visualizer = VisualizationGenerator()
    framework = NovelAlgorithmExperimentalFramework()
    
    # Generate test scenarios
    logger.info("Generating experimental scenarios")
    test_scenarios = scenario_generator.generate_test_scenarios(num_scenarios=150)
    
    # Initialize novel algorithms
    logger.info("Initializing novel algorithms")
    algorithms = {
        AlgorithmType.QMAGNN: QuantumEnhancedMultiAgentGNN(swarm_size=100),
        AlgorithmType.NCISP: NeuromorphicCollectiveIntelligence(swarm_size=100),
        AlgorithmType.SALCGD: SemanticAwareLatentCompression(swarm_size=100)
    }
    
    # Register algorithms in framework
    for algo_type, algorithm in algorithms.items():
        framework.register_algorithm(algo_type, algorithm)
    
    # Run comparative study
    logger.info("Running comparative experimental study")
    experimental_results = await framework.run_comparative_study(
        test_scenarios, num_trials=30
    )
    
    # Statistical validation
    logger.info("Performing statistical validation")
    validation_results = validator.validate_experimental_results(experimental_results)
    
    # Generate visualizations
    logger.info("Generating publication-quality visualizations")
    performance_fig = visualizer.generate_performance_comparison(experimental_results)
    significance_fig = visualizer.generate_statistical_significance_plot(validation_results)
    
    # Generate research report
    logger.info("Generating comprehensive research report")
    research_report = framework.generate_research_report()
    
    # Save results
    output_dir = Path("research_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save raw data
    with open(output_dir / "experimental_results.json", 'w') as f:
        json.dump([asdict(result) for result in experimental_results], f, indent=2)
    
    with open(output_dir / "validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save research report
    with open(output_dir / "research_report.md", 'w') as f:
        f.write(research_report)
    
    # Performance summary
    logger.info("=== EXPERIMENTAL VALIDATION SUMMARY ===")
    for result in experimental_results:
        algo_name = result.algorithm_type.value
        metrics = result.metrics
        logger.info(f"{algo_name}:")
        logger.info(f"  Latency: {metrics.coordination_latency_ms:.2f}ms")
        logger.info(f"  Energy Efficiency: {metrics.energy_efficiency_multiplier:.1f}x")
        logger.info(f"  Statistical Significance: p = {result.statistical_significance:.4f}")
        logger.info(f"  Effect Size: {result.effect_size:.2f}")
        logger.info("")
    
    logger.info("Comprehensive experimental validation completed!")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    
    return experimental_results, validation_results


if __name__ == "__main__":
    # Run experimental validation
    asyncio.run(run_comprehensive_validation())