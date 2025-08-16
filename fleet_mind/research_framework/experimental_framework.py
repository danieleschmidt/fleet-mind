"""Experimental Framework for Advanced Drone Swarm Research.

Comprehensive framework for conducting controlled experiments, statistical analysis,
and reproducible research in drone swarm coordination and AI algorithms.
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

class ExperimentType(Enum):
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    SCALABILITY_TEST = "scalability_test"
    ROBUSTNESS_ANALYSIS = "robustness_analysis"
    ABLATION_STUDY = "ablation_study"

class StatisticalTest(Enum):
    T_TEST = "t_test"
    ANOVA = "anova"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"

@dataclass
class ControlledExperiment:
    """Defines a controlled experiment configuration."""
    experiment_id: str
    experiment_type: ExperimentType
    hypothesis: str
    independent_variables: List[str]
    dependent_variables: List[str]
    control_conditions: Dict[str, Any]
    experimental_conditions: List[Dict[str, Any]]
    sample_size: int
    confidence_level: float = 0.95
    power: float = 0.8
    randomization_seed: Optional[int] = None
    
@dataclass
class ExperimentResult:
    """Results from a single experimental trial."""
    trial_id: str
    experiment_id: str
    condition: Dict[str, Any]
    measurements: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: float
    duration: float
    success: bool = True

@dataclass
class StatisticalAnalysis:
    """Statistical analysis configuration and results."""
    analysis_id: str
    test_type: StatisticalTest
    data_groups: List[List[float]]
    group_labels: List[str]
    alpha: float = 0.05
    results: Dict[str, Any] = field(default_factory=dict)
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the specified statistical test."""
        
        if self.test_type == StatisticalTest.T_TEST:
            return self._run_t_test()
        elif self.test_type == StatisticalTest.ANOVA:
            return self._run_anova()
        elif self.test_type == StatisticalTest.MANN_WHITNEY:
            return self._run_mann_whitney()
        elif self.test_type == StatisticalTest.KRUSKAL_WALLIS:
            return self._run_kruskal_wallis()
        elif self.test_type == StatisticalTest.CHI_SQUARE:
            return self._run_chi_square()
        else:
            return {'error': 'Unknown test type'}
    
    def _run_t_test(self) -> Dict[str, Any]:
        """Run independent samples t-test."""
        
        if len(self.data_groups) != 2:
            return {'error': 'T-test requires exactly 2 groups'}
        
        group1, group2 = self.data_groups
        
        if not group1 or not group2:
            return {'error': 'Empty data groups'}
        
        # Calculate means and standard deviations
        mean1 = statistics.mean(group1)
        mean2 = statistics.mean(group2)
        
        if len(group1) > 1:
            std1 = statistics.stdev(group1)
        else:
            std1 = 0.0
            
        if len(group2) > 1:
            std2 = statistics.stdev(group2)
        else:
            std2 = 0.0
        
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard error
        if n1 > 1 and n2 > 1:
            pooled_std = math.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            standard_error = pooled_std * math.sqrt(1/n1 + 1/n2)
            
            # t-statistic
            if standard_error > 0:
                t_stat = (mean1 - mean2) / standard_error
            else:
                t_stat = 0.0
            
            # Degrees of freedom
            df = n1 + n2 - 2
            
            # Effect size (Cohen's d)
            if pooled_std > 0:
                cohens_d = (mean1 - mean2) / pooled_std
            else:
                cohens_d = 0.0
        else:
            t_stat = 0.0
            df = 0
            cohens_d = 0.0
        
        # p-value approximation (simplified)
        p_value = self._approximate_p_value(abs(t_stat), df)
        
        return {
            'test_type': 't_test',
            'statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'effect_size': cohens_d,
            'mean_difference': mean1 - mean2,
            'group_means': [mean1, mean2],
            'group_stds': [std1, std2],
            'significant': p_value < self.alpha
        }
    
    def _run_anova(self) -> Dict[str, Any]:
        """Run one-way ANOVA."""
        
        if len(self.data_groups) < 2:
            return {'error': 'ANOVA requires at least 2 groups'}
        
        # Calculate group statistics
        group_means = []
        group_sizes = []
        all_values = []
        
        for group in self.data_groups:
            if group:
                group_means.append(statistics.mean(group))
                group_sizes.append(len(group))
                all_values.extend(group)
            else:
                group_means.append(0.0)
                group_sizes.append(0)
        
        if not all_values:
            return {'error': 'No data available'}
        
        # Grand mean
        grand_mean = statistics.mean(all_values)
        
        # Between-group sum of squares
        ss_between = sum(n * (mean - grand_mean)**2 
                        for n, mean in zip(group_sizes, group_means))
        
        # Within-group sum of squares
        ss_within = 0.0
        for group, group_mean in zip(self.data_groups, group_means):
            ss_within += sum((x - group_mean)**2 for x in group)
        
        # Degrees of freedom
        df_between = len(self.data_groups) - 1
        df_within = len(all_values) - len(self.data_groups)
        
        # Mean squares
        if df_between > 0:
            ms_between = ss_between / df_between
        else:
            ms_between = 0.0
            
        if df_within > 0:
            ms_within = ss_within / df_within
        else:
            ms_within = 0.0
        
        # F-statistic
        if ms_within > 0:
            f_stat = ms_between / ms_within
        else:
            f_stat = 0.0
        
        # p-value approximation
        p_value = self._approximate_f_p_value(f_stat, df_between, df_within)
        
        # Effect size (eta-squared)
        total_ss = ss_between + ss_within
        if total_ss > 0:
            eta_squared = ss_between / total_ss
        else:
            eta_squared = 0.0
        
        return {
            'test_type': 'anova',
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'effect_size': eta_squared,
            'group_means': group_means,
            'grand_mean': grand_mean,
            'significant': p_value < self.alpha
        }
    
    def _run_mann_whitney(self) -> Dict[str, Any]:
        """Run Mann-Whitney U test (non-parametric)."""
        
        if len(self.data_groups) != 2:
            return {'error': 'Mann-Whitney test requires exactly 2 groups'}
        
        group1, group2 = self.data_groups
        
        if not group1 or not group2:
            return {'error': 'Empty data groups'}
        
        # Combine and rank all values
        combined = [(x, 1) for x in group1] + [(x, 2) for x in group2]
        combined.sort(key=lambda x: x[0])
        
        # Assign ranks
        ranks = {}
        for i, (value, group) in enumerate(combined):
            if value not in ranks:
                ranks[value] = []
            ranks[value].append(i + 1)
        
        # Handle ties by averaging ranks
        for value in ranks:
            avg_rank = sum(ranks[value]) / len(ranks[value])
            ranks[value] = avg_rank
        
        # Calculate rank sums
        rank_sum1 = sum(ranks[x] for x in group1)
        rank_sum2 = sum(ranks[x] for x in group2)
        
        n1, n2 = len(group1), len(group2)
        
        # U statistics
        u1 = rank_sum1 - n1 * (n1 + 1) / 2
        u2 = rank_sum2 - n2 * (n2 + 1) / 2
        
        # Test statistic (smaller U)
        u_stat = min(u1, u2)
        
        # p-value approximation (normal approximation for large samples)
        if n1 >= 8 and n2 >= 8:
            mean_u = n1 * n2 / 2
            std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            
            if std_u > 0:
                z_score = (u_stat - mean_u) / std_u
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
            else:
                p_value = 1.0
        else:
            p_value = 0.5  # Simplified for small samples
        
        return {
            'test_type': 'mann_whitney',
            'u_statistic': u_stat,
            'p_value': p_value,
            'rank_sum_1': rank_sum1,
            'rank_sum_2': rank_sum2,
            'significant': p_value < self.alpha
        }
    
    def _run_kruskal_wallis(self) -> Dict[str, Any]:
        """Run Kruskal-Wallis test (non-parametric ANOVA)."""
        
        if len(self.data_groups) < 2:
            return {'error': 'Kruskal-Wallis test requires at least 2 groups'}
        
        # Combine all values and assign group labels
        all_values = []
        group_labels = []
        
        for i, group in enumerate(self.data_groups):
            for value in group:
                all_values.append(value)
                group_labels.append(i)
        
        if not all_values:
            return {'error': 'No data available'}
        
        # Rank all values
        sorted_indices = sorted(range(len(all_values)), key=lambda i: all_values[i])
        ranks = [0] * len(all_values)
        
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        # Calculate rank sums for each group
        rank_sums = [0] * len(self.data_groups)
        group_sizes = [0] * len(self.data_groups)
        
        for i, group_label in enumerate(group_labels):
            rank_sums[group_label] += ranks[i]
            group_sizes[group_label] += 1
        
        # Kruskal-Wallis H statistic
        n = len(all_values)
        h_stat = (12 / (n * (n + 1))) * sum(
            (rank_sum**2 / group_size) for rank_sum, group_size in zip(rank_sums, group_sizes)
        ) - 3 * (n + 1)
        
        # Degrees of freedom
        df = len(self.data_groups) - 1
        
        # p-value approximation (chi-square distribution)
        p_value = self._approximate_chi_square_p_value(h_stat, df)
        
        return {
            'test_type': 'kruskal_wallis',
            'h_statistic': h_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'rank_sums': rank_sums,
            'significant': p_value < self.alpha
        }
    
    def _run_chi_square(self) -> Dict[str, Any]:
        """Run chi-square test of independence."""
        
        # For chi-square, data_groups should contain contingency table data
        if len(self.data_groups) < 2:
            return {'error': 'Chi-square test requires contingency table data'}
        
        # Simple 2x2 contingency table example
        observed = self.data_groups
        
        # Calculate row and column totals
        row_totals = [sum(row) for row in observed]
        col_totals = [sum(observed[i][j] for i in range(len(observed))) 
                     for j in range(len(observed[0]))]
        grand_total = sum(row_totals)
        
        if grand_total == 0:
            return {'error': 'No data in contingency table'}
        
        # Calculate expected frequencies
        expected = []
        for i in range(len(observed)):
            expected_row = []
            for j in range(len(observed[i])):
                expected_freq = (row_totals[i] * col_totals[j]) / grand_total
                expected_row.append(expected_freq)
            expected.append(expected_row)
        
        # Calculate chi-square statistic
        chi_square = 0.0
        for i in range(len(observed)):
            for j in range(len(observed[i])):
                if expected[i][j] > 0:
                    chi_square += (observed[i][j] - expected[i][j])**2 / expected[i][j]
        
        # Degrees of freedom
        df = (len(observed) - 1) * (len(observed[0]) - 1)
        
        # p-value approximation
        p_value = self._approximate_chi_square_p_value(chi_square, df)
        
        return {
            'test_type': 'chi_square',
            'chi_square_statistic': chi_square,
            'p_value': p_value,
            'degrees_freedom': df,
            'observed': observed,
            'expected': expected,
            'significant': p_value < self.alpha
        }
    
    def _approximate_p_value(self, t_stat: float, df: int) -> float:
        """Approximate p-value for t-distribution."""
        
        if df <= 0:
            return 1.0
        
        # Simple approximation using normal distribution for large df
        if df >= 30:
            return 2 * (1 - self._standard_normal_cdf(abs(t_stat)))
        else:
            # Rough approximation for small df
            adjusted_stat = t_stat * math.sqrt(df / (df + t_stat**2))
            return 2 * (1 - self._standard_normal_cdf(abs(adjusted_stat)))
    
    def _approximate_f_p_value(self, f_stat: float, df1: int, df2: int) -> float:
        """Approximate p-value for F-distribution."""
        
        if df1 <= 0 or df2 <= 0 or f_stat <= 0:
            return 1.0
        
        # Simple approximation
        if f_stat < 1.0:
            return 0.8
        elif f_stat < 2.0:
            return 0.3
        elif f_stat < 4.0:
            return 0.1
        elif f_stat < 6.0:
            return 0.05
        else:
            return 0.01
    
    def _approximate_chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Approximate p-value for chi-square distribution."""
        
        if df <= 0 or chi_square <= 0:
            return 1.0
        
        # Simple thresholds for common significance levels
        critical_values = {
            1: {0.05: 3.84, 0.01: 6.64, 0.001: 10.83},
            2: {0.05: 5.99, 0.01: 9.21, 0.001: 13.82},
            3: {0.05: 7.81, 0.01: 11.34, 0.001: 16.27}
        }
        
        if df in critical_values:
            if chi_square >= critical_values[df][0.001]:
                return 0.001
            elif chi_square >= critical_values[df][0.01]:
                return 0.01
            elif chi_square >= critical_values[df][0.05]:
                return 0.05
            else:
                return 0.1
        else:
            # Simple approximation for other df values
            if chi_square > df * 2:
                return 0.01
            elif chi_square > df * 1.5:
                return 0.05
            else:
                return 0.1
    
    def _standard_normal_cdf(self, x: float) -> float:
        """Approximate standard normal cumulative distribution function."""
        
        # Simple approximation using error function
        return 0.5 * (1 + self._erf(x / math.sqrt(2)))
    
    def _erf(self, x: float) -> float:
        """Approximate error function."""
        
        # Abramowitz and Stegun approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return sign * y

class ExperimentalFramework:
    """Comprehensive experimental framework for drone swarm research."""
    
    def __init__(self, base_output_dir: str = "/tmp/experiments"):
        self.base_output_dir = base_output_dir
        
        # Experiment management
        self.active_experiments: Dict[str, ControlledExperiment] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.statistical_analyses: Dict[str, StatisticalAnalysis] = {}
        
        # Execution tracking
        self.execution_queue: List[str] = []
        self.running_experiments: Dict[str, asyncio.Task] = {}
        
        # Framework statistics
        self.framework_stats = {
            'total_experiments': 0,
            'completed_experiments': 0,
            'total_trials': 0,
            'successful_trials': 0,
            'statistical_tests_run': 0
        }
        
        # Start experiment executor
        asyncio.create_task(self._experiment_executor())
    
    async def design_experiment(self,
                              experiment_type: ExperimentType,
                              hypothesis: str,
                              independent_variables: List[str],
                              dependent_variables: List[str],
                              conditions: List[Dict[str, Any]],
                              sample_size: int = 30) -> str:
        """Design a new controlled experiment."""
        
        experiment_id = f"exp_{experiment_type.value}_{int(time.time() * 1000)}"
        
        # Create experiment configuration
        experiment = ControlledExperiment(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            hypothesis=hypothesis,
            independent_variables=independent_variables,
            dependent_variables=dependent_variables,
            control_conditions=conditions[0] if conditions else {},
            experimental_conditions=conditions[1:] if len(conditions) > 1 else conditions,
            sample_size=sample_size,
            randomization_seed=random.randint(1, 1000000)
        )
        
        self.active_experiments[experiment_id] = experiment
        self.framework_stats['total_experiments'] += 1
        
        return experiment_id
    
    async def execute_experiment(self,
                               experiment_id: str,
                               trial_function: Callable,
                               **kwargs) -> bool:
        """Execute a designed experiment."""
        
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Set randomization seed for reproducibility
        if experiment.randomization_seed:
            random.seed(experiment.randomization_seed)
        
        # Create execution task
        execution_task = asyncio.create_task(
            self._run_experiment_trials(experiment, trial_function, **kwargs)
        )
        
        self.running_experiments[experiment_id] = execution_task
        
        return True
    
    async def _run_experiment_trials(self,
                                   experiment: ControlledExperiment,
                                   trial_function: Callable,
                                   **kwargs) -> bool:
        """Run all trials for an experiment."""
        
        try:
            all_conditions = [experiment.control_conditions] + experiment.experimental_conditions
            
            for condition in all_conditions:
                for trial_num in range(experiment.sample_size):
                    trial_id = f"{experiment.experiment_id}_trial_{trial_num}"
                    
                    # Run single trial
                    trial_result = await self._run_single_trial(
                        trial_id, experiment, condition, trial_function, **kwargs
                    )
                    
                    # Store result
                    self.experiment_results[experiment.experiment_id].append(trial_result)
                    self.framework_stats['total_trials'] += 1
                    
                    if trial_result.success:
                        self.framework_stats['successful_trials'] += 1
            
            # Mark experiment as completed
            self.framework_stats['completed_experiments'] += 1
            
            return True
            
        except Exception as e:
            print(f"Experiment {experiment.experiment_id} failed: {e}")
            return False
        
        finally:
            # Clean up
            if experiment.experiment_id in self.running_experiments:
                del self.running_experiments[experiment.experiment_id]
    
    async def _run_single_trial(self,
                              trial_id: str,
                              experiment: ControlledExperiment,
                              condition: Dict[str, Any],
                              trial_function: Callable,
                              **kwargs) -> ExperimentResult:
        """Run a single experimental trial."""
        
        start_time = time.time()
        
        try:
            # Execute trial with given conditions
            measurements = await trial_function(condition, **kwargs)
            
            if not isinstance(measurements, dict):
                measurements = {'result': measurements}
            
            # Validate dependent variables are measured
            missing_vars = set(experiment.dependent_variables) - set(measurements.keys())
            for var in missing_vars:
                measurements[var] = 0.0  # Default value for missing measurements
            
            success = True
            
        except Exception as e:
            measurements = {var: 0.0 for var in experiment.dependent_variables}
            success = False
        
        end_time = time.time()
        
        return ExperimentResult(
            trial_id=trial_id,
            experiment_id=experiment.experiment_id,
            condition=condition.copy(),
            measurements=measurements,
            metadata={'trial_function': trial_function.__name__},
            timestamp=start_time,
            duration=end_time - start_time,
            success=success
        )
    
    async def analyze_experiment_results(self,
                                       experiment_id: str,
                                       analysis_variables: List[str],
                                       test_type: StatisticalTest = StatisticalTest.ANOVA) -> Optional[str]:
        """Analyze experimental results using statistical tests."""
        
        if experiment_id not in self.experiment_results:
            return None
        
        results = self.experiment_results[experiment_id]
        experiment = self.active_experiments[experiment_id]
        
        # Group results by experimental conditions
        condition_groups = defaultdict(list)
        
        for result in results:
            # Create condition key
            condition_key = str(sorted(result.condition.items()))
            condition_groups[condition_key].append(result)
        
        # Analyze each dependent variable
        analysis_results = {}
        
        for variable in analysis_variables:
            if variable not in experiment.dependent_variables:
                continue
            
            # Extract data for each condition group
            data_groups = []
            group_labels = []
            
            for condition_key, group_results in condition_groups.items():
                group_data = [r.measurements.get(variable, 0.0) for r in group_results]
                if group_data:  # Only include non-empty groups
                    data_groups.append(group_data)
                    group_labels.append(condition_key)
            
            if len(data_groups) >= 2:
                # Create statistical analysis
                analysis_id = f"analysis_{experiment_id}_{variable}_{int(time.time() * 1000)}"
                
                analysis = StatisticalAnalysis(
                    analysis_id=analysis_id,
                    test_type=test_type,
                    data_groups=data_groups,
                    group_labels=group_labels
                )
                
                # Run analysis
                analysis.results = analysis.run_analysis()
                
                # Store analysis
                self.statistical_analyses[analysis_id] = analysis
                analysis_results[variable] = analysis
                
                self.framework_stats['statistical_tests_run'] += 1
        
        return analysis_results
    
    async def generate_experiment_report(self,
                                       experiment_id: str,
                                       include_raw_data: bool = False) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        
        if experiment_id not in self.active_experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.active_experiments[experiment_id]
        results = self.experiment_results.get(experiment_id, [])
        
        # Calculate basic statistics
        total_trials = len(results)
        successful_trials = sum(1 for r in results if r.success)
        success_rate = successful_trials / total_trials if total_trials > 0 else 0.0
        
        # Calculate descriptive statistics for dependent variables
        descriptive_stats = {}
        
        for variable in experiment.dependent_variables:
            values = [r.measurements.get(variable, 0.0) for r in results if r.success]
            
            if values:
                descriptive_stats[variable] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                descriptive_stats[variable] = {
                    'mean': 0.0, 'median': 0.0, 'std_dev': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        # Find related statistical analyses
        related_analyses = {
            aid: analysis for aid, analysis in self.statistical_analyses.items()
            if analysis.analysis_id.startswith(f"analysis_{experiment_id}")
        }
        
        report = {
            'experiment_info': {
                'experiment_id': experiment_id,
                'experiment_type': experiment.experiment_type.value,
                'hypothesis': experiment.hypothesis,
                'independent_variables': experiment.independent_variables,
                'dependent_variables': experiment.dependent_variables,
                'sample_size': experiment.sample_size,
                'confidence_level': experiment.confidence_level
            },
            'execution_summary': {
                'total_trials': total_trials,
                'successful_trials': successful_trials,
                'success_rate': success_rate,
                'total_conditions': len(experiment.experimental_conditions) + 1
            },
            'descriptive_statistics': descriptive_stats,
            'statistical_analyses': {
                aid: analysis.results for aid, analysis in related_analyses.items()
            }
        }
        
        if include_raw_data:
            report['raw_data'] = [
                {
                    'trial_id': r.trial_id,
                    'condition': r.condition,
                    'measurements': r.measurements,
                    'duration': r.duration,
                    'success': r.success
                }
                for r in results
            ]
        
        return report
    
    async def run_ablation_study(self,
                               base_configuration: Dict[str, Any],
                               components_to_ablate: List[str],
                               evaluation_function: Callable,
                               sample_size: int = 20) -> str:
        """Run ablation study to determine component importance."""
        
        # Design ablation experiment
        conditions = [base_configuration.copy()]  # Control (all components)
        
        # Create ablated conditions (remove one component at a time)
        for component in components_to_ablate:
            ablated_condition = base_configuration.copy()
            ablated_condition[component] = False  # Disable component
            conditions.append(ablated_condition)
        
        # Create experiment
        experiment_id = await self.design_experiment(
            experiment_type=ExperimentType.ABLATION_STUDY,
            hypothesis=f"Components {components_to_ablate} contribute to system performance",
            independent_variables=components_to_ablate,
            dependent_variables=['performance_score'],
            conditions=conditions,
            sample_size=sample_size
        )
        
        # Execute experiment
        await self.execute_experiment(experiment_id, evaluation_function)
        
        return experiment_id
    
    async def run_scalability_test(self,
                                 system_function: Callable,
                                 scale_parameters: List[int],
                                 performance_metrics: List[str],
                                 sample_size: int = 10) -> str:
        """Run scalability test across different system scales."""
        
        conditions = []
        for scale in scale_parameters:
            conditions.append({'scale': scale})
        
        # Create scalability experiment
        experiment_id = await self.design_experiment(
            experiment_type=ExperimentType.SCALABILITY_TEST,
            hypothesis=f"System performance scales with size up to {max(scale_parameters)}",
            independent_variables=['scale'],
            dependent_variables=performance_metrics,
            conditions=conditions,
            sample_size=sample_size
        )
        
        # Execute experiment
        await self.execute_experiment(experiment_id, system_function)
        
        return experiment_id
    
    async def compare_algorithms(self,
                               algorithms: Dict[str, Callable],
                               test_scenarios: List[Dict[str, Any]],
                               performance_metrics: List[str],
                               sample_size: int = 25) -> str:
        """Compare multiple algorithms across test scenarios."""
        
        conditions = []
        for algorithm_name in algorithms.keys():
            for scenario in test_scenarios:
                condition = scenario.copy()
                condition['algorithm'] = algorithm_name
                conditions.append(condition)
        
        # Create comparison experiment
        experiment_id = await self.design_experiment(
            experiment_type=ExperimentType.ALGORITHM_COMPARISON,
            hypothesis=f"Algorithms {list(algorithms.keys())} have different performance characteristics",
            independent_variables=['algorithm'] + list(test_scenarios[0].keys()),
            dependent_variables=performance_metrics,
            conditions=conditions,
            sample_size=sample_size
        )
        
        # Create unified trial function
        async def algorithm_trial(condition, **kwargs):
            algorithm_name = condition['algorithm']
            algorithm_func = algorithms[algorithm_name]
            
            # Remove algorithm from condition for passing to algorithm
            test_condition = {k: v for k, v in condition.items() if k != 'algorithm'}
            
            return await algorithm_func(test_condition, **kwargs)
        
        # Execute experiment
        await self.execute_experiment(experiment_id, algorithm_trial)
        
        return experiment_id
    
    async def _experiment_executor(self):
        """Background executor for queued experiments."""
        
        while True:
            try:
                # Clean up completed experiments
                completed_experiments = [
                    exp_id for exp_id, task in self.running_experiments.items()
                    if task.done()
                ]
                
                for exp_id in completed_experiments:
                    del self.running_experiments[exp_id]
                
                # Process execution queue
                if self.execution_queue and len(self.running_experiments) < 3:  # Max concurrent
                    experiment_id = self.execution_queue.pop(0)
                    
                    if experiment_id in self.active_experiments:
                        # Would execute experiment here if we had default trial function
                        pass
                
                await asyncio.sleep(1.0)
                
            except Exception:
                await asyncio.sleep(0.1)
    
    def get_framework_statistics(self) -> Dict[str, Any]:
        """Get comprehensive framework statistics."""
        
        # Calculate success rates
        total_experiments = self.framework_stats['total_experiments']
        total_trials = self.framework_stats['total_trials']
        
        experiment_success_rate = 0.0
        if total_experiments > 0:
            experiment_success_rate = self.framework_stats['completed_experiments'] / total_experiments
        
        trial_success_rate = 0.0
        if total_trials > 0:
            trial_success_rate = self.framework_stats['successful_trials'] / total_trials
        
        # Calculate average trial duration
        all_results = []
        for results_list in self.experiment_results.values():
            all_results.extend(results_list)
        
        avg_trial_duration = 0.0
        if all_results:
            avg_trial_duration = sum(r.duration for r in all_results) / len(all_results)
        
        return {
            'framework_overview': {
                'total_experiments_designed': total_experiments,
                'completed_experiments': self.framework_stats['completed_experiments'],
                'running_experiments': len(self.running_experiments),
                'queued_experiments': len(self.execution_queue)
            },
            'trial_statistics': {
                'total_trials_run': total_trials,
                'successful_trials': self.framework_stats['successful_trials'],
                'trial_success_rate': trial_success_rate,
                'average_trial_duration': avg_trial_duration
            },
            'analysis_statistics': {
                'statistical_tests_run': self.framework_stats['statistical_tests_run'],
                'total_analyses': len(self.statistical_analyses)
            },
            'performance_metrics': {
                'experiment_success_rate': experiment_success_rate,
                'framework_utilization': len(self.running_experiments) / 3.0  # Max concurrent
            },
            'experiment_types': {
                exp_type.value: sum(1 for exp in self.active_experiments.values() 
                                  if exp.experiment_type == exp_type)
                for exp_type in ExperimentType
            }
        }