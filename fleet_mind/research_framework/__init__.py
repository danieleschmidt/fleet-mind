"""Advanced Research Framework for Novel Algorithm Development.

This module provides comprehensive research capabilities:
- Novel algorithm development and benchmarking
- Academic publication-ready research infrastructure
- Open-source research contribution framework
- Automated experimental design and validation
"""

from .algorithm_research import AlgorithmResearcher, NovelAlgorithm, ResearchHypothesis
from .experimental_framework import ExperimentalFramework, ControlledExperiment, StatisticalAnalysis
from .benchmark_suite import BenchmarkSuite, PerformanceMetric, ComparativeStudy
from .publication_toolkit import PublicationToolkit, ResearchPaper, DataVisualization

__all__ = [
    "AlgorithmResearcher",
    "NovelAlgorithm", 
    "ResearchHypothesis",
    "ExperimentalFramework",
    "ControlledExperiment",
    "StatisticalAnalysis",
    "BenchmarkSuite",
    "PerformanceMetric",
    "ComparativeStudy", 
    "PublicationToolkit",
    "ResearchPaper",
    "DataVisualization",
]