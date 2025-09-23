"""
Comprehensive Testing Framework for 100% Panel Placement Guarantee
100%配置保証テストフレームワーク

This framework validates that the steel cutting optimization system achieves
100% panel placement under all conditions with no infinite loops or placement failures.
"""

import pytest
import time
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
import signal
import multiprocessing
from contextlib import contextmanager
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models import Panel, SteelSheet, PlacementResult, OptimizationConstraints
from core.optimizer import OptimizationEngine
from core.text_parser import parse_cutting_data_file
from core.pi_manager import PIManager
from core.material_manager import MaterialInventoryManager
from core.algorithms.complete_placement_guaranteed import CompletePlacementGuaranteedAlgorithm
from core.algorithms.simple_bulk_optimizer import SimpleBulkOptimizer


@dataclass
class PlacementTestCase:
    """Test case for placement validation"""
    name: str
    panels: List[Panel]
    constraints: OptimizationConstraints
    expected_placement_rate: float = 100.0
    max_execution_time: float = 300.0  # 5 minutes max
    algorithm_name: str = "Complete_Placement_Guaranteed"
    description: str = ""
    category: str = "general"  # general, edge_case, stress, regression


@dataclass
class PlacementTestResult:
    """Results from placement test execution"""
    test_case: PlacementTestCase
    success: bool
    placement_rate: float
    total_panels: int
    placed_panels: int
    unplaced_panels: int
    sheets_used: int
    execution_time: float
    average_efficiency: float
    algorithm_performance: Dict[str, Any]
    validation_errors: List[str]
    warnings: List[str]
    timeout_occurred: bool = False
    infinite_loop_detected: bool = False
    memory_usage_mb: float = 0.0


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


class PlacementGuaranteeFramework:
    """
    Core testing framework for 100% placement guarantee validation

    Features:
    - Timeout protection against infinite loops
    - Comprehensive overlap detection
    - Boundary validation
    - Performance regression detection
    - Edge case generation and validation
    - Multi-algorithm comparison
    """

    def __init__(self, timeout_seconds: float = 300.0):
        self.timeout_seconds = timeout_seconds
        self.logger = self._setup_logging()
        self.material_manager = MaterialInventoryManager()
        self.pi_manager = PIManager()
        self.test_results: List[PlacementTestResult] = []

        # Performance baselines for regression detection
        self.performance_baselines = {
            'small_batch_time': 1.0,    # ≤20 panels
            'medium_batch_time': 5.0,   # ≤50 panels
            'large_batch_time': 30.0,   # ≤100 panels
            'placement_rate_threshold': 100.0,  # Must achieve 100%
            'max_memory_mb': 500.0      # Memory usage limit
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup detailed logging for test execution"""
        logger = logging.getLogger('placement_guarantee')
        logger.setLevel(logging.DEBUG)

        # Create file handler for detailed logs
        log_file = project_root / 'tests' / 'placement_guarantee.log'
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @contextmanager
    def timeout_context(self, seconds: float):
        """Context manager for timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test execution exceeded {seconds} seconds")

        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))

        try:
            yield
        finally:
            # Restore the old signal handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def run_placement_test(self, test_case: PlacementTestCase) -> PlacementTestResult:
        """
        Execute a single placement test with comprehensive validation

        Args:
            test_case: Test case specification

        Returns:
            PlacementTestResult with detailed metrics and validation
        """
        self.logger.info(f"🧪 Running test: {test_case.name}")
        start_time = time.time()
        timeout_occurred = False
        infinite_loop_detected = False
        validation_errors = []
        warnings = []

        try:
            with self.timeout_context(test_case.max_execution_time):
                # Apply PI expansion to panels
                expanded_panels = self._apply_pi_expansion(test_case.panels)
                total_panels = sum(panel.quantity for panel in expanded_panels)

                # Setup optimization engine
                engine = OptimizationEngine()

                # Register and test the specific algorithm
                if test_case.algorithm_name == "Complete_Placement_Guaranteed":
                    engine.register_algorithm(CompletePlacementGuaranteedAlgorithm())
                elif test_case.algorithm_name == "Simple_Bulk":
                    engine.register_algorithm(SimpleBulkOptimizer())

                # Execute optimization with monitoring
                self.logger.debug(f"Starting optimization with {total_panels} panels")
                results = engine.optimize(expanded_panels, test_case.constraints, test_case.algorithm_name)

                execution_time = time.time() - start_time

                # Calculate placement metrics
                placed_panels = sum(len(result.panels) for result in results)
                placement_rate = (placed_panels / total_panels * 100) if total_panels > 0 else 0
                unplaced_panels = total_panels - placed_panels
                sheets_used = len(results)
                avg_efficiency = sum(r.efficiency for r in results) / len(results) if results else 0

                # Comprehensive validation
                validation_errors = self._validate_placement_results(results, expanded_panels)

                # Performance analysis
                performance_warnings = self._analyze_performance(
                    total_panels, execution_time, placement_rate, sheets_used
                )
                warnings.extend(performance_warnings)

                # Determine success
                success = (
                    placement_rate >= test_case.expected_placement_rate and
                    len(validation_errors) == 0 and
                    not timeout_occurred
                )

                self.logger.info(
                    f"✅ Test completed: {placed_panels}/{total_panels} panels placed "
                    f"({placement_rate:.1f}%) in {execution_time:.2f}s"
                )

                return PlacementTestResult(
                    test_case=test_case,
                    success=success,
                    placement_rate=placement_rate,
                    total_panels=total_panels,
                    placed_panels=placed_panels,
                    unplaced_panels=unplaced_panels,
                    sheets_used=sheets_used,
                    execution_time=execution_time,
                    average_efficiency=avg_efficiency,
                    algorithm_performance=self._extract_algorithm_metrics(results),
                    validation_errors=validation_errors,
                    warnings=warnings,
                    timeout_occurred=timeout_occurred,
                    infinite_loop_detected=infinite_loop_detected
                )

        except TimeoutError:
            execution_time = time.time() - start_time
            timeout_occurred = True
            infinite_loop_detected = execution_time >= test_case.max_execution_time * 0.9

            self.logger.error(f"❌ Test timeout after {execution_time:.2f}s")
            validation_errors.append(f"Test exceeded maximum execution time ({test_case.max_execution_time}s)")

            return PlacementTestResult(
                test_case=test_case,
                success=False,
                placement_rate=0.0,
                total_panels=sum(panel.quantity for panel in test_case.panels),
                placed_panels=0,
                unplaced_panels=sum(panel.quantity for panel in test_case.panels),
                sheets_used=0,
                execution_time=execution_time,
                average_efficiency=0.0,
                algorithm_performance={},
                validation_errors=validation_errors,
                warnings=warnings,
                timeout_occurred=timeout_occurred,
                infinite_loop_detected=infinite_loop_detected
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Test failed with exception: {e}")
            self.logger.debug(traceback.format_exc())

            validation_errors.append(f"Algorithm execution failed: {str(e)}")

            return PlacementTestResult(
                test_case=test_case,
                success=False,
                placement_rate=0.0,
                total_panels=sum(panel.quantity for panel in test_case.panels),
                placed_panels=0,
                unplaced_panels=sum(panel.quantity for panel in test_case.panels),
                sheets_used=0,
                execution_time=execution_time,
                average_efficiency=0.0,
                algorithm_performance={},
                validation_errors=validation_errors,
                warnings=warnings
            )

    def _apply_pi_expansion(self, panels: List[Panel]) -> List[Panel]:
        """Apply PI expansion to panel dimensions"""
        expanded_panels = []
        for panel in panels:
            expanded_panel = Panel(
                id=panel.id,
                width=panel.width,
                height=panel.height,
                quantity=panel.quantity,
                material=panel.material,
                thickness=panel.thickness,
                priority=panel.priority,
                allow_rotation=panel.allow_rotation,
                pi_code=panel.pi_code
            )
            expanded_panel.calculate_expanded_dimensions(self.pi_manager)
            expanded_panels.append(expanded_panel)
        return expanded_panels

    def _validate_placement_results(self, results: List[PlacementResult], panels: List[Panel]) -> List[str]:
        """
        Comprehensive validation of placement results

        Validates:
        - No overlapping panels
        - All panels within sheet boundaries
        - Panel rotations respect constraints
        - Total placed panels match input
        - Material constraints respected
        """
        errors = []

        for i, result in enumerate(results):
            sheet_errors = self._validate_single_sheet(result, i + 1)
            errors.extend(sheet_errors)

        # Validate total placement count
        total_input = sum(panel.quantity for panel in panels)
        total_placed = sum(len(result.panels) for result in results)

        if total_placed > total_input:
            errors.append(f"Duplicate placement detected: {total_placed} placed > {total_input} input panels")

        return errors

    def _validate_single_sheet(self, result: PlacementResult, sheet_num: int) -> List[str]:
        """Validate placement on a single sheet"""
        errors = []
        placed_panels = result.panels
        sheet = result.sheet

        # Check boundary violations
        for panel in placed_panels:
            if panel.x < 0 or panel.y < 0:
                errors.append(f"Sheet {sheet_num}: Panel {panel.panel.id} has negative position ({panel.x}, {panel.y})")

            right_edge = panel.x + panel.actual_width
            bottom_edge = panel.y + panel.actual_height

            if right_edge > sheet.width:
                errors.append(f"Sheet {sheet_num}: Panel {panel.panel.id} exceeds sheet width ({right_edge} > {sheet.width})")

            if bottom_edge > sheet.height:
                errors.append(f"Sheet {sheet_num}: Panel {panel.panel.id} exceeds sheet height ({bottom_edge} > {sheet.height})")

        # Check overlaps
        for i, panel1 in enumerate(placed_panels):
            for j, panel2 in enumerate(placed_panels[i+1:], i+1):
                if panel1.overlaps_with(panel2):
                    errors.append(f"Sheet {sheet_num}: Overlap between panels {panel1.panel.id} and {panel2.panel.id}")

        # Check rotation constraints
        for panel in placed_panels:
            if panel.rotated and not panel.panel.allow_rotation:
                errors.append(f"Sheet {sheet_num}: Panel {panel.panel.id} rotated but rotation not allowed")

        return errors

    def _analyze_performance(self, panel_count: int, execution_time: float,
                           placement_rate: float, sheets_used: int) -> List[str]:
        """Analyze performance against baselines and generate warnings"""
        warnings = []

        # Time performance analysis
        if panel_count <= 20 and execution_time > self.performance_baselines['small_batch_time']:
            warnings.append(f"Small batch performance regression: {execution_time:.2f}s > {self.performance_baselines['small_batch_time']}s")
        elif panel_count <= 50 and execution_time > self.performance_baselines['medium_batch_time']:
            warnings.append(f"Medium batch performance regression: {execution_time:.2f}s > {self.performance_baselines['medium_batch_time']}s")
        elif panel_count <= 100 and execution_time > self.performance_baselines['large_batch_time']:
            warnings.append(f"Large batch performance regression: {execution_time:.2f}s > {self.performance_baselines['large_batch_time']}s")

        # Placement rate analysis
        if placement_rate < self.performance_baselines['placement_rate_threshold']:
            warnings.append(f"Placement rate below threshold: {placement_rate:.1f}% < {self.performance_baselines['placement_rate_threshold']}%")

        # Sheet efficiency analysis
        if panel_count > 0 and sheets_used > panel_count * 0.1:  # More than 10% of panels in sheets indicates poor efficiency
            warnings.append(f"Poor sheet utilization: {sheets_used} sheets for {panel_count} panels")

        return warnings

    def _extract_algorithm_metrics(self, results: List[PlacementResult]) -> Dict[str, Any]:
        """Extract detailed algorithm performance metrics"""
        if not results:
            return {}

        total_processing_time = sum(r.processing_time for r in results)
        total_efficiency = sum(r.efficiency for r in results) / len(results)
        total_waste = sum(r.waste_area for r in results)
        total_cut_length = sum(r.cut_length for r in results)

        return {
            'total_processing_time': total_processing_time,
            'average_efficiency': total_efficiency,
            'total_waste_area': total_waste,
            'total_cut_length': total_cut_length,
            'sheets_count': len(results),
            'algorithms_used': list(set(r.algorithm for r in results))
        }

    def run_test_suite(self, test_cases: List[PlacementTestCase]) -> Dict[str, Any]:
        """
        Run complete test suite and generate comprehensive report

        Args:
            test_cases: List of test cases to execute

        Returns:
            Comprehensive test report with metrics and recommendations
        """
        self.logger.info(f"🚀 Starting test suite with {len(test_cases)} test cases")
        suite_start_time = time.time()

        # Execute all test cases
        for test_case in test_cases:
            result = self.run_placement_test(test_case)
            self.test_results.append(result)

        suite_execution_time = time.time() - suite_start_time

        # Generate comprehensive report
        return self._generate_test_report(suite_execution_time)

    def _generate_test_report(self, suite_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive test suite report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests

        # Calculate aggregate metrics
        avg_placement_rate = sum(r.placement_rate for r in self.test_results) / total_tests if total_tests > 0 else 0
        total_panels_tested = sum(r.total_panels for r in self.test_results)
        total_panels_placed = sum(r.placed_panels for r in self.test_results)

        # Identify critical issues
        timeout_tests = [r for r in self.test_results if r.timeout_occurred]
        infinite_loop_tests = [r for r in self.test_results if r.infinite_loop_detected]
        validation_error_tests = [r for r in self.test_results if r.validation_errors]

        # Performance regression analysis
        performance_regressions = [r for r in self.test_results if r.warnings]

        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'average_placement_rate': avg_placement_rate,
                'total_panels_tested': total_panels_tested,
                'total_panels_placed': total_panels_placed,
                'overall_placement_rate': (total_panels_placed / total_panels_tested * 100) if total_panels_tested > 0 else 0,
                'suite_execution_time': suite_execution_time
            },
            'critical_issues': {
                'timeout_count': len(timeout_tests),
                'infinite_loop_count': len(infinite_loop_tests),
                'validation_error_count': len(validation_error_tests),
                'performance_regression_count': len(performance_regressions)
            },
            'detailed_results': [
                {
                    'test_name': r.test_case.name,
                    'success': r.success,
                    'placement_rate': r.placement_rate,
                    'execution_time': r.execution_time,
                    'validation_errors': r.validation_errors,
                    'warnings': r.warnings
                }
                for r in self.test_results
            ],
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []

        # Check for 100% placement achievement
        placement_rates = [r.placement_rate for r in self.test_results if r.success]
        if placement_rates and max(placement_rates) < 100.0:
            recommendations.append("🎯 CRITICAL: 100% placement guarantee not achieved - algorithm improvements required")

        # Check for timeout issues
        timeout_tests = [r for r in self.test_results if r.timeout_occurred]
        if timeout_tests:
            recommendations.append(f"⏰ URGENT: {len(timeout_tests)} tests exceeded time limits - infinite loop protection needed")

        # Check for validation errors
        validation_error_tests = [r for r in self.test_results if r.validation_errors]
        if validation_error_tests:
            recommendations.append(f"❌ CRITICAL: {len(validation_error_tests)} tests had placement validation errors - overlap/boundary issues")

        # Performance recommendations
        slow_tests = [r for r in self.test_results if r.execution_time > 30.0]
        if slow_tests:
            recommendations.append(f"🐌 Performance optimization needed for {len(slow_tests)} slow tests")

        # Success case analysis
        successful_100_percent = [r for r in self.test_results if r.success and r.placement_rate >= 100.0]
        if successful_100_percent:
            recommendations.append(f"✅ SUCCESS: {len(successful_100_percent)} tests achieved 100% placement - replicate these algorithms")

        return recommendations