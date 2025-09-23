"""
Automated Testing Pipeline for 100% Panel Placement Guarantee
100%配置保証自動テストパイプライン

Comprehensive automated testing system that validates:
- 100% placement guarantee across all test scenarios
- Algorithm correctness and mathematical validation
- Performance regression detection and monitoring
- Edge case coverage and stress testing
- Continuous integration and quality gates
"""

import sys
import os
import time
import json
import asyncio
import subprocess
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.placement_guarantee_framework import PlacementGuaranteeFramework, PlacementTestCase, PlacementTestResult
from tests.test_case_generators import TestCaseGenerator, ProductionDataTestGenerator, generate_all_test_cases
from tests.algorithm_correctness_validator import AlgorithmCorrectnessValidator, ValidationLevel
from tests.performance_regression_detector import PerformanceRegressionDetector, RegressionAlert
from core.models import OptimizationConstraints


@dataclass
class PipelineConfiguration:
    """Configuration for automated testing pipeline"""
    # Test execution settings
    max_parallel_tests: int = 4
    timeout_per_test: float = 300.0  # 5 minutes
    validation_level: ValidationLevel = ValidationLevel.STANDARD

    # Test categories to run
    run_basic_tests: bool = True
    run_edge_case_tests: bool = True
    run_stress_tests: bool = True
    run_real_world_tests: bool = True
    run_regression_tests: bool = True
    run_production_data_tests: bool = True

    # Quality gates
    min_placement_rate: float = 100.0        # Must achieve 100% placement
    max_execution_time: float = 300.0        # 5 minutes max per test
    max_memory_usage_mb: float = 1000.0      # 1GB memory limit
    max_validation_errors: int = 0           # Zero validation errors allowed

    # Regression detection
    enable_regression_detection: bool = True
    update_baselines: bool = False
    regression_alert_threshold: float = 20.0  # 20% performance degradation

    # Reporting
    generate_html_report: bool = True
    generate_json_report: bool = True
    output_directory: Path = Path("test_reports")


@dataclass
class PipelineResults:
    """Complete results from automated testing pipeline"""
    configuration: PipelineConfiguration
    execution_summary: Dict[str, Any]
    test_results: List[PlacementTestResult]
    validation_reports: List[Dict[str, Any]]
    regression_alerts: List[RegressionAlert]
    quality_gate_status: Dict[str, bool]
    recommendations: List[str]
    execution_time: float
    timestamp: datetime


class QualityGateValidator:
    """Validates test results against quality gates"""

    def __init__(self, config: PipelineConfiguration):
        self.config = config
        self.logger = logging.getLogger('quality_gates')

    def validate_quality_gates(self, test_results: List[PlacementTestResult],
                             regression_alerts: List[RegressionAlert]) -> Dict[str, bool]:
        """
        Validate all test results against quality gates

        Returns:
            Dictionary indicating pass/fail status for each quality gate
        """
        gates = {}

        # Gate 1: 100% Placement Rate Achievement
        gates['placement_rate'] = self._validate_placement_rate_gate(test_results)

        # Gate 2: Zero Validation Errors
        gates['validation_errors'] = self._validate_error_gate(test_results)

        # Gate 3: Execution Time Limits
        gates['execution_time'] = self._validate_time_gate(test_results)

        # Gate 4: Memory Usage Limits
        gates['memory_usage'] = self._validate_memory_gate(test_results)

        # Gate 5: No Critical Regressions
        gates['regression_control'] = self._validate_regression_gate(regression_alerts)

        # Gate 6: Algorithm Correctness
        gates['algorithm_correctness'] = self._validate_correctness_gate(test_results)

        return gates

    def _validate_placement_rate_gate(self, test_results: List[PlacementTestResult]) -> bool:
        """Validate that all tests achieve required placement rate"""
        failed_tests = []
        for result in test_results:
            if result.placement_rate < self.config.min_placement_rate:
                failed_tests.append(f"{result.test_case.name}: {result.placement_rate:.1f}%")

        if failed_tests:
            self.logger.error(f"❌ Placement Rate Gate FAILED: {len(failed_tests)} tests below {self.config.min_placement_rate}%")
            for test in failed_tests[:5]:  # Show first 5 failures
                self.logger.error(f"  - {test}")
            return False

        self.logger.info(f"✅ Placement Rate Gate PASSED: All tests achieved ≥{self.config.min_placement_rate}%")
        return True

    def _validate_error_gate(self, test_results: List[PlacementTestResult]) -> bool:
        """Validate that no tests have validation errors"""
        error_tests = []
        for result in test_results:
            if result.validation_errors:
                error_tests.append(f"{result.test_case.name}: {len(result.validation_errors)} errors")

        if error_tests:
            self.logger.error(f"❌ Validation Error Gate FAILED: {len(error_tests)} tests with errors")
            for test in error_tests[:5]:
                self.logger.error(f"  - {test}")
            return False

        self.logger.info("✅ Validation Error Gate PASSED: No validation errors found")
        return True

    def _validate_time_gate(self, test_results: List[PlacementTestResult]) -> bool:
        """Validate execution time limits"""
        slow_tests = []
        for result in test_results:
            if result.execution_time > self.config.max_execution_time:
                slow_tests.append(f"{result.test_case.name}: {result.execution_time:.2f}s")

        if slow_tests:
            self.logger.error(f"❌ Execution Time Gate FAILED: {len(slow_tests)} tests exceeded {self.config.max_execution_time}s")
            for test in slow_tests[:5]:
                self.logger.error(f"  - {test}")
            return False

        self.logger.info(f"✅ Execution Time Gate PASSED: All tests completed within {self.config.max_execution_time}s")
        return True

    def _validate_memory_gate(self, test_results: List[PlacementTestResult]) -> bool:
        """Validate memory usage limits"""
        memory_tests = []
        for result in test_results:
            if result.memory_usage_mb > self.config.max_memory_usage_mb:
                memory_tests.append(f"{result.test_case.name}: {result.memory_usage_mb:.1f}MB")

        if memory_tests:
            self.logger.error(f"❌ Memory Usage Gate FAILED: {len(memory_tests)} tests exceeded {self.config.max_memory_usage_mb}MB")
            for test in memory_tests[:5]:
                self.logger.error(f"  - {test}")
            return False

        self.logger.info(f"✅ Memory Usage Gate PASSED: All tests used ≤{self.config.max_memory_usage_mb}MB")
        return True

    def _validate_regression_gate(self, regression_alerts: List[RegressionAlert]) -> bool:
        """Validate no critical performance regressions"""
        critical_alerts = [a for a in regression_alerts if a.severity == 'CRITICAL']

        if critical_alerts:
            self.logger.error(f"❌ Regression Gate FAILED: {len(critical_alerts)} critical regressions detected")
            for alert in critical_alerts[:3]:
                self.logger.error(f"  - {alert.test_name}: {alert.metric_name} regression {alert.regression_percent:.1f}%")
            return False

        self.logger.info("✅ Regression Gate PASSED: No critical performance regressions")
        return True

    def _validate_correctness_gate(self, test_results: List[PlacementTestResult]) -> bool:
        """Validate algorithm correctness across all tests"""
        failed_tests = []
        for result in test_results:
            if not result.success:
                failed_tests.append(result.test_case.name)

        if failed_tests:
            self.logger.error(f"❌ Algorithm Correctness Gate FAILED: {len(failed_tests)} tests failed")
            for test in failed_tests[:5]:
                self.logger.error(f"  - {test}")
            return False

        self.logger.info("✅ Algorithm Correctness Gate PASSED: All tests completed successfully")
        return True


class TestReportGenerator:
    """Generates comprehensive test reports in multiple formats"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, pipeline_results: PipelineResults) -> Path:
        """Generate comprehensive HTML test report"""
        html_content = self._build_html_report(pipeline_results)
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_file

    def generate_json_report(self, pipeline_results: PipelineResults) -> Path:
        """Generate machine-readable JSON report"""
        report_data = {
            'timestamp': pipeline_results.timestamp.isoformat(),
            'configuration': asdict(pipeline_results.configuration),
            'execution_summary': pipeline_results.execution_summary,
            'test_results': [
                {
                    'test_name': result.test_case.name,
                    'category': result.test_case.category,
                    'success': result.success,
                    'placement_rate': result.placement_rate,
                    'execution_time': result.execution_time,
                    'validation_errors': result.validation_errors,
                    'warnings': result.warnings,
                    'timeout_occurred': result.timeout_occurred
                }
                for result in pipeline_results.test_results
            ],
            'regression_alerts': [asdict(alert) for alert in pipeline_results.regression_alerts],
            'quality_gate_status': pipeline_results.quality_gate_status,
            'recommendations': pipeline_results.recommendations
        }

        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_file

    def _build_html_report(self, results: PipelineResults) -> str:
        """Build comprehensive HTML report"""
        # Calculate summary statistics
        total_tests = len(results.test_results)
        passed_tests = sum(1 for r in results.test_results if r.success)
        failed_tests = total_tests - passed_tests

        avg_placement_rate = sum(r.placement_rate for r in results.test_results) / total_tests if total_tests > 0 else 0
        total_panels = sum(r.total_panels for r in results.test_results)
        total_placed = sum(r.placed_panels for r in results.test_results)

        # Quality gate status summary
        gates_passed = sum(1 for status in results.quality_gate_status.values() if status)
        total_gates = len(results.quality_gate_status)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steel Cutting Optimization - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .success {{ border-left-color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }}
        .danger {{ border-left-color: #dc3545; }}
        .quality-gates {{ margin: 20px 0; }}
        .gate {{ display: flex; justify-content: space-between; align-items: center; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .gate.passed {{ background: #d4edda; color: #155724; }}
        .gate.failed {{ background: #f8d7da; color: #721c24; }}
        .test-results {{ margin: 20px 0; }}
        .test-result {{ margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid; }}
        .test-result.success {{ background: #d4edda; border-left-color: #28a745; }}
        .test-result.failure {{ background: #f8d7da; border-left-color: #dc3545; }}
        .recommendations {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 Steel Cutting Optimization - Test Report</h1>
            <p class="timestamp">Generated: {results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Execution Time: {results.execution_time:.2f} seconds</p>
        </div>

        <div class="summary">
            <div class="metric {'success' if passed_tests == total_tests else 'danger'}">
                <div class="metric-value">{passed_tests}/{total_tests}</div>
                <div class="metric-label">Tests Passed</div>
            </div>
            <div class="metric {'success' if avg_placement_rate >= 100 else 'warning'}">
                <div class="metric-value">{avg_placement_rate:.1f}%</div>
                <div class="metric-label">Avg Placement Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_placed:,}</div>
                <div class="metric-label">Panels Placed</div>
            </div>
            <div class="metric {'success' if gates_passed == total_gates else 'danger'}">
                <div class="metric-value">{gates_passed}/{total_gates}</div>
                <div class="metric-label">Quality Gates</div>
            </div>
        </div>

        <div class="quality-gates">
            <h2>🚪 Quality Gates Status</h2>
        """

        for gate_name, status in results.quality_gate_status.items():
            status_class = "passed" if status else "failed"
            status_icon = "✅" if status else "❌"
            html += f"""
            <div class="gate {status_class}">
                <span>{status_icon} {gate_name.replace('_', ' ').title()}</span>
                <span>{'PASSED' if status else 'FAILED'}</span>
            </div>
            """

        html += """
        </div>

        <div class="test-results">
            <h2>📋 Test Results Summary</h2>
        """

        # Group results by category
        categories = {}
        for result in results.test_results:
            category = result.test_case.category
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, category_results in categories.items():
            category_passed = sum(1 for r in category_results if r.success)
            category_total = len(category_results)

            html += f"""
            <h3>{category.replace('_', ' ').title()} ({category_passed}/{category_total})</h3>
            """

            for result in category_results:
                result_class = "success" if result.success else "failure"
                result_icon = "✅" if result.success else "❌"

                html += f"""
                <div class="test-result {result_class}">
                    <strong>{result_icon} {result.test_case.name}</strong>
                    <br>
                    Placement Rate: {result.placement_rate:.1f}% |
                    Execution Time: {result.execution_time:.2f}s |
                    Panels: {result.placed_panels}/{result.total_panels}
                """

                if result.validation_errors:
                    html += f"<br><span style='color: #dc3545;'>Validation Errors: {len(result.validation_errors)}</span>"

                if result.warnings:
                    html += f"<br><span style='color: #ffc107;'>Warnings: {len(result.warnings)}</span>"

                html += "</div>"

        if results.recommendations:
            html += """
            <div class="recommendations">
                <h2>💡 Recommendations</h2>
                <ul>
            """
            for rec in results.recommendations:
                html += f"<li>{rec}</li>"

            html += """
                </ul>
            </div>
            """

        html += """
        </div>
    </div>
</body>
</html>
        """

        return html


class AutomatedTestingPipeline:
    """
    Comprehensive automated testing pipeline for 100% placement guarantee

    Features:
    - Parallel test execution with timeout protection
    - Comprehensive validation and regression detection
    - Quality gate enforcement
    - Detailed reporting and recommendations
    """

    def __init__(self, config: PipelineConfiguration = None):
        """Initialize pipeline with configuration"""
        self.config = config or PipelineConfiguration()
        self.logger = self._setup_logging()

        # Initialize components
        self.placement_framework = PlacementGuaranteeFramework(timeout_seconds=self.config.timeout_per_test)
        self.validator = AlgorithmCorrectnessValidator(validation_level=self.config.validation_level)
        self.regression_detector = PerformanceRegressionDetector()
        self.quality_gate_validator = QualityGateValidator(self.config)
        self.report_generator = TestReportGenerator(self.config.output_directory)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for pipeline"""
        logger = logging.getLogger('automated_pipeline')
        logger.setLevel(logging.INFO)

        # Console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = self.config.output_directory / "pipeline.log"
            self.config.output_directory.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def run_complete_pipeline(self) -> PipelineResults:
        """
        Execute complete automated testing pipeline

        Returns:
            PipelineResults with comprehensive test outcomes and analysis
        """
        self.logger.info("🚀 Starting Automated Testing Pipeline for 100% Placement Guarantee")
        pipeline_start_time = time.time()

        try:
            # Step 1: Generate test cases
            test_cases = self._generate_test_cases()
            self.logger.info(f"📋 Generated {len(test_cases)} test cases")

            # Step 2: Execute tests with parallel processing
            test_results = self._execute_tests_parallel(test_cases)
            self.logger.info(f"✅ Completed {len(test_results)} tests")

            # Step 3: Validate algorithm correctness
            validation_reports = self._validate_algorithm_correctness(test_results)
            self.logger.info(f"🔍 Generated {len(validation_reports)} validation reports")

            # Step 4: Detect performance regressions
            regression_alerts = self._detect_performance_regressions(test_results)
            self.logger.info(f"📊 Generated {len(regression_alerts)} regression alerts")

            # Step 5: Validate quality gates
            quality_gate_status = self.quality_gate_validator.validate_quality_gates(test_results, regression_alerts)
            gates_passed = sum(1 for status in quality_gate_status.values() if status)
            self.logger.info(f"🚪 Quality Gates: {gates_passed}/{len(quality_gate_status)} passed")

            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(test_results, regression_alerts, quality_gate_status)

            # Step 7: Prepare results
            execution_time = time.time() - pipeline_start_time
            execution_summary = self._generate_execution_summary(test_results, execution_time)

            pipeline_results = PipelineResults(
                configuration=self.config,
                execution_summary=execution_summary,
                test_results=test_results,
                validation_reports=validation_reports,
                regression_alerts=regression_alerts,
                quality_gate_status=quality_gate_status,
                recommendations=recommendations,
                execution_time=execution_time,
                timestamp=datetime.now()
            )

            # Step 8: Generate reports
            if self.config.generate_html_report:
                html_report = self.report_generator.generate_html_report(pipeline_results)
                self.logger.info(f"📄 HTML report generated: {html_report}")

            if self.config.generate_json_report:
                json_report = self.report_generator.generate_json_report(pipeline_results)
                self.logger.info(f"📊 JSON report generated: {json_report}")

            # Step 9: Update performance baselines if requested
            if self.config.update_baselines:
                self._update_performance_baselines(test_results)

            # Final pipeline status
            pipeline_success = all(quality_gate_status.values())
            self.logger.info(f"🏁 Pipeline completed in {execution_time:.2f}s - {'SUCCESS' if pipeline_success else 'FAILED'}")

            return pipeline_results

        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def _generate_test_cases(self) -> List[PlacementTestCase]:
        """Generate comprehensive test case suite based on configuration"""
        generator = TestCaseGenerator()
        production_generator = ProductionDataTestGenerator()
        all_test_cases = []

        if self.config.run_basic_tests:
            all_test_cases.extend(generator.generate_simple_test_cases())

        if self.config.run_edge_case_tests:
            all_test_cases.extend(generator.generate_edge_case_tests())

        if self.config.run_stress_tests:
            all_test_cases.extend(generator.generate_stress_tests())

        if self.config.run_real_world_tests:
            all_test_cases.extend(generator.generate_real_world_tests())

        if self.config.run_regression_tests:
            all_test_cases.extend(generator.generate_regression_tests())

        if self.config.run_production_data_tests:
            all_test_cases.extend(production_generator.generate_production_tests())

        return all_test_cases

    def _execute_tests_parallel(self, test_cases: List[PlacementTestCase]) -> List[PlacementTestResult]:
        """Execute tests in parallel with proper resource management"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_tests) as executor:
            # Submit all test cases
            future_to_test = {
                executor.submit(self.placement_framework.run_placement_test, test_case): test_case
                for test_case in test_cases
            }

            # Collect results as they complete
            for future in future_to_test:
                test_case = future_to_test[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_test)
                    results.append(result)

                    status_icon = "✅" if result.success else "❌"
                    self.logger.info(f"{status_icon} {test_case.name}: {result.placement_rate:.1f}% in {result.execution_time:.2f}s")

                except Exception as e:
                    self.logger.error(f"❌ Test {test_case.name} failed: {e}")
                    # Create a failed result for tracking
                    failed_result = PlacementTestResult(
                        test_case=test_case,
                        success=False,
                        placement_rate=0.0,
                        total_panels=sum(p.quantity for p in test_case.panels),
                        placed_panels=0,
                        unplaced_panels=sum(p.quantity for p in test_case.panels),
                        sheets_used=0,
                        execution_time=self.config.timeout_per_test,
                        average_efficiency=0.0,
                        algorithm_performance={},
                        validation_errors=[f"Test execution failed: {str(e)}"],
                        warnings=[],
                        timeout_occurred=True
                    )
                    results.append(failed_result)

        return results

    def _validate_algorithm_correctness(self, test_results: List[PlacementTestResult]) -> List[Dict[str, Any]]:
        """Validate algorithm correctness for all test results"""
        validation_reports = []

        for result in test_results:
            if result.success and hasattr(result, 'placement_results'):
                # Validate the placement results using the correctness validator
                # Note: This assumes the framework stores placement results
                try:
                    validation_report = self.validator.validate_placement_results(
                        results=getattr(result, 'placement_results', []),
                        input_panels=result.test_case.panels,
                        constraints=result.test_case.constraints
                    )
                    validation_reports.append({
                        'test_name': result.test_case.name,
                        'validation_summary': validation_report.summary,
                        'error_count': validation_report.error_count,
                        'warning_count': validation_report.warning_count,
                        'has_errors': validation_report.has_errors
                    })
                except Exception as e:
                    self.logger.warning(f"Validation failed for {result.test_case.name}: {e}")

        return validation_reports

    def _detect_performance_regressions(self, test_results: List[PlacementTestResult]) -> List[RegressionAlert]:
        """Detect performance regressions across all test results"""
        all_alerts = []

        for result in test_results:
            if result.success:
                try:
                    # Create performance metrics from test result
                    performance_metrics = self.regression_detector.PerformanceMetrics(
                        execution_time=result.execution_time,
                        memory_usage_mb=result.memory_usage_mb,
                        cpu_usage_percent=0.0,  # Not tracked in current framework
                        placement_rate=result.placement_rate,
                        efficiency=result.average_efficiency,
                        panels_per_second=result.total_panels / result.execution_time if result.execution_time > 0 else 0,
                        sheets_used=result.sheets_used
                    )

                    # Detect regressions
                    alerts = self.regression_detector.detect_regressions(
                        test_name=result.test_case.name,
                        algorithm=result.test_case.algorithm_name,
                        current_metrics=performance_metrics
                    )
                    all_alerts.extend(alerts)

                    # Establish or update baseline if configured
                    if self.config.update_baselines:
                        self.regression_detector.establish_baseline(
                            test_name=result.test_case.name,
                            algorithm=result.test_case.algorithm_name,
                            panel_count=result.total_panels,
                            performance_metrics=performance_metrics
                        )

                except Exception as e:
                    self.logger.warning(f"Regression detection failed for {result.test_case.name}: {e}")

        return all_alerts

    def _generate_recommendations(self, test_results: List[PlacementTestResult],
                                regression_alerts: List[RegressionAlert],
                                quality_gate_status: Dict[str, bool]) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []

        # Analyze overall results
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.success)
        avg_placement_rate = sum(r.placement_rate for r in test_results) / total_tests if total_tests > 0 else 0

        # 100% placement analysis
        perfect_placement_tests = sum(1 for r in test_results if r.placement_rate >= 100.0)
        if perfect_placement_tests < total_tests:
            recommendations.append(f"🎯 CRITICAL: Only {perfect_placement_tests}/{total_tests} tests achieved 100% placement - algorithm improvements needed")

        # Quality gate failures
        failed_gates = [gate for gate, status in quality_gate_status.items() if not status]
        if failed_gates:
            recommendations.append(f"🚪 QUALITY GATES: {len(failed_gates)} gates failed - {', '.join(failed_gates)}")

        # Performance issues
        critical_alerts = [a for a in regression_alerts if a.severity == 'CRITICAL']
        if critical_alerts:
            recommendations.append(f"⚡ PERFORMANCE: {len(critical_alerts)} critical regressions detected - immediate optimization required")

        # Timeout issues
        timeout_tests = [r for r in test_results if r.timeout_occurred]
        if timeout_tests:
            recommendations.append(f"⏰ TIMEOUT: {len(timeout_tests)} tests timed out - infinite loop protection needed")

        # Algorithm-specific recommendations
        algorithm_performance = {}
        for result in test_results:
            alg = result.test_case.algorithm_name
            if alg not in algorithm_performance:
                algorithm_performance[alg] = []
            algorithm_performance[alg].append(result.placement_rate)

        for algorithm, rates in algorithm_performance.items():
            avg_rate = sum(rates) / len(rates)
            if avg_rate < 100.0:
                recommendations.append(f"🔧 ALGORITHM: {algorithm} averaging {avg_rate:.1f}% placement - requires optimization")

        # Success cases
        if avg_placement_rate >= 100.0 and not failed_gates:
            recommendations.append("🎉 SUCCESS: 100% placement guarantee achieved across all test scenarios!")

        return recommendations

    def _generate_execution_summary(self, test_results: List[PlacementTestResult], execution_time: float) -> Dict[str, Any]:
        """Generate execution summary statistics"""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.success)
        total_panels = sum(r.total_panels for r in test_results)
        total_placed = sum(r.placed_panels for r in test_results)

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_panels_tested': total_panels,
            'total_panels_placed': total_placed,
            'overall_placement_rate': (total_placed / total_panels * 100) if total_panels > 0 else 0,
            'average_execution_time': sum(r.execution_time for r in test_results) / total_tests if total_tests > 0 else 0,
            'total_execution_time': execution_time,
            'tests_by_category': self._summarize_by_category(test_results)
        }

    def _summarize_by_category(self, test_results: List[PlacementTestResult]) -> Dict[str, Dict[str, Any]]:
        """Summarize test results by category"""
        categories = {}
        for result in test_results:
            category = result.test_case.category
            if category not in categories:
                categories[category] = {'total': 0, 'passed': 0, 'avg_placement_rate': 0.0}

            categories[category]['total'] += 1
            if result.success:
                categories[category]['passed'] += 1

        # Calculate averages
        for category in categories:
            category_results = [r for r in test_results if r.test_case.category == category]
            categories[category]['avg_placement_rate'] = sum(r.placement_rate for r in category_results) / len(category_results)

        return categories

    def _update_performance_baselines(self, test_results: List[PlacementTestResult]):
        """Update performance baselines with current results"""
        self.regression_detector.save_baselines()
        self.logger.info("📊 Performance baselines updated")


def main():
    """Main entry point for automated testing pipeline"""
    # Configure pipeline
    config = PipelineConfiguration(
        max_parallel_tests=4,
        timeout_per_test=300.0,
        validation_level=ValidationLevel.STANDARD,
        min_placement_rate=100.0,
        enable_regression_detection=True,
        generate_html_report=True,
        generate_json_report=True
    )

    # Run pipeline
    pipeline = AutomatedTestingPipeline(config)
    results = pipeline.run_complete_pipeline()

    # Exit with appropriate code
    all_gates_passed = all(results.quality_gate_status.values())
    exit_code = 0 if all_gates_passed else 1

    print(f"\n🏁 Pipeline completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)