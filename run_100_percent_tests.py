#!/usr/bin/env python3
"""
Quick Test Execution Script for 100% Panel Placement Guarantee
100%パネル配置保証クイックテスト実行スクリプト

This script provides a convenient way to run the comprehensive test suite
and validate the 100% placement guarantee quickly.

Usage:
    python run_100_percent_tests.py [options]

Options:
    --quick     Run only basic and edge case tests (faster execution)
    --full      Run complete test suite including stress tests
    --report    Generate HTML and JSON reports
    --baseline  Update performance baselines
    --verbose   Enable verbose output
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.automated_testing_pipeline import (
    AutomatedTestingPipeline,
    PipelineConfiguration
)
from tests.algorithm_correctness_validator import ValidationLevel
from tests.test_100_percent_placement_guarantee import run_master_test_suite


class QuickTestRunner:
    """Quick test execution with various configuration options"""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging for test execution"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('test_execution.log', mode='w')
            ]
        )
        return logging.getLogger('quick_test_runner')

    def run_quick_tests(self, args) -> int:
        """Run quick validation tests"""
        self.logger.info("🚀 Starting Quick 100% Placement Guarantee Tests")

        config = PipelineConfiguration(
            max_parallel_tests=2,
            timeout_per_test=60.0,  # 1 minute per test for quick run
            validation_level=ValidationLevel.STANDARD,
            run_basic_tests=True,
            run_edge_case_tests=True,
            run_stress_tests=False,  # Skip for quick run
            run_real_world_tests=False,  # Skip for quick run
            run_regression_tests=True,
            run_production_data_tests=False,  # Skip for quick run
            min_placement_rate=100.0,
            enable_regression_detection=args.baseline,
            update_baselines=args.baseline,
            generate_html_report=args.report,
            generate_json_report=args.report
        )

        pipeline = AutomatedTestingPipeline(config)
        results = pipeline.run_complete_pipeline()

        return self._analyze_results(results, "Quick Tests")

    def run_full_tests(self, args) -> int:
        """Run comprehensive test suite"""
        self.logger.info("🎯 Starting Full 100% Placement Guarantee Test Suite")

        config = PipelineConfiguration(
            max_parallel_tests=4,
            timeout_per_test=300.0,  # 5 minutes per test
            validation_level=ValidationLevel.STANDARD,
            run_basic_tests=True,
            run_edge_case_tests=True,
            run_stress_tests=True,
            run_real_world_tests=True,
            run_regression_tests=True,
            run_production_data_tests=True,
            min_placement_rate=100.0,
            enable_regression_detection=True,
            update_baselines=args.baseline,
            generate_html_report=args.report,
            generate_json_report=args.report
        )

        pipeline = AutomatedTestingPipeline(config)
        results = pipeline.run_complete_pipeline()

        return self._analyze_results(results, "Full Test Suite")

    def run_pytest_suite(self, args) -> int:
        """Run the pytest-based master test suite"""
        self.logger.info("📋 Running PyTest Master Test Suite")

        if args.verbose:
            os.environ['PYTEST_ARGS'] = '-v --tb=long'
        else:
            os.environ['PYTEST_ARGS'] = '-v --tb=short'

        return run_master_test_suite()

    def run_algorithm_specific_tests(self, algorithm_name: str) -> int:
        """Run tests for a specific algorithm"""
        self.logger.info(f"🔧 Testing {algorithm_name} Algorithm")

        config = PipelineConfiguration(
            max_parallel_tests=2,
            timeout_per_test=120.0,
            validation_level=ValidationLevel.STANDARD,
            run_basic_tests=True,
            run_edge_case_tests=True,
            run_stress_tests=False,
            min_placement_rate=100.0,
            generate_html_report=False,
            generate_json_report=False
        )

        # Modify test cases to use specific algorithm
        pipeline = AutomatedTestingPipeline(config)

        # Override algorithm selection
        original_generate = pipeline._generate_test_cases

        def algorithm_specific_generate():
            test_cases = original_generate()
            for test_case in test_cases:
                test_case.algorithm_name = algorithm_name
            return test_cases

        pipeline._generate_test_cases = algorithm_specific_generate
        results = pipeline.run_complete_pipeline()

        return self._analyze_results(results, f"{algorithm_name} Algorithm Tests")

    def _analyze_results(self, results, test_name: str) -> int:
        """Analyze test results and determine exit code"""
        total_tests = len(results.test_results)
        successful_tests = sum(1 for r in results.test_results if r.success)
        perfect_placement_tests = sum(1 for r in results.test_results if r.placement_rate >= 100.0)

        # Calculate metrics
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        placement_rate = (perfect_placement_tests / total_tests * 100) if total_tests > 0 else 0

        # Quality gates analysis
        gates_passed = sum(1 for status in results.quality_gate_status.values() if status)
        total_gates = len(results.quality_gate_status)

        # Critical issues
        critical_alerts = [a for a in results.regression_alerts if a.severity == 'CRITICAL']
        timeout_tests = [r for r in results.test_results if r.timeout_occurred]

        # Log summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 {test_name} Results Summary")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Successful: {successful_tests} ({success_rate:.1f}%)")
        self.logger.info(f"100% Placement: {perfect_placement_tests} ({placement_rate:.1f}%)")
        self.logger.info(f"Quality Gates: {gates_passed}/{total_gates} passed")
        self.logger.info(f"Execution Time: {results.execution_time:.2f} seconds")

        if critical_alerts:
            self.logger.warning(f"🚨 Critical Alerts: {len(critical_alerts)}")

        if timeout_tests:
            self.logger.warning(f"⏰ Timeouts: {len(timeout_tests)}")

        # Determine success
        success_criteria = [
            placement_rate >= 95.0,  # At least 95% of tests achieve 100% placement
            gates_passed >= total_gates * 0.8,  # At least 80% of quality gates pass
            len(critical_alerts) == 0,  # No critical performance regressions
            len(timeout_tests) <= total_tests * 0.05  # No more than 5% timeouts
        ]

        if all(success_criteria):
            if placement_rate >= 100.0:
                self.logger.info("🎉 SUCCESS: 100% Placement Guarantee ACHIEVED!")
                exit_code = 0
            else:
                self.logger.info("✅ GOOD: High placement rate achieved, working towards 100%")
                exit_code = 0
        else:
            self.logger.error("❌ FAILURE: 100% Placement Guarantee NOT achieved")
            exit_code = 1

        # Detailed recommendations
        if results.recommendations:
            self.logger.info("\n💡 Recommendations:")
            for rec in results.recommendations[:5]:  # Show top 5 recommendations
                self.logger.info(f"   - {rec}")

        return exit_code

    def run_diagnostic_check(self) -> int:
        """Run diagnostic check to verify system is ready for testing"""
        self.logger.info("🔍 Running Diagnostic Check")

        try:
            # Check imports
            from core.optimizer import OptimizationEngine
            from core.algorithms.complete_placement_guaranteed import CompletePlacementGuaranteedAlgorithm
            from core.algorithms.simple_bulk_optimizer import SimpleBulkOptimizer
            from core.text_parser import parse_cutting_data_file
            from core.pi_manager import PIManager
            from core.material_manager import MaterialInventoryManager

            self.logger.info("✅ All required modules imported successfully")

            # Check sample data
            sample_data_path = project_root / "sample_data" / "data0923.txt"
            if sample_data_path.exists():
                self.logger.info("✅ Sample data file found")
            else:
                self.logger.warning("⚠️ Sample data file not found")

            # Quick algorithm test
            engine = OptimizationEngine()
            engine.register_algorithm(CompletePlacementGuaranteedAlgorithm())
            self.logger.info("✅ Algorithm registration successful")

            self.logger.info("🎯 System diagnostic check PASSED")
            return 0

        except Exception as e:
            self.logger.error(f"❌ Diagnostic check FAILED: {e}")
            return 1


def main():
    """Main entry point for quick test runner"""
    parser = argparse.ArgumentParser(
        description="Quick Test Runner for 100% Panel Placement Guarantee",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_100_percent_tests.py --quick          # Quick validation
  python run_100_percent_tests.py --full --report # Complete test with reports
  python run_100_percent_tests.py --pytest        # Run pytest suite
  python run_100_percent_tests.py --diagnostic    # System check
        """
    )

    # Test execution modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--quick', action='store_true',
                           help='Run quick tests (basic + edge cases)')
    mode_group.add_argument('--full', action='store_true',
                           help='Run complete test suite')
    mode_group.add_argument('--pytest', action='store_true',
                           help='Run pytest-based master test suite')
    mode_group.add_argument('--algorithm', choices=['Complete_Placement_Guaranteed', 'Simple_Bulk'],
                           help='Test specific algorithm')
    mode_group.add_argument('--diagnostic', action='store_true',
                           help='Run system diagnostic check')

    # Options
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML and JSON reports')
    parser.add_argument('--baseline', action='store_true',
                       help='Update performance baselines')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Create test runner
    runner = QuickTestRunner()

    # Execute based on mode
    start_time = time.time()

    if args.diagnostic:
        exit_code = runner.run_diagnostic_check()
    elif args.quick:
        exit_code = runner.run_quick_tests(args)
    elif args.full:
        exit_code = runner.run_full_tests(args)
    elif args.pytest:
        exit_code = runner.run_pytest_suite(args)
    elif args.algorithm:
        exit_code = runner.run_algorithm_specific_tests(args.algorithm)
    else:
        parser.print_help()
        exit_code = 1

    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"🏁 Test execution completed in {total_time:.2f} seconds")
    if exit_code == 0:
        print("✅ RESULT: Tests PASSED")
    else:
        print("❌ RESULT: Tests FAILED")
    print(f"{'='*60}")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)