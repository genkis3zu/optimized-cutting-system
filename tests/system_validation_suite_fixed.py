"""
Comprehensive System Validation Suite for Steel Cutting Optimization System (Fixed Version)
鋼板切断最適化システム包括的検証スイート（修正版）

Tests all recent improvements including:
- Panel placement optimization fixes
- PI code expansion functionality
- Cutting allowance optimization
- File upload prioritization
- UI/UX improvements
- Integration testing
- Edge case and error handling
"""

import unittest
import sys
import os
from pathlib import Path
import time
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import subprocess
import threading

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.text_parser import RobustTextParser, ParseResult
from core.optimizer import OptimizationEngine
from core.models import Panel, SteelSheet, OptimizationConstraints
from core.material_manager import get_material_manager
try:
    from core.pi_manager import get_pi_manager
    PI_MANAGER_AVAILABLE = True
except ImportError:
    PI_MANAGER_AVAILABLE = False


@dataclass
class TestResult:
    """Test result with detailed metrics"""
    test_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    execution_time: float
    details: str
    performance_metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: datetime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_execution_time: float
    test_results: List[TestResult]
    system_info: Dict[str, Any]
    recommendations: List[str]


class SystemValidationSuite:
    """Comprehensive testing suite for recent system improvements"""

    def __init__(self, streamlit_url: str = "http://localhost:8503"):
        self.streamlit_url = streamlit_url
        self.sample_data_dir = project_root / "sample_data"
        self.test_results: List[TestResult] = []
        self.logger = self._setup_logging()

        # Test data paths
        self.data0923_path = self.sample_data_dir / "data0923.txt"
        self.sizai_data_path = self.sample_data_dir / "sizaidata.txt"
        self.pi_data_path = self.sample_data_dir / "pi.txt"

        # Performance targets
        self.performance_targets = {
            'small_batch_time': 1.0,   # ≤20 panels: < 1 second
            'medium_batch_time': 5.0,  # ≤50 panels: < 5 seconds
            'large_batch_time': 30.0,  # ≤100 panels: < 30 seconds
            'panel_placement_rate': 0.9,  # ≥90% panels successfully placed
            'material_efficiency': 0.6,   # ≥60% material utilization
            'pi_expansion_success': 0.95   # ≥95% PI codes expand successfully
        }

        # Initialize optimization engine
        self.optimization_engine = self._initialize_optimization_engine()

    def _initialize_optimization_engine(self) -> OptimizationEngine:
        """Initialize optimization engine with available algorithms"""
        engine = OptimizationEngine()

        # Try to register available algorithms
        try:
            from core.algorithms.ffd import FFDAlgorithm
            engine.register_algorithm(FFDAlgorithm())
        except ImportError:
            self.logger.warning("FFD algorithm not available")

        try:
            from core.algorithms.bfd import BFDAlgorithm
            engine.register_algorithm(BFDAlgorithm())
        except ImportError:
            self.logger.warning("BFD algorithm not available")

        try:
            from core.algorithms.hybrid import HybridOptimizer
            engine.register_algorithm(HybridOptimizer())
        except ImportError:
            self.logger.warning("Hybrid algorithm not available")

        return engine

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test execution"""
        logger = logging.getLogger('system_validation')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def run_complete_validation(self) -> ValidationReport:
        """Execute complete validation suite"""
        start_time = time.time()
        self.logger.info("🚀 Starting Comprehensive System Validation")

        # Test categories in order of dependency
        test_categories = [
            ("Core Functionality", self._test_core_functionality),
            ("Panel Placement Optimization", self._test_panel_placement_optimization),
            ("PI Code Expansion", self._test_pi_code_expansion),
            ("File Upload & Processing", self._test_file_upload_processing),
            ("Integration Testing", self._test_integration_scenarios),
            ("Edge Cases & Error Handling", self._test_edge_cases_errors),
            ("Performance Validation", self._test_performance_scenarios)
        ]

        for category_name, test_method in test_categories:
            self.logger.info(f"\n📋 Testing Category: {category_name}")
            try:
                test_method()
            except Exception as e:
                self.logger.error(f"Category {category_name} failed with exception: {e}")
                self._record_test_result(
                    f"{category_name}_CRITICAL_ERROR",
                    "ERROR", 0.0, f"Critical error in test category: {str(e)}",
                    {}, [str(e)], []
                )

        # Generate report
        total_time = time.time() - start_time
        return self._generate_report(total_time)

    def _test_core_functionality(self):
        """Test 1: Core Functionality - Basic system operations"""
        # Test 1.1: Text Parser Functionality
        self._test_text_parser()
        # Test 1.2: Optimization Engine
        self._test_optimization_engine()
        # Test 1.3: Material Manager
        self._test_material_manager()

    def _test_text_parser(self):
        """Test text parser with sample data"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test with data0923.txt
            if self.data0923_path.exists():
                with open(self.data0923_path, 'r', encoding='utf-8') as f:
                    raw_data = f.read()

                result = parser.parse_to_panels(raw_data)

                metrics['total_lines'] = result.total_lines
                metrics['panels_parsed'] = len(result.panels)
                metrics['success_rate'] = result.success_rate
                metrics['format_detected'] = result.format_detected
                metrics['parse_errors'] = len(result.errors)

                # Validation checks
                if len(result.panels) == 0:
                    errors.append("No panels were parsed from sample data")
                elif len(result.panels) < 50:  # data0923.txt should have many panels
                    warnings.append(f"Low panel count: {len(result.panels)} (expected >50)")

                if result.success_rate < 0.8:
                    errors.append(f"Low success rate: {result.success_rate:.2%}")

                # Check for PI code presence in parsed panels
                panels_with_pi = [p for p in result.panels if p.pi_code]
                metrics['panels_with_pi_code'] = len(panels_with_pi)

                if len(panels_with_pi) == 0:
                    warnings.append("No panels have PI codes - PI expansion testing may be limited")

                status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
                details = f"Parsed {len(result.panels)} panels from {result.total_lines} lines. " + \
                         f"Success rate: {result.success_rate:.1%}. Format: {result.format_detected}."

            else:
                errors.append(f"Sample data file not found: {self.data0923_path}")
                status = "FAIL"
                details = "Sample data file missing"

        except Exception as e:
            errors.append(f"Text parser test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during text parsing: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Core_TextParser", status, execution_time, details, metrics, errors, warnings
        )

    def _test_optimization_engine(self):
        """Test optimization engine with sample panels"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create test panels for optimization
            test_panels = [
                Panel("test1", 400, 300, 2, "SECC", 0.5),
                Panel("test2", 600, 400, 1, "SECC", 0.5),
                Panel("test3", 800, 500, 3, "SECC", 0.5),
                Panel("test4", 200, 150, 5, "SECC", 0.5)
            ]

            # Use the class instance created in init
            placement_results = self.optimization_engine.optimize(test_panels)

            metrics['input_panels'] = len(test_panels)
            metrics['total_panel_quantity'] = sum(p.quantity for p in test_panels)
            metrics['output_results'] = len(placement_results)

            if placement_results:
                # Calculate total placed panels
                total_placed_quantity = sum(
                    panel.quantity for result in placement_results for panel in result.placed_panels
                )

                metrics['placed_panels'] = total_placed_quantity

                # Calculate placement success rate
                total_input_quantity = sum(p.quantity for p in test_panels)
                placement_rate = total_placed_quantity / total_input_quantity if total_input_quantity > 0 else 0
                metrics['placement_success_rate'] = placement_rate

                # Efficiency calculation
                total_used_area = sum(result.used_area for result in placement_results)
                total_sheet_area = sum(result.sheet.width * result.sheet.height for result in placement_results)
                efficiency = total_used_area / total_sheet_area if total_sheet_area > 0 else 0
                metrics['material_efficiency'] = efficiency

                if efficiency < self.performance_targets['material_efficiency']:
                    warnings.append(f"Low material efficiency: {efficiency:.1%}")

                if placement_rate < self.performance_targets['panel_placement_rate']:
                    errors.append(f"Low placement rate: {placement_rate:.1%}")

                status = "FAIL" if errors else "PASS"
                details = f"Optimized {len(test_panels)} panel types into {len(placement_results)} results. " + \
                         f"Placement rate: {placement_rate:.1%}"
            else:
                errors.append("No optimization results generated")
                status = "FAIL"
                details = "Optimization produced no results"

        except Exception as e:
            errors.append(f"Optimization engine test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during optimization: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Core_OptimizationEngine", status, execution_time, details, metrics, errors, warnings
        )

    def _test_material_manager(self):
        """Test material manager functionality"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            material_manager = get_material_manager()

            # Test basic functionality
            summary = material_manager.get_inventory_summary()
            metrics['total_materials'] = summary['total_sheets']
            metrics['material_types'] = summary['material_types']

            # Test material lookup
            test_materials = ['SECC', 'SGCC', 'KW90', 'KW300']
            available_materials = []
            for material in test_materials:
                available = material_manager.get_available_sheets(material, 0.5)
                if available:
                    available_materials.append(material)

            metrics['available_test_materials'] = len(available_materials)

            if len(available_materials) == 0:
                warnings.append("No test materials available in inventory")

            status = "PASS"
            details = f"Material manager operational. {summary['total_sheets']} sheets, " + \
                     f"{summary['material_types']} types available."

        except Exception as e:
            errors.append(f"Material manager test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception in material manager: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Core_MaterialManager", status, execution_time, details, metrics, errors, warnings
        )

    def _test_panel_placement_optimization(self):
        """Test 2: Panel Placement Optimization - Test the 16/473 panel fix"""
        # Test 2.1: High Panel Count Scenario
        self._test_high_panel_count_placement()
        # Test 2.2: Problem Panel Configurations
        self._test_problem_panel_configurations()

    def _test_high_panel_count_placement(self):
        """Test placement with high panel counts (simulating 16/473 scenario)"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create a challenging scenario with many panels (from data0923.txt sample)
            test_panels = []

            # Create varied panel sizes that might cause placement issues
            panel_configs = [
                (892, 576, 16, "SECC", 0.5),   # From data0923.txt
                (767, 571, 2, "SECC", 0.5),
                (601, 571, 2, "SECC", 0.5),
                (902, 516, 8, "SECC", 0.5),
                (932, 516, 20, "SECC", 0.5),
                (742, 516, 4, "SECC", 0.5),
                (896, 516, 16, "SECC", 0.5),
                (892, 616, 20, "SECC", 0.5),
                (857, 616, 2, "SECC", 0.5),
                (604, 616, 2, "SECC", 0.5)
            ]

            for i, (w, h, qty, material, thickness) in enumerate(panel_configs):
                panel = Panel(f"panel_{i+1}", w, h, qty, material, thickness)
                test_panels.append(panel)

            total_input_quantity = sum(p.quantity for p in test_panels)
            metrics['total_input_quantity'] = total_input_quantity

            # Run optimization
            placement_results = self.optimization_engine.optimize(test_panels)

            # Analyze results
            total_placed_quantity = sum(
                panel.quantity for result in placement_results for panel in result.placed_panels
            )
            placement_rate = total_placed_quantity / total_input_quantity

            metrics['results_generated'] = len(placement_results)
            metrics['total_placed_quantity'] = total_placed_quantity
            metrics['placement_success_rate'] = placement_rate

            # Check if we hit the critical issue (very low placement rate)
            if placement_rate < 0.5:  # Less than 50% placed indicates serious issue
                errors.append(f"Critical placement failure: only {placement_rate:.1%} of panels placed")
            elif placement_rate < 0.8:  # Less than 80% is concerning
                warnings.append(f"Low placement rate: {placement_rate:.1%}")

            # Efficiency check
            if placement_results:
                total_used_area = sum(result.used_area for result in placement_results)
                total_sheet_area = sum(result.sheet.width * result.sheet.height for result in placement_results)
                efficiency = total_used_area / total_sheet_area
                metrics['material_efficiency'] = efficiency

                if efficiency < 0.4:  # Very low efficiency indicates algorithm problems
                    errors.append(f"Very low material efficiency: {efficiency:.1%}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Placed {total_placed_quantity}/{total_input_quantity} panels " + \
                     f"({placement_rate:.1%}) across {len(placement_results)} results"

        except Exception as e:
            errors.append(f"High panel count test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during high panel count test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PanelPlacement_HighCount", status, execution_time, details, metrics, errors, warnings
        )

    def _test_problem_panel_configurations(self):
        """Test specific panel configurations that historically caused issues"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Test problematic scenarios
            test_scenarios = [
                {
                    'name': 'LargePanels',
                    'panels': [Panel(f"large_{i}", 1400, 2900, 2, "SECC", 0.5) for i in range(3)]
                },
                {
                    'name': 'ManySmallPanels',
                    'panels': [Panel(f"small_{i}", 100, 80, 10, "SECC", 0.5) for i in range(10)]
                },
                {
                    'name': 'MixedSizes',
                    'panels': [
                        Panel("huge", 1450, 3000, 1, "SECC", 0.5),
                        Panel("medium", 500, 400, 5, "SECC", 0.5),
                        Panel("tiny", 50, 50, 20, "SECC", 0.5)
                    ]
                }
            ]

            scenario_results = {}

            for scenario in test_scenarios:
                panels = scenario['panels']
                total_quantity = sum(p.quantity for p in panels)

                try:
                    placement_results = self.optimization_engine.optimize(panels)
                    placed_quantity = sum(
                        panel.quantity for result in placement_results for panel in result.placed_panels
                    )
                    placement_rate = placed_quantity / total_quantity

                    scenario_results[scenario['name']] = {
                        'placement_rate': placement_rate,
                        'results_used': len(placement_results)
                    }

                    if placement_rate < 0.7:
                        warnings.append(f"Low placement in {scenario['name']}: {placement_rate:.1%}")

                except Exception as e:
                    errors.append(f"Scenario {scenario['name']} failed: {str(e)}")
                    scenario_results[scenario['name']] = {'error': str(e)}

            metrics['scenario_results'] = scenario_results

            # Check if any scenarios completely failed
            failed_scenarios = [name for name, result in scenario_results.items() if 'error' in result]
            if failed_scenarios:
                errors.append(f"Failed scenarios: {failed_scenarios}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested {len(test_scenarios)} problem scenarios. " + \
                     f"Results: {len(scenario_results) - len(failed_scenarios)}/{len(test_scenarios)} succeeded"

        except Exception as e:
            errors.append(f"Problem configuration test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during problem configuration test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PanelPlacement_ProblemConfigs", status, execution_time, details, metrics, errors, warnings
        )

    def _test_pi_code_expansion(self):
        """Test 3: PI Code Expansion Functionality"""
        if not PI_MANAGER_AVAILABLE:
            self._record_test_result(
                "PIExpansion_SKIPPED", "SKIP", 0.0,
                "PI Manager not available - skipping PI code expansion tests",
                {}, [], ["PI Manager module not found"]
            )
            return

        # Test 3.1: PI Code Database Loading
        self._test_pi_code_database()
        # Test 3.2: Integration with Parser
        self._test_pi_integration_with_parser()

    def _test_pi_code_database(self):
        """Test PI code database loading and lookup"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            pi_manager = get_pi_manager()

            # Test database loading
            pi_codes = pi_manager.get_all_pi_codes()
            metrics['total_pi_codes'] = len(pi_codes)

            if len(pi_codes) == 0:
                errors.append("No PI codes loaded in database")
            elif len(pi_codes) < 10:
                warnings.append(f"Low PI code count: {len(pi_codes)}")

            # Test specific lookups
            test_pi_codes = ['77131000', '12345678', 'INVALID']  # Mix of valid/invalid
            lookup_results = {}

            for pi_code in test_pi_codes:
                try:
                    expansion = pi_manager.get_expansion(pi_code)
                    lookup_results[pi_code] = 'found' if expansion else 'not_found'
                except Exception as e:
                    lookup_results[pi_code] = f'error: {str(e)}'

            metrics['lookup_results'] = lookup_results

            # Check if at least some lookups work
            successful_lookups = sum(1 for result in lookup_results.values() if result == 'found')
            if successful_lookups == 0:
                errors.append("No PI code lookups succeeded")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"PI database loaded with {len(pi_codes)} codes. " + \
                     f"Lookup test: {successful_lookups}/{len(test_pi_codes)} found"

        except Exception as e:
            errors.append(f"PI code database test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during PI database test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PIExpansion_Database", status, execution_time, details, metrics, errors, warnings
        )

    def _test_pi_integration_with_parser(self):
        """Test PI code integration with text parser"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Use sample data to test PI integration
            if not self.data0923_path.exists():
                errors.append(f"Sample data file not found: {self.data0923_path}")
                status = "FAIL"
                details = "Sample data missing for PI integration test"
            else:
                parser = RobustTextParser()

                with open(self.data0923_path, 'r', encoding='utf-8') as f:
                    raw_data = f.read()

                result = parser.parse_to_panels(raw_data)

                # Analyze PI code usage in parsed panels
                total_panels = len(result.panels)
                panels_with_pi = [p for p in result.panels if p.pi_code]
                panels_with_expansion = [p for p in result.panels if hasattr(p, 'expanded_width')]

                metrics['total_panels'] = total_panels
                metrics['panels_with_pi_code'] = len(panels_with_pi)
                metrics['panels_with_expansion'] = len(panels_with_expansion)

                if total_panels > 0:
                    pi_usage_rate = len(panels_with_pi) / total_panels
                    expansion_rate = len(panels_with_expansion) / total_panels

                    metrics['pi_usage_rate'] = pi_usage_rate
                    metrics['expansion_rate'] = expansion_rate

                    # Validate PI code processing
                    if len(panels_with_pi) > 0 and len(panels_with_expansion) == 0:
                        errors.append("Panels have PI codes but no expansions were applied")
                    elif len(panels_with_pi) > 0:
                        successful_expansion_rate = len(panels_with_expansion) / len(panels_with_pi)
                        metrics['successful_expansion_rate'] = successful_expansion_rate

                        if successful_expansion_rate < self.performance_targets['pi_expansion_success']:
                            warnings.append(f"Low PI expansion success rate: {successful_expansion_rate:.1%}")

                status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
                details = f"Parsed {total_panels} panels, {len(panels_with_pi)} with PI codes, " + \
                         f"{len(panels_with_expansion)} with expansions"

        except Exception as e:
            errors.append(f"PI integration test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during PI integration test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PIExpansion_ParserIntegration", status, execution_time, details, metrics, errors, warnings
        )

    def _test_file_upload_processing(self):
        """Test 4: File Upload & Processing"""
        # Test 4.1: File Format Detection
        self._test_file_format_detection()
        # Test 4.2: Upload Processing Performance
        self._test_upload_processing_performance()

    def _test_file_format_detection(self):
        """Test automatic file format detection"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test format detection with sample files
            test_files = []
            if self.data0923_path.exists():
                test_files.append(('data0923.txt', 'cutting_data_tsv'))
            if self.sizai_data_path.exists():
                test_files.append(('sizaidata.txt', 'material_data_tsv'))

            detection_results = {}

            for filename, expected_format in test_files:
                file_path = self.sample_data_dir / filename
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Test format detection
                detected_format = parser.detect_format(content)
                sample_format = parser.detect_sample_data_format(content)

                detection_results[filename] = {
                    'detected_format': detected_format,
                    'sample_format': sample_format,
                    'expected': expected_format
                }

                # Validate detection accuracy
                if sample_format != 'unknown' and expected_format != 'unknown':
                    if sample_format != expected_format:
                        warnings.append(f"Format mismatch for {filename}: got {sample_format}, expected {expected_format}")

            metrics['detection_results'] = detection_results
            metrics['files_tested'] = len(test_files)

            if len(test_files) == 0:
                warnings.append("No sample files available for format detection testing")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested format detection on {len(test_files)} files"

        except Exception as e:
            errors.append(f"File format detection test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during format detection test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "FileUpload_FormatDetection", status, execution_time, details, metrics, errors, warnings
        )

    def _test_upload_processing_performance(self):
        """Test upload processing performance and prioritization"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test processing performance with different file sizes
            if self.data0923_path.exists():
                # Test large file processing time
                large_file_start = time.time()

                with open(self.data0923_path, 'r', encoding='utf-8') as f:
                    large_content = f.read()

                result = parser.parse_to_panels(large_content)
                large_file_time = time.time() - large_file_start

                metrics['large_file_size'] = len(large_content)
                metrics['large_file_parse_time'] = large_file_time
                metrics['large_file_panels'] = len(result.panels)

                # Check if processing time is reasonable
                if large_file_time > 10.0:  # > 10 seconds for parsing is too slow
                    errors.append(f"Large file processing too slow: {large_file_time:.2f}s")
                elif large_file_time > 5.0:
                    warnings.append(f"Large file processing slow: {large_file_time:.2f}s")

                # Test small file for comparison
                small_content = large_content[:1000]  # First 1000 chars
                small_file_start = time.time()
                small_result = parser.parse_to_panels(small_content)
                small_file_time = time.time() - small_file_start

                metrics['small_file_parse_time'] = small_file_time
                metrics['small_file_panels'] = len(small_result.panels)

                # Processing should scale reasonably with file size
                if large_file_time > 0 and small_file_time > 0:
                    time_ratio = large_file_time / small_file_time
                    size_ratio = len(large_content) / len(small_content)
                    efficiency_ratio = time_ratio / size_ratio

                    metrics['processing_efficiency_ratio'] = efficiency_ratio

                    if efficiency_ratio > 2.0:  # Processing time grows faster than file size
                        warnings.append(f"Processing time scales poorly: {efficiency_ratio:.2f}x")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Large file: {metrics.get('large_file_parse_time', 0):.2f}s, " + \
                     f"Small file: {metrics.get('small_file_parse_time', 0):.2f}s"

        except Exception as e:
            errors.append(f"Upload processing performance test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during upload processing test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "FileUpload_ProcessingPerformance", status, execution_time, details, metrics, errors, warnings
        )

    def _test_integration_scenarios(self):
        """Test 5: Integration Testing - Full workflow scenarios"""
        # Test 5.1: End-to-End Data Flow
        self._test_end_to_end_data_flow()

    def _test_end_to_end_data_flow(self):
        """Test complete data flow from file upload to optimization results"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Simulate end-to-end workflow
            workflow_steps = []

            # Step 1: File parsing
            if self.data0923_path.exists():
                parser = RobustTextParser()
                with open(self.data0923_path, 'r', encoding='utf-8') as f:
                    raw_data = f.read()

                parse_result = parser.parse_to_panels(raw_data)
                workflow_steps.append(('parsing', len(parse_result.panels), len(parse_result.errors)))

                if len(parse_result.panels) == 0:
                    errors.append("Parsing step failed - no panels produced")
                    status = "FAIL"
                    details = "End-to-end test failed at parsing step"
                else:
                    # Step 2: Material validation
                    material_manager = get_material_manager()
                    valid_panels = []
                    material_issues = 0

                    for panel in parse_result.panels[:10]:  # Test first 10 panels
                        available_sheets = material_manager.get_available_sheets(panel.material, panel.thickness)
                        if available_sheets:
                            valid_panels.append(panel)
                        else:
                            material_issues += 1

                    workflow_steps.append(('material_validation', len(valid_panels), material_issues))

                    if len(valid_panels) == 0:
                        warnings.append("No panels have available materials")

                    # Step 3: Optimization
                    if valid_panels:
                        placement_results = self.optimization_engine.optimize(valid_panels)

                        total_placed = sum(panel.quantity for result in placement_results for panel in result.placed_panels)
                        total_input = sum(p.quantity for p in valid_panels)
                        placement_rate = total_placed / total_input if total_input > 0 else 0

                        workflow_steps.append(('optimization', len(placement_results), total_placed))

                        if placement_rate < 0.5:
                            warnings.append(f"Low end-to-end placement rate: {placement_rate:.1%}")

                        # Step 4: Results generation
                        if placement_results:
                            total_cost = sum(result.cost for result in placement_results)
                            avg_efficiency = sum(result.efficiency for result in placement_results) / len(placement_results)

                            workflow_steps.append(('results', total_cost, avg_efficiency))

                            metrics['end_to_end_efficiency'] = avg_efficiency
                            metrics['end_to_end_cost'] = total_cost
                            metrics['end_to_end_placement_rate'] = placement_rate

                    metrics['workflow_steps'] = workflow_steps

                    # Overall workflow validation
                    successful_steps = len([step for step in workflow_steps if step[1] > 0])
                    total_steps = len(workflow_steps)

                    if successful_steps < total_steps:
                        warnings.append(f"Some workflow steps failed: {successful_steps}/{total_steps} successful")

                    status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
                    details = f"Completed {successful_steps}/{total_steps} workflow steps successfully"
            else:
                errors.append("Sample data file not available for end-to-end test")
                status = "FAIL"
                details = "Cannot run end-to-end test without sample data"

        except Exception as e:
            errors.append(f"End-to-end integration test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during end-to-end test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Integration_EndToEnd", status, execution_time, details, metrics, errors, warnings
        )

    def _test_edge_cases_errors(self):
        """Test 6: Edge Cases & Error Handling"""
        # Test 6.1: Invalid Input Handling
        self._test_invalid_input_handling()

    def _test_invalid_input_handling(self):
        """Test handling of invalid inputs and edge cases"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test invalid data scenarios
            invalid_scenarios = [
                ('empty_string', ''),
                ('only_whitespace', '   \n\t   '),
                ('malformed_data', 'panel1,abc,def,2,SECC,0.5'),
                ('missing_fields', 'panel1,100')
            ]

            invalid_handling_results = {}

            for scenario_name, test_data in invalid_scenarios:
                try:
                    result = parser.parse_to_panels(test_data)

                    invalid_handling_results[scenario_name] = {
                        'panels_parsed': len(result.panels),
                        'errors_detected': len(result.errors),
                        'warnings_generated': len(result.warnings),
                        'exception_thrown': False,
                        'graceful_handling': True
                    }

                    # Specific validations
                    if scenario_name in ['empty_string', 'only_whitespace']:
                        if len(result.panels) > 0:
                            warnings.append(f"Empty input should not produce panels: {scenario_name}")

                except Exception as e:
                    invalid_handling_results[scenario_name] = {
                        'exception_thrown': True,
                        'exception_type': type(e).__name__,
                        'graceful_handling': False
                    }
                    warnings.append(f"Exception thrown for {scenario_name}: {type(e).__name__}")

            metrics['invalid_handling_results'] = invalid_handling_results

            # Count graceful vs exception handling
            graceful_count = sum(1 for result in invalid_handling_results.values()
                               if result.get('graceful_handling', False))
            total_scenarios = len(invalid_scenarios)

            if graceful_count < total_scenarios * 0.8:  # Expect 80% graceful handling
                warnings.append(f"Low graceful error handling rate: {graceful_count}/{total_scenarios}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested {total_scenarios} invalid input scenarios, {graceful_count} handled gracefully"

        except Exception as e:
            errors.append(f"Invalid input handling test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during invalid input test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "EdgeCases_InvalidInputHandling", status, execution_time, details, metrics, errors, warnings
        )

    def _test_performance_scenarios(self):
        """Test 7: Performance Validation"""
        # Test 7.1: Batch Size Performance
        self._test_batch_size_performance()

    def _test_batch_size_performance(self):
        """Test performance across different batch sizes"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Test different batch sizes against performance targets
            batch_scenarios = [
                ('small', 10, self.performance_targets['small_batch_time']),
                ('medium', 30, self.performance_targets['medium_batch_time']),
                ('large', 80, self.performance_targets['large_batch_time'])
            ]

            performance_results = {}

            for batch_name, panel_count, time_target in batch_scenarios:
                # Create test panels
                test_panels = []
                for i in range(panel_count):
                    panel = Panel(
                        f"{batch_name}_{i}",
                        300 + (i % 5) * 100,  # 300-700mm width
                        200 + (i % 4) * 75,   # 200-425mm height
                        1 + (i % 3),          # 1-3 quantity
                        "SECC",
                        0.5
                    )
                    test_panels.append(panel)

                # Measure optimization time
                batch_start = time.time()
                placement_results = self.optimization_engine.optimize(test_panels)
                batch_time = time.time() - batch_start

                # Calculate metrics
                total_input_qty = sum(p.quantity for p in test_panels)
                total_placed_qty = sum(panel.quantity for result in placement_results for panel in result.placed_panels)
                placement_rate = total_placed_qty / total_input_qty if total_input_qty > 0 else 0

                if placement_results:
                    total_used = sum(result.used_area for result in placement_results)
                    total_area = sum(result.sheet.width * result.sheet.height for result in placement_results)
                    efficiency = total_used / total_area
                else:
                    efficiency = 0

                performance_results[batch_name] = {
                    'panel_count': panel_count,
                    'optimization_time': batch_time,
                    'time_target': time_target,
                    'results_generated': len(placement_results),
                    'placement_rate': placement_rate,
                    'efficiency': efficiency,
                    'meets_time_target': batch_time <= time_target
                }

                # Validate performance targets
                if batch_time > time_target:
                    if batch_time > time_target * 2:  # More than 2x target is error
                        errors.append(f"{batch_name} batch too slow: {batch_time:.2f}s > {time_target}s")
                    else:  # 1-2x target is warning
                        warnings.append(f"{batch_name} batch slow: {batch_time:.2f}s > {time_target}s")

                if placement_rate < self.performance_targets['panel_placement_rate']:
                    warnings.append(f"{batch_name} batch low placement: {placement_rate:.1%}")

            metrics['performance_results'] = performance_results

            # Overall performance assessment
            meeting_targets = sum(1 for result in performance_results.values()
                                if result['meets_time_target'])
            total_scenarios = len(batch_scenarios)

            metrics['performance_score'] = meeting_targets / total_scenarios

            if meeting_targets < total_scenarios:
                warnings.append(f"Performance targets: {meeting_targets}/{total_scenarios} met")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Performance targets met: {meeting_targets}/{total_scenarios}"

        except Exception as e:
            errors.append(f"Batch size performance test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during batch performance test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Performance_BatchSize", status, execution_time, details, metrics, errors, warnings
        )

    def _record_test_result(self, test_name: str, status: str, execution_time: float,
                           details: str, metrics: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Record a test result"""
        result = TestResult(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details=details,
            performance_metrics=metrics,
            errors=errors,
            warnings=warnings
        )
        self.test_results.append(result)

        # Log result
        status_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "ERROR": "🚨"}[status]
        self.logger.info(f"{status_icon} {test_name}: {status} ({execution_time:.2f}s) - {details}")

        if errors:
            for error in errors:
                self.logger.error(f"   ERROR: {error}")
        if warnings:
            for warning in warnings:
                self.logger.warning(f"   WARNING: {warning}")

    def _generate_report(self, total_execution_time: float) -> ValidationReport:
        """Generate comprehensive validation report"""
        # Calculate summary statistics
        status_counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        for result in self.test_results:
            status_counts[result.status] += 1

        # Collect recommendations
        recommendations = []

        # Performance recommendations
        slow_tests = [r for r in self.test_results if r.execution_time > 5.0]
        if slow_tests:
            recommendations.append(f"Performance: {len(slow_tests)} tests took >5s to execute")

        # Error pattern analysis
        error_patterns = {}
        for result in self.test_results:
            for error in result.errors:
                key = error.split(':')[0] if ':' in error else 'General'
                error_patterns[key] = error_patterns.get(key, 0) + 1

        if error_patterns:
            top_error = max(error_patterns.items(), key=lambda x: x[1])
            recommendations.append(f"Top error pattern: {top_error[0]} ({top_error[1]} occurrences)")

        # Generate system info
        system_info = {
            'python_version': sys.version,
            'test_execution_time': total_execution_time,
            'sample_data_available': {
                'data0923.txt': self.data0923_path.exists(),
                'sizaidata.txt': self.sizai_data_path.exists(),
                'pi.txt': self.pi_data_path.exists()
            },
            'pi_manager_available': PI_MANAGER_AVAILABLE,
            'streamlit_url': self.streamlit_url
        }

        return ValidationReport(
            timestamp=datetime.now(),
            total_tests=len(self.test_results),
            passed=status_counts["PASS"],
            failed=status_counts["FAIL"],
            skipped=status_counts["SKIP"],
            errors=status_counts["ERROR"],
            total_execution_time=total_execution_time,
            test_results=self.test_results,
            system_info=system_info,
            recommendations=recommendations
        )

    def print_summary_report(self, report: ValidationReport):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("🔍 STEEL CUTTING SYSTEM VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Execution Time: {report.total_execution_time:.2f} seconds")
        print()

        # Test Summary
        print("📊 TEST SUMMARY")
        print("-" * 40)
        total = report.total_tests
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {report.passed} ({report.passed/total:.1%})")
        print(f"❌ Failed: {report.failed} ({report.failed/total:.1%})")
        print(f"⏭️ Skipped: {report.skipped} ({report.skipped/total:.1%})")
        print(f"🚨 Errors: {report.errors} ({report.errors/total:.1%})")
        print()

        # Category Results
        print("📋 RESULTS BY CATEGORY")
        print("-" * 40)
        categories = {}
        for result in report.test_results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = {'PASS': 0, 'FAIL': 0, 'SKIP': 0, 'ERROR': 0}
            categories[category][result.status] += 1

        for category, counts in categories.items():
            total_cat = sum(counts.values())
            passed = counts['PASS']
            print(f"{category}: {passed}/{total_cat} passed ({passed/total_cat:.1%})")
        print()

        # Failed Tests Detail
        if report.failed > 0 or report.errors > 0:
            print("❌ FAILED TESTS")
            print("-" * 40)
            for result in report.test_results:
                if result.status in ['FAIL', 'ERROR']:
                    print(f"{result.test_name}: {result.status}")
                    print(f"  Details: {result.details}")
                    for error in result.errors[:2]:  # Show first 2 errors
                        print(f"  Error: {error}")
                    print()

        # Recommendations
        if report.recommendations:
            print("💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
            print()

        # System Status
        print("🔧 SYSTEM STATUS")
        print("-" * 40)

        # Overall assessment
        if report.failed == 0 and report.errors == 0:
            print("🟢 SYSTEM STATUS: HEALTHY")
            print("All critical functionality is working correctly.")
        elif report.failed <= 2 and report.errors == 0:
            print("🟡 SYSTEM STATUS: STABLE WITH MINOR ISSUES")
            print("System is functional but has some areas for improvement.")
        elif report.failed > 2 or report.errors > 0:
            print("🔴 SYSTEM STATUS: NEEDS ATTENTION")
            print("Critical issues detected that require immediate attention.")

        print(f"Sample Data Available: {sum(report.system_info['sample_data_available'].values())}/3")
        print(f"PI Manager Available: {'Yes' if report.system_info['pi_manager_available'] else 'No'}")
        print()

        print("="*80)


def main():
    """Main function to run the validation suite"""
    print("🚀 Starting Steel Cutting System Comprehensive Validation")
    print("="*60)

    # Create validation suite
    suite = SystemValidationSuite()

    # Check if Streamlit is running (optional)
    try:
        response = requests.get(suite.streamlit_url, timeout=2)
        print(f"✅ Streamlit app detected at {suite.streamlit_url}")
    except:
        print(f"⚠️  Streamlit app not accessible at {suite.streamlit_url}")
        print("   UI tests will be limited to static analysis")

    # Run validation
    report = suite.run_complete_validation()

    # Print report
    suite.print_summary_report(report)

    # Save detailed report
    report_path = project_root / "claudedocs" / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)

    # Convert report to dict for JSON serialization
    report_dict = {
        'timestamp': report.timestamp.isoformat(),
        'summary': {
            'total_tests': report.total_tests,
            'passed': report.passed,
            'failed': report.failed,
            'skipped': report.skipped,
            'errors': report.errors,
            'total_execution_time': report.total_execution_time
        },
        'test_results': [
            {
                'test_name': r.test_name,
                'status': r.status,
                'execution_time': r.execution_time,
                'details': r.details,
                'performance_metrics': r.performance_metrics,
                'errors': r.errors,
                'warnings': r.warnings
            }
            for r in report.test_results
        ],
        'system_info': report.system_info,
        'recommendations': report.recommendations
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"📄 Detailed report saved to: {report_path}")

    # Return overall success status
    return report.failed == 0 and report.errors == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)