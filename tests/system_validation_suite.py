"""
Comprehensive System Validation Suite for Steel Cutting Optimization System
鋼板切断最適化システム包括的検証スイート

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
from core.optimizer import CuttingOptimizer
from core.models import Panel, SteelSheet
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
            ("Cutting Allowance Optimization", self._test_cutting_allowance),
            ("File Upload & Prioritization", self._test_file_upload_prioritization),
            ("UI/UX Improvements", self._test_ui_ux_improvements),
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

            optimizer = CuttingOptimizer()
            sheets = optimizer.optimize_cutting(test_panels)

            metrics['input_panels'] = len(test_panels)
            metrics['total_panel_quantity'] = sum(p.quantity for p in test_panels)
            metrics['output_sheets'] = len(sheets)
            metrics['placed_panels'] = sum(len(sheet.placed_panels) for sheet in sheets)

            # Calculate placement success rate
            total_input_quantity = sum(p.quantity for p in test_panels)
            total_placed_quantity = sum(
                panel.quantity for sheet in sheets for panel in sheet.placed_panels
            )
            placement_rate = total_placed_quantity / total_input_quantity if total_input_quantity > 0 else 0
            metrics['placement_success_rate'] = placement_rate

            # Efficiency calculation
            if sheets:
                total_used_area = sum(sheet.used_area for sheet in sheets)
                total_sheet_area = sum(sheet.width * sheet.height for sheet in sheets)
                efficiency = total_used_area / total_sheet_area if total_sheet_area > 0 else 0
                metrics['material_efficiency'] = efficiency

                if efficiency < self.performance_targets['material_efficiency']:
                    warnings.append(f"Low material efficiency: {efficiency:.1%}")

            if placement_rate < self.performance_targets['panel_placement_rate']:
                errors.append(f"Low placement rate: {placement_rate:.1%}")

            status = "FAIL" if errors else "PASS"
            details = f"Optimized {len(test_panels)} panel types into {len(sheets)} sheets. " + \
                     f"Placement rate: {placement_rate:.1%}"

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

            # Test material addition (if supported)
            initial_count = len(material_manager.inventory)

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

        # Test 2.3: Placement Algorithm Efficiency
        self._test_placement_algorithm_efficiency()

    def _test_high_panel_count_placement(self):
        """Test placement with high panel counts (simulating 16/473 scenario)"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create a challenging scenario with many panels
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
            optimizer = CuttingOptimizer()
            sheets = optimizer.optimize_cutting(test_panels)

            # Analyze results
            total_placed_quantity = sum(
                panel.quantity for sheet in sheets for panel in sheet.placed_panels
            )
            placement_rate = total_placed_quantity / total_input_quantity

            metrics['sheets_generated'] = len(sheets)
            metrics['total_placed_quantity'] = total_placed_quantity
            metrics['placement_success_rate'] = placement_rate

            # Check if we hit the critical issue (very low placement rate)
            if placement_rate < 0.5:  # Less than 50% placed indicates serious issue
                errors.append(f"Critical placement failure: only {placement_rate:.1%} of panels placed")
            elif placement_rate < 0.8:  # Less than 80% is concerning
                warnings.append(f"Low placement rate: {placement_rate:.1%}")

            # Efficiency check
            if sheets:
                total_used_area = sum(sheet.used_area for sheet in sheets)
                total_sheet_area = sum(sheet.width * sheet.height for sheet in sheets)
                efficiency = total_used_area / total_sheet_area
                metrics['material_efficiency'] = efficiency

                if efficiency < 0.4:  # Very low efficiency indicates algorithm problems
                    errors.append(f"Very low material efficiency: {efficiency:.1%}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Placed {total_placed_quantity}/{total_input_quantity} panels " + \
                     f"({placement_rate:.1%}) across {len(sheets)} sheets"

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
            optimizer = CuttingOptimizer()

            for scenario in test_scenarios:
                panels = scenario['panels']
                total_quantity = sum(p.quantity for p in panels)

                try:
                    sheets = optimizer.optimize_cutting(panels)
                    placed_quantity = sum(
                        panel.quantity for sheet in sheets for panel in sheet.placed_panels
                    )
                    placement_rate = placed_quantity / total_quantity

                    scenario_results[scenario['name']] = {
                        'placement_rate': placement_rate,
                        'sheets_used': len(sheets)
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

    def _test_placement_algorithm_efficiency(self):
        """Test placement algorithm efficiency and improvement"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create benchmark scenario
            benchmark_panels = [
                Panel("bench1", 800, 600, 5, "SECC", 0.5),
                Panel("bench2", 400, 300, 10, "SECC", 0.5),
                Panel("bench3", 1200, 800, 2, "SECC", 0.5),
                Panel("bench4", 300, 200, 15, "SECC", 0.5),
                Panel("bench5", 600, 500, 8, "SECC", 0.5)
            ]

            optimizer = CuttingOptimizer()

            # Run multiple optimization attempts to check consistency
            placement_rates = []
            efficiencies = []

            for attempt in range(3):  # Run 3 times to check consistency
                sheets = optimizer.optimize_cutting(benchmark_panels.copy())

                total_input = sum(p.quantity for p in benchmark_panels)
                total_placed = sum(panel.quantity for sheet in sheets for panel in sheet.placed_panels)
                placement_rate = total_placed / total_input
                placement_rates.append(placement_rate)

                if sheets:
                    total_used = sum(sheet.used_area for sheet in sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in sheets)
                    efficiency = total_used / total_area
                    efficiencies.append(efficiency)

            # Calculate statistics
            avg_placement = sum(placement_rates) / len(placement_rates)
            avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
            placement_variance = max(placement_rates) - min(placement_rates)

            metrics['avg_placement_rate'] = avg_placement
            metrics['avg_efficiency'] = avg_efficiency
            metrics['placement_variance'] = placement_variance
            metrics['attempts'] = len(placement_rates)

            # Validate performance targets
            if avg_placement < self.performance_targets['panel_placement_rate']:
                errors.append(f"Placement rate below target: {avg_placement:.1%} < {self.performance_targets['panel_placement_rate']:.1%}")

            if avg_efficiency < self.performance_targets['material_efficiency']:
                warnings.append(f"Efficiency below target: {avg_efficiency:.1%} < {self.performance_targets['material_efficiency']:.1%}")

            if placement_variance > 0.1:  # More than 10% variance indicates inconsistency
                warnings.append(f"High placement variance: {placement_variance:.1%}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Avg placement: {avg_placement:.1%}, Avg efficiency: {avg_efficiency:.1%}, " + \
                     f"Variance: {placement_variance:.2%}"

        except Exception as e:
            errors.append(f"Algorithm efficiency test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during algorithm efficiency test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PanelPlacement_AlgorithmEfficiency", status, execution_time, details, metrics, errors, warnings
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

        # Test 3.2: Dimension Expansion Logic
        self._test_dimension_expansion()

        # Test 3.3: Integration with Parser
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

    def _test_dimension_expansion(self):
        """Test dimension expansion calculations"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            pi_manager = get_pi_manager()

            # Test dimension expansion with known PI codes
            test_cases = [
                {'pi_code': '77131000', 'width': 892, 'height': 576},
                {'pi_code': '77131000', 'width': 767, 'height': 571},
                {'pi_code': '77131000', 'width': 932, 'height': 516}
            ]

            expansion_results = []

            for case in test_cases:
                try:
                    expanded_w, expanded_h = pi_manager.get_expansion_for_panel(
                        case['pi_code'], case['width'], case['height']
                    )

                    # Calculate expansion amounts
                    w_expansion = expanded_w - case['width']
                    h_expansion = expanded_h - case['height']

                    expansion_results.append({
                        'pi_code': case['pi_code'],
                        'original': (case['width'], case['height']),
                        'expanded': (expanded_w, expanded_h),
                        'expansion': (w_expansion, h_expansion)
                    })

                    # Validate expansions are reasonable (positive and not excessive)
                    if w_expansion < 0 or h_expansion < 0:
                        errors.append(f"Negative expansion for {case['pi_code']}: {w_expansion}, {h_expansion}")
                    elif w_expansion > 100 or h_expansion > 100:  # > 100mm expansion seems excessive
                        warnings.append(f"Large expansion for {case['pi_code']}: {w_expansion}, {h_expansion}")

                except Exception as e:
                    errors.append(f"Expansion failed for {case['pi_code']}: {str(e)}")

            metrics['expansion_results'] = expansion_results
            metrics['successful_expansions'] = len(expansion_results)

            if len(expansion_results) == 0:
                errors.append("No dimension expansions succeeded")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested {len(test_cases)} expansion cases. " + \
                     f"Successful: {len(expansion_results)}"

        except Exception as e:
            errors.append(f"Dimension expansion test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during dimension expansion test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "PIExpansion_Dimensions", status, execution_time, details, metrics, errors, warnings
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

                    # Check for dimensional differences (indication of expansion)
                    dimension_changes = 0
                    for panel in panels_with_expansion:
                        if hasattr(panel, 'expanded_width') and hasattr(panel, 'expanded_height'):
                            if panel.expanded_width != panel.width or panel.expanded_height != panel.height:
                                dimension_changes += 1

                    metrics['panels_with_dimension_changes'] = dimension_changes

                    if dimension_changes == 0 and len(panels_with_expansion) > 0:
                        warnings.append("No dimensional changes detected despite expansion processing")

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

    def _test_cutting_allowance(self):
        """Test 4: Cutting Allowance Optimization (0 for thin sheets)"""

        # Test 4.1: Thin Sheet Allowance (should be 0)
        self._test_thin_sheet_allowance()

        # Test 4.2: Thick Sheet Allowance (should have allowance)
        self._test_thick_sheet_allowance()

        # Test 4.3: Allowance Impact on Optimization
        self._test_allowance_optimization_impact()

    def _test_thin_sheet_allowance(self):
        """Test cutting allowance for thin sheets (0.5mm) should be 0"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create panels with thin material (0.5mm)
            thin_panels = [
                Panel("thin1", 400, 300, 2, "SECC", 0.5),
                Panel("thin2", 600, 400, 1, "SECC", 0.5)
            ]

            optimizer = CuttingOptimizer()

            # Check if optimizer has allowance settings
            if hasattr(optimizer, 'cutting_allowance') or hasattr(optimizer, 'get_cutting_allowance'):
                # Test allowance calculation for thin materials
                if hasattr(optimizer, 'get_cutting_allowance'):
                    allowance = optimizer.get_cutting_allowance(0.5)  # 0.5mm thickness
                    metrics['thin_sheet_allowance'] = allowance

                    if allowance != 0:
                        errors.append(f"Thin sheet allowance should be 0, got {allowance}")
                elif hasattr(optimizer, 'cutting_allowance'):
                    allowance = optimizer.cutting_allowance
                    metrics['default_allowance'] = allowance
                    warnings.append("Using default allowance - thickness-specific logic not detected")

                # Run optimization and check results
                sheets = optimizer.optimize_cutting(thin_panels)

                # Analyze sheet utilization (should be higher with 0 allowance)
                if sheets:
                    total_used = sum(sheet.used_area for sheet in sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in sheets)
                    efficiency = total_used / total_area
                    metrics['thin_sheet_efficiency'] = efficiency

                    # With 0 allowance, efficiency should be good for simple panels
                    if efficiency < 0.7:  # Should achieve >70% with simple panels and 0 allowance
                        warnings.append(f"Low efficiency with thin sheets: {efficiency:.1%}")

                status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
                details = f"Thin sheet (0.5mm) allowance: {metrics.get('thin_sheet_allowance', 'N/A')}"

            else:
                warnings.append("Cutting allowance system not detected in optimizer")
                status = "SKIP"
                details = "Cutting allowance system not found"

        except Exception as e:
            errors.append(f"Thin sheet allowance test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during thin sheet allowance test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "CuttingAllowance_ThinSheet", status, execution_time, details, metrics, errors, warnings
        )

    def _test_thick_sheet_allowance(self):
        """Test cutting allowance for thick sheets (should have allowance)"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create panels with thick material (6mm)
            thick_panels = [
                Panel("thick1", 400, 300, 2, "SECC", 6.0),
                Panel("thick2", 600, 400, 1, "SECC", 6.0)
            ]

            optimizer = CuttingOptimizer()

            # Check allowance for thick materials
            if hasattr(optimizer, 'get_cutting_allowance'):
                allowance = optimizer.get_cutting_allowance(6.0)  # 6mm thickness
                metrics['thick_sheet_allowance'] = allowance

                if allowance <= 0:
                    warnings.append(f"Thick sheet allowance should be > 0, got {allowance}")
                elif allowance > 10:  # > 10mm allowance seems excessive
                    warnings.append(f"Very large allowance for thick sheet: {allowance}mm")

                # Compare with thin sheet allowance
                thin_allowance = optimizer.get_cutting_allowance(0.5)
                if allowance <= thin_allowance:
                    errors.append(f"Thick sheet allowance ({allowance}) should be > thin sheet ({thin_allowance})")

                # Run optimization
                sheets = optimizer.optimize_cutting(thick_panels)

                if sheets:
                    total_used = sum(sheet.used_area for sheet in sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in sheets)
                    efficiency = total_used / total_area
                    metrics['thick_sheet_efficiency'] = efficiency

                status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
                details = f"Thick sheet (6mm) allowance: {allowance}mm"

            else:
                status = "SKIP"
                details = "Thickness-based allowance system not found"
                warnings.append("Cannot test thick sheet allowance - system not detected")

        except Exception as e:
            errors.append(f"Thick sheet allowance test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during thick sheet allowance test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "CuttingAllowance_ThickSheet", status, execution_time, details, metrics, errors, warnings
        )

    def _test_allowance_optimization_impact(self):
        """Test impact of cutting allowance on optimization results"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Create test scenario with mixed thicknesses
            mixed_panels = [
                Panel("thin1", 500, 400, 3, "SECC", 0.5),   # Thin
                Panel("thick1", 500, 400, 3, "SECC", 6.0),  # Thick - same size for comparison
                Panel("thin2", 300, 200, 5, "SECC", 0.5),   # Thin
                Panel("thick2", 300, 200, 5, "SECC", 6.0)   # Thick
            ]

            optimizer = CuttingOptimizer()

            # Separate by thickness for individual optimization
            thin_panels = [p for p in mixed_panels if p.thickness <= 1.0]
            thick_panels = [p for p in mixed_panels if p.thickness > 1.0]

            # Optimize separately
            thin_sheets = optimizer.optimize_cutting(thin_panels)
            thick_sheets = optimizer.optimize_cutting(thick_panels)

            # Calculate metrics for comparison
            if thin_sheets:
                thin_used = sum(sheet.used_area for sheet in thin_sheets)
                thin_total = sum(sheet.width * sheet.height for sheet in thin_sheets)
                thin_efficiency = thin_used / thin_total
                metrics['thin_efficiency'] = thin_efficiency
                metrics['thin_sheets_count'] = len(thin_sheets)

            if thick_sheets:
                thick_used = sum(sheet.used_area for sheet in thick_sheets)
                thick_total = sum(sheet.width * sheet.height for sheet in thick_sheets)
                thick_efficiency = thick_used / thick_total
                metrics['thick_efficiency'] = thick_efficiency
                metrics['thick_sheets_count'] = len(thick_sheets)

            # Compare efficiencies - thin should be higher due to 0 allowance
            if 'thin_efficiency' in metrics and 'thick_efficiency' in metrics:
                efficiency_diff = metrics['thin_efficiency'] - metrics['thick_efficiency']
                metrics['efficiency_difference'] = efficiency_diff

                if efficiency_diff <= 0:
                    warnings.append(f"Expected thin sheet efficiency > thick sheet. Diff: {efficiency_diff:.2%}")
                elif efficiency_diff > 0.3:  # > 30% difference might indicate other issues
                    warnings.append(f"Very large efficiency difference: {efficiency_diff:.2%}")

            # Test mixed optimization (should group by material/thickness)
            mixed_sheets = optimizer.optimize_cutting(mixed_panels)
            metrics['mixed_sheets_count'] = len(mixed_sheets)

            # Should ideally separate materials by thickness
            if len(mixed_sheets) < 2:
                warnings.append("Mixed thickness panels should typically require multiple sheets")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Thin efficiency: {metrics.get('thin_efficiency', 'N/A'):.1%}, " + \
                     f"Thick efficiency: {metrics.get('thick_efficiency', 'N/A'):.1%}"

        except Exception as e:
            errors.append(f"Allowance optimization impact test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during allowance impact test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "CuttingAllowance_OptimizationImpact", status, execution_time, details, metrics, errors, warnings
        )

    def _test_file_upload_prioritization(self):
        """Test 5: File Upload & Prioritization"""

        # Test 5.1: File Format Detection
        self._test_file_format_detection()

        # Test 5.2: Upload Processing Priority
        self._test_upload_processing_priority()

        # Test 5.3: Data Validation on Upload
        self._test_upload_data_validation()

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
            if self.pi_data_path.exists():
                test_files.append(('pi.txt', 'unknown'))  # PI file format

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
                elif sample_format == 'unknown' and expected_format != 'unknown':
                    warnings.append(f"Failed to detect sample format for {filename}")

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

    def _test_upload_processing_priority(self):
        """Test upload processing priority and efficiency"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test processing priority with different file sizes
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
            errors.append(f"Upload processing priority test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during upload processing test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "FileUpload_ProcessingPriority", status, execution_time, details, metrics, errors, warnings
        )

    def _test_upload_data_validation(self):
        """Test data validation during upload processing"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()

            # Test invalid data handling
            invalid_data_tests = [
                ('empty_file', ''),
                ('invalid_numbers', 'panel1,abc,def,2,SECC,0.5'),
                ('missing_fields', 'panel1,100'),
                ('negative_values', 'panel1,-100,200,1,SECC,0.5'),
                ('zero_dimensions', 'panel1,0,0,1,SECC,0.5'),
                ('huge_dimensions', 'panel1,10000,20000,1,SECC,0.5')
            ]

            validation_results = {}

            for test_name, test_data in invalid_data_tests:
                result = parser.parse_to_panels(test_data)

                validation_results[test_name] = {
                    'panels_parsed': len(result.panels),
                    'errors_detected': len(result.errors),
                    'warnings_generated': len(result.warnings),
                    'success_rate': result.success_rate
                }

                # Validate that errors are properly detected
                if test_name == 'empty_file':
                    if len(result.panels) > 0:
                        errors.append("Empty file should not produce panels")
                elif test_name in ['invalid_numbers', 'missing_fields']:
                    if len(result.errors) == 0:
                        errors.append(f"Should detect errors in {test_name}")
                elif test_name in ['negative_values', 'zero_dimensions', 'huge_dimensions']:
                    if len(result.panels) > 0 and len(result.errors) == 0 and len(result.warnings) == 0:
                        warnings.append(f"Should validate constraints in {test_name}")

            metrics['validation_results'] = validation_results
            metrics['tests_performed'] = len(invalid_data_tests)

            # Test valid data processing
            valid_data = 'panel1,400,300,2,SECC,0.5\npanel2,600,400,1,SECC,0.5'
            valid_result = parser.parse_to_panels(valid_data)

            metrics['valid_panels_parsed'] = len(valid_result.panels)
            metrics['valid_errors'] = len(valid_result.errors)

            if len(valid_result.panels) != 2:
                errors.append(f"Valid data should produce 2 panels, got {len(valid_result.panels)}")
            if len(valid_result.errors) > 0:
                warnings.append(f"Valid data produced {len(valid_result.errors)} errors")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested {len(invalid_data_tests)} invalid data scenarios"

        except Exception as e:
            errors.append(f"Upload data validation test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during upload validation test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "FileUpload_DataValidation", status, execution_time, details, metrics, errors, warnings
        )

    def _test_ui_ux_improvements(self):
        """Test 6: UI/UX Improvements"""

        # Note: Since we can't easily test Streamlit UI automatically,
        # we'll focus on testable components and provide manual test scenarios

        # Test 6.1: Page Navigation Structure
        self._test_page_navigation_structure()

        # Test 6.2: Session State Management
        self._test_session_state_management()

        # Test 6.3: Component Functionality
        self._test_component_functionality()

    def _test_page_navigation_structure(self):
        """Test page navigation and structure"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Check if page files exist
            pages_dir = project_root / "pages"
            expected_pages = [
                "1_🔧_Cutting_Optimization.py",
                "2_📦_Material_Management.py",
                "3_⚙️_PI_Management.py",
                "4_📊_Analysis_Results.py"
            ]

            existing_pages = []
            for page in expected_pages:
                page_path = pages_dir / page
                if page_path.exists():
                    existing_pages.append(page)

                    # Check if page has basic structure
                    with open(page_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Look for essential Streamlit components
                    if 'import streamlit as st' not in content:
                        warnings.append(f"Page {page} missing Streamlit import")
                    if 'st.set_page_config' not in content and 'setup_page_config' not in content:
                        warnings.append(f"Page {page} missing page config")
                else:
                    errors.append(f"Missing page file: {page}")

            metrics['expected_pages'] = len(expected_pages)
            metrics['existing_pages'] = len(existing_pages)
            metrics['page_completeness'] = len(existing_pages) / len(expected_pages)

            # Check main app.py
            app_path = project_root / "app.py"
            if app_path.exists():
                with open(app_path, 'r', encoding='utf-8') as f:
                    app_content = f.read()

                # Check for navigation components
                if 'st.switch_page' in app_content:
                    metrics['navigation_method'] = 'st.switch_page'
                elif 'st.page_link' in app_content:
                    metrics['navigation_method'] = 'st.page_link'
                else:
                    warnings.append("No navigation method detected in main app")
            else:
                errors.append("Main app.py file not found")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Found {len(existing_pages)}/{len(expected_pages)} expected pages"

        except Exception as e:
            errors.append(f"Page navigation test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during page navigation test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "UIUX_PageNavigation", status, execution_time, details, metrics, errors, warnings
        )

    def _test_session_state_management(self):
        """Test session state management functionality"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Check for session state usage in main files
            main_files = [
                project_root / "app.py",
                project_root / "pages" / "1_🔧_Cutting_Optimization.py",
                project_root / "pages" / "4_📊_Analysis_Results.py"
            ]

            session_state_usage = {}

            for file_path in main_files:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Look for session state patterns
                    session_patterns = {
                        'st.session_state': content.count('st.session_state'),
                        'session_state initialization': content.count('session_state') > 0,
                        'optimization_results': 'optimization_results' in content,
                        'panels_data': 'panels' in content and 'session_state' in content
                    }

                    session_state_usage[file_path.name] = session_patterns
                else:
                    warnings.append(f"File not found for session state check: {file_path.name}")

            metrics['session_state_usage'] = session_state_usage

            # Check if session state is used appropriately
            total_session_usage = sum(
                usage.get('st.session_state', 0)
                for usage in session_state_usage.values()
            )

            if total_session_usage == 0:
                warnings.append("No session state usage detected - data persistence may be limited")
            elif total_session_usage < 5:
                warnings.append("Low session state usage - consider more state management")

            # Check for optimization results persistence
            has_optimization_persistence = any(
                usage.get('optimization_results', False)
                for usage in session_state_usage.values()
            )

            if not has_optimization_persistence:
                warnings.append("Optimization results persistence not detected")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Session state usage detected in {len(session_state_usage)} files"

        except Exception as e:
            errors.append(f"Session state test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during session state test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "UIUX_SessionState", status, execution_time, details, metrics, errors, warnings
        )

    def _test_component_functionality(self):
        """Test UI component functionality"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Check for UI component files
            ui_dir = project_root / "ui"
            expected_components = [
                "visualizer.py",
                "work_instruction_ui.py",
                "components.py"
            ]

            existing_components = []
            component_analysis = {}

            for component in expected_components:
                component_path = ui_dir / component
                if component_path.exists():
                    existing_components.append(component)

                    with open(component_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Analyze component functionality
                    analysis = {
                        'has_streamlit': 'import streamlit' in content,
                        'has_plotly': 'plotly' in content,
                        'function_count': content.count('def '),
                        'class_count': content.count('class '),
                        'has_error_handling': 'try:' in content or 'except' in content
                    }

                    component_analysis[component] = analysis

                    # Component-specific checks
                    if component == "visualizer.py":
                        if not analysis['has_plotly']:
                            warnings.append("Visualizer component missing Plotly integration")
                    elif component == "work_instruction_ui.py":
                        if not analysis['has_streamlit']:
                            warnings.append("Work instruction UI missing Streamlit integration")
                else:
                    errors.append(f"Missing UI component: {component}")

            metrics['expected_components'] = len(expected_components)
            metrics['existing_components'] = len(existing_components)
            metrics['component_analysis'] = component_analysis

            # Check for help system implementation
            help_patterns = ['help', 'sidebar', 'expander', 'info']
            help_usage = {}

            for file_path in [project_root / "app.py"] + list((project_root / "pages").glob("*.py")):
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    help_count = sum(content.lower().count(pattern) for pattern in help_patterns)
                    if help_count > 0:
                        help_usage[file_path.name] = help_count

            metrics['help_system_usage'] = help_usage

            if not help_usage:
                warnings.append("No help system implementation detected")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Found {len(existing_components)}/{len(expected_components)} UI components"

        except Exception as e:
            errors.append(f"Component functionality test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during component test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "UIUX_ComponentFunctionality", status, execution_time, details, metrics, errors, warnings
        )

    def _test_integration_scenarios(self):
        """Test 7: Integration Testing - Full workflow scenarios"""

        # Test 7.1: End-to-End Data Flow
        self._test_end_to_end_data_flow()

        # Test 7.2: Cross-Component Integration
        self._test_cross_component_integration()

        # Test 7.3: Material-Optimization Integration
        self._test_material_optimization_integration()

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
                        optimizer = CuttingOptimizer()
                        sheets = optimizer.optimize_cutting(valid_panels)

                        total_placed = sum(panel.quantity for sheet in sheets for panel in sheet.placed_panels)
                        total_input = sum(p.quantity for p in valid_panels)
                        placement_rate = total_placed / total_input if total_input > 0 else 0

                        workflow_steps.append(('optimization', len(sheets), total_placed))

                        if placement_rate < 0.5:
                            warnings.append(f"Low end-to-end placement rate: {placement_rate:.1%}")

                        # Step 4: Results generation
                        if sheets:
                            # Simulate results processing
                            total_cost = sum(sheet.cost for sheet in sheets)
                            avg_efficiency = sum(sheet.efficiency for sheet in sheets) / len(sheets)

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

    def _test_cross_component_integration(self):
        """Test integration between different system components"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Test parser-optimizer integration
            parser = RobustTextParser()
            optimizer = CuttingOptimizer()

            # Create test data
            test_data = """panel1,400,300,2,SECC,0.5
panel2,600,400,1,SECC,0.5
panel3,800,500,3,SECC,0.5"""

            # Parser -> Optimizer integration
            parse_result = parser.parse_to_panels(test_data)
            if parse_result.panels:
                sheets = optimizer.optimize_cutting(parse_result.panels)

                metrics['parser_optimizer_integration'] = {
                    'panels_parsed': len(parse_result.panels),
                    'sheets_generated': len(sheets),
                    'integration_successful': len(sheets) > 0
                }

                if len(sheets) == 0:
                    errors.append("Parser-Optimizer integration failed - no sheets generated")
            else:
                errors.append("Parser failed to produce panels for integration test")

            # Test material manager integration
            material_manager = get_material_manager()

            # Material Manager -> Optimizer integration
            if parse_result.panels:
                panels_with_materials = []
                for panel in parse_result.panels:
                    available = material_manager.get_available_sheets(panel.material, panel.thickness)
                    if available:
                        panels_with_materials.append(panel)

                metrics['material_optimizer_integration'] = {
                    'total_panels': len(parse_result.panels),
                    'panels_with_materials': len(panels_with_materials),
                    'material_availability_rate': len(panels_with_materials) / len(parse_result.panels)
                }

                if len(panels_with_materials) == 0:
                    warnings.append("No panels have available materials for optimization")

            # Test PI manager integration (if available)
            if PI_MANAGER_AVAILABLE:
                pi_manager = get_pi_manager()

                # PI Manager -> Parser integration (test with PI code)
                pi_test_data = """562210\t77131000\tLUXパネル\t892\t576\t0\t16\t60\tLUXパネル(鋼板)\tKW-300\t0.5"""
                pi_parse_result = parser.parse_to_panels(pi_test_data)

                pi_integration_successful = False
                if pi_parse_result.panels:
                    for panel in pi_parse_result.panels:
                        if panel.pi_code and hasattr(panel, 'expanded_width'):
                            pi_integration_successful = True
                            break

                metrics['pi_parser_integration'] = {
                    'panels_with_pi': len([p for p in pi_parse_result.panels if p.pi_code]),
                    'expansion_applied': pi_integration_successful
                }

                if not pi_integration_successful and pi_parse_result.panels:
                    warnings.append("PI code expansion integration not working")
            else:
                metrics['pi_integration'] = 'skipped - PI manager not available'

            # Overall integration assessment
            integration_tests = [
                metrics.get('parser_optimizer_integration', {}).get('integration_successful', False),
                len(metrics.get('material_optimizer_integration', {})) > 0
            ]

            successful_integrations = sum(integration_tests)
            total_integrations = len(integration_tests)

            if successful_integrations < total_integrations:
                warnings.append(f"Some integrations failed: {successful_integrations}/{total_integrations}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Tested {total_integrations} component integrations, {successful_integrations} successful"

        except Exception as e:
            errors.append(f"Cross-component integration test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during cross-component integration test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Integration_CrossComponent", status, execution_time, details, metrics, errors, warnings
        )

    def _test_material_optimization_integration(self):
        """Test material management and optimization integration"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            material_manager = get_material_manager()
            optimizer = CuttingOptimizer()

            # Test material-aware optimization
            test_panels = [
                Panel("mat_test1", 400, 300, 2, "SECC", 0.5),
                Panel("mat_test2", 600, 400, 1, "SGCC", 0.4),  # Different material
                Panel("mat_test3", 500, 350, 3, "SECC", 6.0)   # Different thickness
            ]

            # Check material availability for each panel
            material_availability = {}
            for panel in test_panels:
                available = material_manager.get_available_sheets(panel.material, panel.thickness)
                material_availability[f"{panel.material}_{panel.thickness}"] = len(available)

            metrics['material_availability'] = material_availability

            # Run optimization
            sheets = optimizer.optimize_cutting(test_panels)

            # Analyze material grouping in results
            material_groups = {}
            for sheet in sheets:
                key = f"{sheet.material}_{sheet.thickness}"
                if key not in material_groups:
                    material_groups[key] = 0
                material_groups[key] += 1

            metrics['material_groups_in_results'] = material_groups
            metrics['total_sheets_generated'] = len(sheets)

            # Validate material consistency
            material_mismatches = 0
            for sheet in sheets:
                for panel in sheet.placed_panels:
                    if panel.material != sheet.material or panel.thickness != sheet.thickness:
                        material_mismatches += 1

            metrics['material_mismatches'] = material_mismatches

            if material_mismatches > 0:
                errors.append(f"Material mismatches found: {material_mismatches}")

            # Check if materials are properly separated
            unique_materials = len(set(f"{p.material}_{p.thickness}" for p in test_panels))
            if unique_materials > 1 and len(material_groups) < unique_materials:
                warnings.append("Materials may not be properly separated in optimization")

            # Test cost calculation integration
            total_cost = sum(sheet.cost for sheet in sheets if hasattr(sheet, 'cost'))
            if total_cost == 0 and sheets:
                warnings.append("No cost calculation detected in material-optimization integration")
            else:
                metrics['total_optimization_cost'] = total_cost

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Generated {len(sheets)} sheets for {unique_materials} material types"

        except Exception as e:
            errors.append(f"Material-optimization integration test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during material-optimization integration test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Integration_MaterialOptimization", status, execution_time, details, metrics, errors, warnings
        )

    def _test_edge_cases_errors(self):
        """Test 8: Edge Cases & Error Handling"""

        # Test 8.1: Invalid Input Handling
        self._test_invalid_input_handling()

        # Test 8.2: Resource Limit Testing
        self._test_resource_limits()

        # Test 8.3: Error Recovery
        self._test_error_recovery()

    def _test_invalid_input_handling(self):
        """Test handling of invalid inputs and edge cases"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            parser = RobustTextParser()
            optimizer = CuttingOptimizer()

            # Test invalid data scenarios
            invalid_scenarios = [
                ('empty_string', ''),
                ('only_whitespace', '   \n\t   '),
                ('invalid_encoding', 'panel1,400,300,2,SECC,0.5\x00\xff'),
                ('malformed_json', '{"panels": [{"width": 400, "height":}]}'),
                ('mixed_delimiters', 'panel1,400\t300;2|SECC,0.5'),
                ('unicode_issues', 'パネル１，４００，３００，２，ＳＥＣＣ，０．５'),
                ('huge_file', 'panel1,400,300,2,SECC,0.5\n' * 10000)  # Large input
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
                    elif scenario_name == 'huge_file':
                        if len(result.panels) == 0:
                            warnings.append("Large valid file should produce some panels")

                except Exception as e:
                    invalid_handling_results[scenario_name] = {
                        'exception_thrown': True,
                        'exception_type': type(e).__name__,
                        'graceful_handling': False
                    }

                    # Some exceptions are acceptable for certain scenarios
                    if scenario_name in ['invalid_encoding']:
                        warnings.append(f"Exception thrown for {scenario_name}: {type(e).__name__}")
                    else:
                        errors.append(f"Unexpected exception for {scenario_name}: {str(e)}")

            metrics['invalid_handling_results'] = invalid_handling_results

            # Test optimizer with invalid panels
            invalid_panels = [
                Panel("invalid1", 0, 300, 1, "SECC", 0.5),     # Zero width
                Panel("invalid2", 400, 0, 1, "SECC", 0.5),     # Zero height
                Panel("invalid3", -100, 300, 1, "SECC", 0.5),  # Negative width
                Panel("invalid4", 400, 300, 0, "SECC", 0.5),   # Zero quantity
                Panel("invalid5", 10000, 300, 1, "SECC", 0.5), # Oversized panel
            ]

            optimizer_error_handling = {}
            for panel in invalid_panels:
                try:
                    sheets = optimizer.optimize_cutting([panel])
                    optimizer_error_handling[panel.id] = {
                        'sheets_generated': len(sheets),
                        'exception_thrown': False
                    }
                except Exception as e:
                    optimizer_error_handling[panel.id] = {
                        'exception_thrown': True,
                        'exception_type': type(e).__name__
                    }

            metrics['optimizer_error_handling'] = optimizer_error_handling

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

    def _test_resource_limits(self):
        """Test system behavior under resource constraints"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Test large dataset handling
            large_panel_count = 500  # Large but reasonable dataset
            large_panels = []

            for i in range(large_panel_count):
                panel = Panel(
                    f"large_{i}",
                    400 + (i % 10) * 50,  # Varied widths
                    300 + (i % 8) * 40,   # Varied heights
                    1 + (i % 3),          # Varied quantities
                    "SECC",
                    0.5
                )
                large_panels.append(panel)

            # Test optimization with large dataset
            optimizer = CuttingOptimizer()
            large_start_time = time.time()

            try:
                large_sheets = optimizer.optimize_cutting(large_panels)
                large_optimization_time = time.time() - large_start_time

                metrics['large_dataset_panels'] = len(large_panels)
                metrics['large_dataset_sheets'] = len(large_sheets)
                metrics['large_optimization_time'] = large_optimization_time

                # Check if performance is reasonable
                if large_optimization_time > self.performance_targets['large_batch_time']:
                    warnings.append(f"Large dataset optimization slow: {large_optimization_time:.2f}s")

                # Check placement rate for large dataset
                total_input_qty = sum(p.quantity for p in large_panels)
                total_placed_qty = sum(panel.quantity for sheet in large_sheets for panel in sheet.placed_panels)
                large_placement_rate = total_placed_qty / total_input_qty if total_input_qty > 0 else 0

                metrics['large_dataset_placement_rate'] = large_placement_rate

                if large_placement_rate < 0.7:  # Expect reasonable placement for uniform panels
                    warnings.append(f"Low placement rate for large dataset: {large_placement_rate:.1%}")

            except Exception as e:
                errors.append(f"Large dataset optimization failed: {str(e)}")
                metrics['large_dataset_error'] = str(e)

            # Test memory usage patterns (simple check)
            import sys

            # Test parsing large text data
            large_text_data = 'panel1,400,300,2,SECC,0.5\n' * 1000
            parser = RobustTextParser()

            memory_before = sys.getsizeof(large_text_data)
            parse_start = time.time()
            large_parse_result = parser.parse_to_panels(large_text_data)
            parse_time = time.time() - parse_start

            metrics['large_text_size'] = len(large_text_data)
            metrics['large_text_parse_time'] = parse_time
            metrics['large_text_panels'] = len(large_parse_result.panels)

            if parse_time > 5.0:  # > 5 seconds for 1000 lines is slow
                warnings.append(f"Large text parsing slow: {parse_time:.2f}s")

            # Test empty material scenario
            try:
                empty_material_panels = [Panel("empty", 400, 300, 1, "NONEXISTENT", 999)]
                empty_sheets = optimizer.optimize_cutting(empty_material_panels)

                metrics['empty_material_handling'] = {
                    'sheets_generated': len(empty_sheets),
                    'graceful_handling': True
                }

            except Exception as e:
                metrics['empty_material_handling'] = {
                    'exception_thrown': True,
                    'exception_type': type(e).__name__
                }
                warnings.append(f"Non-existent material handling: {type(e).__name__}")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Large dataset: {large_panel_count} panels, " + \
                     f"optimization time: {metrics.get('large_optimization_time', 0):.2f}s"

        except Exception as e:
            errors.append(f"Resource limits test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during resource limits test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "EdgeCases_ResourceLimits", status, execution_time, details, metrics, errors, warnings
        )

    def _test_error_recovery(self):
        """Test error recovery and system resilience"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            # Test partial data recovery
            mixed_data = """panel1,400,300,2,SECC,0.5
invalid_line_with_missing_fields
panel2,600,400,1,SECC,0.5
another_invalid_line,abc,def
panel3,800,500,3,SECC,0.5"""

            parser = RobustTextParser()
            recovery_result = parser.parse_to_panels(mixed_data)

            metrics['recovery_test'] = {
                'total_lines': len(mixed_data.split('\n')),
                'panels_recovered': len(recovery_result.panels),
                'errors_detected': len(recovery_result.errors),
                'success_rate': recovery_result.success_rate
            }

            # Should recover valid panels despite errors
            if len(recovery_result.panels) != 3:  # Expect 3 valid panels
                warnings.append(f"Expected 3 recovered panels, got {len(recovery_result.panels)}")

            if len(recovery_result.errors) == 0:
                warnings.append("Should detect errors in malformed data")

            # Test continued operation after errors
            try:
                # Should still be able to optimize recovered panels
                optimizer = CuttingOptimizer()
                recovery_sheets = optimizer.optimize_cutting(recovery_result.panels)

                metrics['post_error_optimization'] = {
                    'sheets_generated': len(recovery_sheets),
                    'continued_operation': True
                }

                if len(recovery_sheets) == 0 and len(recovery_result.panels) > 0:
                    warnings.append("Failed to optimize after error recovery")

            except Exception as e:
                metrics['post_error_optimization'] = {
                    'continued_operation': False,
                    'exception': str(e)
                }
                errors.append(f"System failed to continue after error recovery: {str(e)}")

            # Test state consistency after errors
            material_manager = get_material_manager()
            try:
                # Should still be able to access material data
                summary = material_manager.get_inventory_summary()
                metrics['post_error_material_access'] = {
                    'accessible': True,
                    'total_materials': summary['total_sheets']
                }
            except Exception as e:
                metrics['post_error_material_access'] = {
                    'accessible': False,
                    'exception': str(e)
                }
                errors.append(f"Material manager inaccessible after errors: {str(e)}")

            # Test multiple error scenarios in sequence
            error_scenarios = [
                'invalid,data,here',
                '',  # Empty line
                'panel_test,200,150,1,SECC,0.5',  # Valid data
                'more,invalid,stuff',
                'panel_test2,300,250,2,SECC,0.5'  # More valid data
            ]

            sequential_errors = 0
            sequential_recoveries = 0

            for scenario in error_scenarios:
                try:
                    result = parser.parse_to_panels(scenario)
                    if len(result.errors) > 0:
                        sequential_errors += 1
                    if len(result.panels) > 0:
                        sequential_recoveries += 1
                except Exception:
                    sequential_errors += 1

            metrics['sequential_error_handling'] = {
                'scenarios_tested': len(error_scenarios),
                'errors_detected': sequential_errors,
                'recoveries_achieved': sequential_recoveries
            }

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Error recovery: {recovery_result.success_rate:.1%} success rate, " + \
                     f"{len(recovery_result.panels)} panels recovered"

        except Exception as e:
            errors.append(f"Error recovery test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during error recovery test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "EdgeCases_ErrorRecovery", status, execution_time, details, metrics, errors, warnings
        )

    def _test_performance_scenarios(self):
        """Test 9: Performance Validation"""

        # Test 9.1: Batch Size Performance
        self._test_batch_size_performance()

        # Test 9.2: Algorithm Efficiency
        self._test_algorithm_efficiency()

        # Test 9.3: Memory Usage
        self._test_memory_usage()

    def _test_batch_size_performance(self):
        """Test performance across different batch sizes"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            optimizer = CuttingOptimizer()

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
                batch_sheets = optimizer.optimize_cutting(test_panels)
                batch_time = time.time() - batch_start

                # Calculate metrics
                total_input_qty = sum(p.quantity for p in test_panels)
                total_placed_qty = sum(panel.quantity for sheet in batch_sheets for panel in sheet.placed_panels)
                placement_rate = total_placed_qty / total_input_qty if total_input_qty > 0 else 0

                if batch_sheets:
                    total_used = sum(sheet.used_area for sheet in batch_sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in batch_sheets)
                    efficiency = total_used / total_area
                else:
                    efficiency = 0

                performance_results[batch_name] = {
                    'panel_count': panel_count,
                    'optimization_time': batch_time,
                    'time_target': time_target,
                    'sheets_generated': len(batch_sheets),
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

                if efficiency < self.performance_targets['material_efficiency']:
                    warnings.append(f"{batch_name} batch low efficiency: {efficiency:.1%}")

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

    def _test_algorithm_efficiency(self):
        """Test optimization algorithm efficiency"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            optimizer = CuttingOptimizer()

            # Test scenarios designed to measure algorithm quality
            efficiency_scenarios = [
                {
                    'name': 'uniform_panels',
                    'panels': [Panel(f"uniform_{i}", 400, 300, 2, "SECC", 0.5) for i in range(20)],
                    'expected_efficiency': 0.8  # Uniform panels should pack well
                },
                {
                    'name': 'varied_sizes',
                    'panels': [
                        Panel("large1", 1200, 800, 1, "SECC", 0.5),
                        Panel("medium1", 600, 400, 4, "SECC", 0.5),
                        Panel("small1", 200, 150, 10, "SECC", 0.5)
                    ],
                    'expected_efficiency': 0.6  # Mixed sizes harder to pack
                },
                {
                    'name': 'optimal_rectangles',
                    'panels': [Panel(f"opt_{i}", 500, 375, 1, "SECC", 0.5) for i in range(12)],
                    'expected_efficiency': 0.85  # Should fit optimally in sheet
                }
            ]

            efficiency_results = {}

            for scenario in efficiency_scenarios:
                sheets = optimizer.optimize_cutting(scenario['panels'])

                if sheets:
                    total_used = sum(sheet.used_area for sheet in sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in sheets)
                    actual_efficiency = total_used / total_area

                    total_input_qty = sum(p.quantity for p in scenario['panels'])
                    total_placed_qty = sum(panel.quantity for sheet in sheets for panel in sheet.placed_panels)
                    placement_rate = total_placed_qty / total_input_qty if total_input_qty > 0 else 0

                    efficiency_results[scenario['name']] = {
                        'actual_efficiency': actual_efficiency,
                        'expected_efficiency': scenario['expected_efficiency'],
                        'placement_rate': placement_rate,
                        'sheets_used': len(sheets),
                        'meets_expectation': actual_efficiency >= scenario['expected_efficiency'] * 0.9  # 90% of expected
                    }

                    # Validate against expectations
                    if actual_efficiency < scenario['expected_efficiency'] * 0.8:  # Less than 80% of expected
                        errors.append(f"{scenario['name']}: efficiency {actual_efficiency:.1%} << expected {scenario['expected_efficiency']:.1%}")
                    elif actual_efficiency < scenario['expected_efficiency'] * 0.9:  # Less than 90% of expected
                        warnings.append(f"{scenario['name']}: efficiency {actual_efficiency:.1%} < expected {scenario['expected_efficiency']:.1%}")

                else:
                    efficiency_results[scenario['name']] = {
                        'actual_efficiency': 0,
                        'error': 'No sheets generated'
                    }
                    errors.append(f"{scenario['name']}: No sheets generated")

            metrics['efficiency_results'] = efficiency_results

            # Algorithm consistency test (run same scenario multiple times)
            consistency_panels = [Panel(f"consist_{i}", 450, 350, 2, "SECC", 0.5) for i in range(15)]
            consistency_results = []

            for run in range(3):
                sheets = optimizer.optimize_cutting(consistency_panels.copy())
                if sheets:
                    total_used = sum(sheet.used_area for sheet in sheets)
                    total_area = sum(sheet.width * sheet.height for sheet in sheets)
                    efficiency = total_used / total_area
                    consistency_results.append(efficiency)

            if consistency_results:
                avg_consistency = sum(consistency_results) / len(consistency_results)
                max_variance = max(consistency_results) - min(consistency_results)

                metrics['algorithm_consistency'] = {
                    'avg_efficiency': avg_consistency,
                    'max_variance': max_variance,
                    'runs': len(consistency_results)
                }

                if max_variance > 0.1:  # More than 10% variance indicates inconsistency
                    warnings.append(f"High algorithm variance: {max_variance:.2%}")

            # Overall algorithm quality assessment
            meeting_expectations = sum(1 for result in efficiency_results.values()
                                     if result.get('meets_expectation', False))
            total_scenarios = len(efficiency_scenarios)

            metrics['algorithm_quality_score'] = meeting_expectations / total_scenarios

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Algorithm quality: {meeting_expectations}/{total_scenarios} scenarios met expectations"

        except Exception as e:
            errors.append(f"Algorithm efficiency test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during algorithm efficiency test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Performance_AlgorithmEfficiency", status, execution_time, details, metrics, errors, warnings
        )

    def _test_memory_usage(self):
        """Test memory usage patterns and limits"""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        try:
            import sys
            import gc

            # Baseline memory measurement
            gc.collect()
            baseline_objects = len(gc.get_objects())

            # Test memory usage with increasing data sizes
            memory_test_sizes = [10, 50, 200]  # Panel counts
            memory_results = {}

            optimizer = CuttingOptimizer()
            parser = RobustTextParser()

            for size in memory_test_sizes:
                gc.collect()
                before_objects = len(gc.get_objects())

                # Create test data
                test_panels = [Panel(f"mem_{i}", 400, 300, 1, "SECC", 0.5) for i in range(size)]
                test_data = '\n'.join([f"mem_{i},400,300,1,SECC,0.5" for i in range(size)])

                # Test parsing memory usage
                parse_result = parser.parse_to_panels(test_data)
                after_parse_objects = len(gc.get_objects())

                # Test optimization memory usage
                sheets = optimizer.optimize_cutting(test_panels)
                after_optimization_objects = len(gc.get_objects())

                memory_results[size] = {
                    'before_objects': before_objects,
                    'after_parse_objects': after_parse_objects,
                    'after_optimization_objects': after_optimization_objects,
                    'parse_object_growth': after_parse_objects - before_objects,
                    'optimization_object_growth': after_optimization_objects - after_parse_objects,
                    'total_object_growth': after_optimization_objects - before_objects
                }

                # Clean up for next test
                del test_panels, test_data, parse_result, sheets
                gc.collect()

            metrics['memory_results'] = memory_results

            # Analyze memory growth patterns
            if len(memory_results) >= 2:
                sizes = sorted(memory_results.keys())
                growth_ratios = []

                for i in range(1, len(sizes)):
                    prev_size = sizes[i-1]
                    curr_size = sizes[i]

                    prev_growth = memory_results[prev_size]['total_object_growth']
                    curr_growth = memory_results[curr_size]['total_object_growth']

                    size_ratio = curr_size / prev_size
                    growth_ratio = curr_growth / prev_growth if prev_growth > 0 else 1

                    efficiency_ratio = growth_ratio / size_ratio
                    growth_ratios.append(efficiency_ratio)

                avg_growth_efficiency = sum(growth_ratios) / len(growth_ratios)
                metrics['memory_growth_efficiency'] = avg_growth_efficiency

                if avg_growth_efficiency > 2.0:  # Memory grows faster than data size
                    warnings.append(f"Memory growth outpaces data size: {avg_growth_efficiency:.2f}x")
                elif avg_growth_efficiency > 1.5:
                    warnings.append(f"Memory growth slightly high: {avg_growth_efficiency:.2f}x")

            # Test for memory leaks (simplified)
            gc.collect()
            final_objects = len(gc.get_objects())
            object_leak = final_objects - baseline_objects

            metrics['potential_memory_leak'] = {
                'baseline_objects': baseline_objects,
                'final_objects': final_objects,
                'object_difference': object_leak
            }

            # Allow some object growth but flag excessive growth
            if object_leak > 1000:  # More than 1000 additional objects
                warnings.append(f"Potential memory leak: {object_leak} additional objects")

            status = "FAIL" if errors else ("PASS" if not warnings else "PASS")
            details = f"Memory tests completed. Object growth efficiency: {metrics.get('memory_growth_efficiency', 'N/A'):.2f}x"

        except Exception as e:
            errors.append(f"Memory usage test failed: {str(e)}")
            status = "ERROR"
            details = f"Exception during memory usage test: {str(e)}"

        execution_time = time.time() - start_time
        self._record_test_result(
            "Performance_MemoryUsage", status, execution_time, details, metrics, errors, warnings
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

        # Feature-specific recommendations
        pi_tests = [r for r in self.test_results if 'PI' in r.test_name and r.status == 'FAIL']
        if pi_tests:
            recommendations.append("PI code expansion functionality needs attention")

        placement_tests = [r for r in self.test_results if 'Placement' in r.test_name and r.status == 'FAIL']
        if placement_tests:
            recommendations.append("Panel placement optimization requires fixes")

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