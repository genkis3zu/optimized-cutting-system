#!/usr/bin/env python3
"""
Test script to verify Streamlit app behavior matches test environment
Streamlitアプリの動作がテスト環境と一致することを検証するテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.bfd import BestFitDecreasing
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_streamlit_equivalent():
    """Test exact same logic as Streamlit app"""
    try:
        logger.info("Testing Streamlit app equivalent behavior...")

        # Parse data (same as Streamlit)
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"Parsed {len(panels)} panel types")

        # Apply PI expansion (same as PanelInputComponent)
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"Total individual panels: {total_panels}")

        # Create constraints exactly as Streamlit UI (after our modifications)
        constraints = OptimizationConstraints(
            kerf_width=0.0,  # Force thin sheet cutting (test environment setting)
            allow_rotation=True,  # User preference
            material_separation=False,  # Force multi-sheet optimization (test environment setting)
            time_budget=0.0,  # Force no time limit - run until completion (test environment setting)
            target_efficiency=0.75,  # Default from UI
            max_sheets=1000  # Allow unlimited sheets for complete placement (test environment setting)
        )

        logger.info(f"Constraints: kerf={constraints.kerf_width}, time_budget={constraints.time_budget}, max_sheets={constraints.max_sheets}, material_separation={constraints.material_separation}")

        # Create engine exactly like test_all_placement.py (67.7% placement rate)
        from core.optimizer import OptimizationEngine
        engine = OptimizationEngine()  # Empty engine like test_all_placement.py

        # Register algorithms exactly like test_all_placement.py (67.7% placement rate)
        ffd_algorithm = FirstFitDecreasing()
        bfd_algorithm = BestFitDecreasing()

        engine.register_algorithm(ffd_algorithm)
        engine.register_algorithm(bfd_algorithm)

        # Force FFD exactly as Streamlit (after our modifications)
        algorithm_hint = 'FFD'
        logger.info(f"Using algorithm: {algorithm_hint}")

        # Run optimization exactly like test_all_placement.py (67.7% placement rate)
        results = engine.optimize(panels, constraints, algorithm_hint)

        # Calculate results
        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100 if total_panels > 0 else 0

        logger.info(f"Streamlit Equivalent Results:")
        logger.info(f"  Sheets used: {len(results)}")
        logger.info(f"  Panels placed: {total_placed}/{total_panels} ({placement_rate:.1f}%)")

        if placement_rate >= 67.0:
            logger.info("✅ SUCCESS: Streamlit equivalent achieves expected placement rate!")
            return True
        else:
            logger.warning(f"❌ FAILURE: Streamlit equivalent only achieves {placement_rate:.1f}% placement rate")
            return False

    except Exception as e:
        logger.error(f"Streamlit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streamlit_equivalent()
    sys.exit(0 if success else 1)