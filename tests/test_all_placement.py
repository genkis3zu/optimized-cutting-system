#!/usr/bin/env python3
"""
Test script to verify all panels in data0923.txt can be placed
すべてのパネルが配置できることを検証するテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.material_manager import get_material_manager
from core.pi_manager import PIManager
from core.models import OptimizationConstraints
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.bfd import BestFitDecreasing
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_placement():
    """Test placing all panels from data0923.txt"""
    try:
        # Load and parse data
        logger.info("Loading data0923.txt...")
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"Parsed {len(panels)} panel types")

        # Total individual panels
        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"Total individual panels: {total_panels}")

        # Get global material manager
        material_manager = get_material_manager()

        # Create PI manager with default expansion rules
        pi_manager = PIManager()

        # Apply PI code expansion
        expanded_panels = []
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)
            expanded_panels.append(panel)

            # Log expansion info
            if panel.has_expansion():
                exp_info = panel.get_expansion_info()
                logger.debug(f"Panel {panel.id}: {panel.width}x{panel.height} -> {panel.cutting_width:.1f}x{panel.cutting_height:.1f} (+{exp_info['w_expansion']:.1f}, +{exp_info['h_expansion']:.1f})")

        logger.info(f"Applied PI expansion to {len(expanded_panels)} panel types")

        # Create unlimited constraints
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,  # Allow many sheets
            time_budget=0.0,  # No time limit
            kerf_width=0.0   # Thin sheet cutting
        )

        logger.info(f"Starting optimization with constraints: max_sheets={constraints.max_sheets}, time_budget={constraints.time_budget}, kerf_width={constraints.kerf_width}")

        # Create optimizer and register algorithms
        optimizer = OptimizationEngine()
        optimizer.register_algorithm(FirstFitDecreasing())
        optimizer.register_algorithm(BestFitDecreasing())

        logger.info("Testing FFD algorithm...")
        results = optimizer.optimize(expanded_panels, constraints, 'FFD')
        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100 if total_panels > 0 else 0

        logger.info(f"FFD Results:")
        logger.info(f"  Sheets used: {len(results)}")
        logger.info(f"  Panels placed: {total_placed}/{total_panels} ({placement_rate:.1f}%)")

        if placement_rate >= 100.0:
            logger.info("✅ SUCCESS: All panels placed with FFD!")
            return True
        else:
            logger.warning(f"❌ FAILURE: Only {placement_rate:.1f}% panels placed with FFD")

            # Log which panels couldn't be placed
            placed_panel_ids = set()
            for result in results:
                for placed_panel in result.panels:
                    placed_panel_ids.add(placed_panel.panel.id)

            # Find unplaced panels
            unplaced_panels = []
            for panel in expanded_panels:
                placed_count = sum(1 for id in placed_panel_ids if id.startswith(panel.id))
                if placed_count < panel.quantity:
                    unplaced_count = panel.quantity - placed_count
                    unplaced_panels.append((panel, unplaced_count))

            logger.warning(f"Unplaced panel types: {len(unplaced_panels)}")
            for panel, count in unplaced_panels[:10]:  # Show first 10
                logger.warning(f"  {panel.id}: {count} pieces ({panel.cutting_width:.1f}x{panel.cutting_height:.1f}mm)")

            return False

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_placement()
    sys.exit(0 if success else 1)