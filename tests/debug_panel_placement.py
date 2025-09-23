#!/usr/bin/env python3
"""
Debug script to analyze why specific panels cannot be placed
特定のパネルが配置できない理由を分析するデバッグスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.material_manager import get_material_manager
from core.pi_manager import PIManager
from core.models import Panel, SteelSheet, OptimizationConstraints
from core.algorithms.ffd import FirstFitDecreasing, GuillotineBinPacker
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_single_panel_placement():
    """Test placing a single problematic panel"""
    try:
        # Load and parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels

        # Find the problematic panel
        problem_panel = None
        for panel in panels:
            if "686632" in panel.id:  # Find the large panel that fails
                problem_panel = panel
                break

        if not problem_panel:
            logger.error("Could not find problematic panel")
            return

        # Apply PI expansion
        pi_manager = PIManager()
        problem_panel.calculate_expanded_dimensions(pi_manager)

        logger.info(f"Testing panel: {problem_panel.id}")
        logger.info(f"  Original size: {problem_panel.width}x{problem_panel.height}mm")
        logger.info(f"  Cutting size: {problem_panel.cutting_width:.1f}x{problem_panel.cutting_height:.1f}mm")
        logger.info(f"  Quantity: {problem_panel.quantity}")
        logger.info(f"  Material: {problem_panel.material}")

        # Get material manager and find suitable sheets
        material_manager = get_material_manager()

        # Create test sheet - largest available
        test_sheet = SteelSheet(
            width=1524.0,
            height=3048.0,
            thickness=problem_panel.thickness,
            material=problem_panel.material,
            cost_per_sheet=15000.0,
            availability=999
        )

        logger.info(f"Test sheet: {test_sheet.width}x{test_sheet.height}mm")
        logger.info(f"Can panel fit? {problem_panel.fits_in_sheet(test_sheet.width, test_sheet.height)}")

        # Test direct placement
        constraints = OptimizationConstraints(kerf_width=0.0)
        packer = GuillotineBinPacker(test_sheet.width, test_sheet.height, constraints.kerf_width)

        # Try to place the panel
        success = packer.place_panel(problem_panel)
        logger.info(f"Direct placement result: {success}")

        if not success:
            # Analyze why it failed
            logger.info("Analyzing failure reasons:")
            logger.info(f"  Panel cutting dimensions: {problem_panel.cutting_width:.1f} x {problem_panel.cutting_height:.1f}")
            logger.info(f"  Sheet dimensions: {test_sheet.width} x {test_sheet.height}")
            logger.info(f"  Panel fits without rotation: {problem_panel.cutting_width <= test_sheet.width and problem_panel.cutting_height <= test_sheet.height}")
            logger.info(f"  Panel fits with rotation: {problem_panel.cutting_height <= test_sheet.width and problem_panel.cutting_width <= test_sheet.height}")
            logger.info(f"  Allow rotation: {problem_panel.allow_rotation}")

            # Check free rectangles
            logger.info(f"Free rectangles: {len(packer.free_rectangles)}")
            for i, rect in enumerate(packer.free_rectangles):
                logger.info(f"    Rect {i}: {rect.x:.1f},{rect.y:.1f} {rect.width:.1f}x{rect.height:.1f}")
        else:
            logger.info("Panel placed successfully!")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_panel_placement()