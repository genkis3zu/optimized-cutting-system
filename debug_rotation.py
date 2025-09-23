#!/usr/bin/env python3
"""
Debug rotation and panel placement for specific problematic panel
特定の問題パネルの回転と配置をデバッグ
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

def test_rotation_placement():
    """Test rotation for problematic KW90 panels"""
    try:
        # Load and parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels

        # Find the problematic KW90 panel (1344x698mm - largest)
        problem_panel = None
        for panel in panels:
            if "561816" in panel.id:  # Find the largest KW90 panel
                problem_panel = panel
                break

        if not problem_panel:
            logger.error("Could not find problematic KW90 panel")
            return

        # Apply PI expansion
        pi_manager = PIManager()
        problem_panel.calculate_expanded_dimensions(pi_manager)

        logger.info(f"Testing panel: {problem_panel.id}")
        logger.info(f"  Original size: {problem_panel.width}x{problem_panel.height}mm")
        logger.info(f"  Cutting size: {problem_panel.cutting_width:.1f}x{problem_panel.cutting_height:.1f}mm")
        logger.info(f"  Material: {problem_panel.material}")
        logger.info(f"  Allow rotation: {problem_panel.allow_rotation}")

        # Get largest KW90 sheet
        material_manager = get_material_manager()
        kw90_sheets = material_manager.get_sheets_by_type('KW90')
        largest_sheet = max(kw90_sheets, key=lambda s: s.width * s.height)

        logger.info(f"Largest KW90 sheet: {largest_sheet.width:.0f}x{largest_sheet.height:.0f}mm")

        # Test fits_in_sheet method
        fits = problem_panel.fits_in_sheet(largest_sheet.width, largest_sheet.height)
        logger.info(f"Panel fits in sheet (using fits_in_sheet): {fits}")

        # Test manual fit check
        normal_fit = (problem_panel.cutting_width <= largest_sheet.width and
                     problem_panel.cutting_height <= largest_sheet.height)
        rotated_fit = (problem_panel.cutting_height <= largest_sheet.width and
                      problem_panel.cutting_width <= largest_sheet.height)

        logger.info(f"Manual fit check:")
        logger.info(f"  Normal orientation ({problem_panel.cutting_width:.1f}x{problem_panel.cutting_height:.1f}): {normal_fit}")
        logger.info(f"  Rotated ({problem_panel.cutting_height:.1f}x{problem_panel.cutting_width:.1f}): {rotated_fit}")

        # Test direct placement with empty sheet
        test_sheet = SteelSheet(
            width=largest_sheet.width,
            height=largest_sheet.height,
            thickness=problem_panel.thickness,
            material="KW90",
            cost_per_sheet=15000.0,
            availability=999
        )

        constraints = OptimizationConstraints(kerf_width=0.0)
        packer = GuillotineBinPacker(test_sheet.width, test_sheet.height, constraints.kerf_width)

        # Try to place the panel
        success = packer.place_panel(problem_panel)
        logger.info(f"Direct placement in empty sheet: {success}")

        if success:
            placed = packer.placed_panels[0]
            logger.info(f"Placed at: ({placed.x:.1f}, {placed.y:.1f})")
            logger.info(f"Rotated: {placed.rotated}")
            logger.info(f"Actual size: {placed.actual_width:.1f}x{placed.actual_height:.1f}mm")

        # Test with FFD algorithm directly
        ffd = FirstFitDecreasing()
        result = ffd.optimize([problem_panel], test_sheet, constraints)
        logger.info(f"FFD single panel result: {len(result.panels)} panels placed")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rotation_placement()