#!/usr/bin/env python3
"""
Debug material mapping and sheet selection for problematic panels
問題のあるパネルの材料マッピングとシート選択をデバッグ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.material_manager import get_material_manager
from core.pi_manager import PIManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_material_mapping():
    """Debug material mapping for problematic panels"""
    try:
        # Load and parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels

        # Get material manager
        material_manager = get_material_manager()

        logger.info("Material mapping analysis:")
        logger.info("==========================")

        # Find problematic panels
        problem_panels = []
        for panel in panels:
            if "686632" in panel.id or "562210" in panel.id:
                problem_panels.append(panel)

        for panel in problem_panels:
            logger.info(f"\\nPanel: {panel.id}")
            logger.info(f"  Material: {panel.material}")
            logger.info(f"  Thickness: {panel.thickness}mm")
            logger.info(f"  Size: {panel.width}x{panel.height}mm")

            # Check material normalization
            normalized = material_manager.normalize_material_code(panel.material)
            logger.info(f"  Normalized material: {normalized}")

            # Find available sheets
            available_sheets = material_manager.get_sheets_by_type(normalized)
            logger.info(f"  Available sheets for {normalized}: {len(available_sheets)}")

            if available_sheets:
                # Sort by area to show largest first
                sorted_sheets = sorted(available_sheets, key=lambda s: s.width * s.height, reverse=True)
                logger.info(f"  Largest 5 sheets:")
                for i, sheet in enumerate(sorted_sheets[:5]):
                    area = sheet.width * sheet.height
                    logger.info(f"    {i+1}. {sheet.width:.0f}x{sheet.height:.0f}mm (area: {area:.0f}, avail: {sheet.availability})")

                # Check if panel fits in largest sheet
                largest = sorted_sheets[0]
                fits_normal = panel.width <= largest.width and panel.height <= largest.height
                fits_rotated = panel.height <= largest.width and panel.width <= largest.height
                logger.info(f"  Fits in largest sheet:")
                logger.info(f"    Normal orientation: {fits_normal}")
                logger.info(f"    Rotated: {fits_rotated}")
                logger.info(f"    Allow rotation: {panel.allow_rotation}")

        # Check material groups
        logger.info("\\n\\nMaterial grouping:")
        logger.info("==================")
        material_groups = {}
        for panel in panels:
            normalized = material_manager.normalize_material_code(panel.material)
            if normalized not in material_groups:
                material_groups[normalized] = []
            material_groups[normalized].append(panel)

        for material, group_panels in material_groups.items():
            total_quantity = sum(p.quantity for p in group_panels)
            logger.info(f"\\n{material}: {len(group_panels)} panel types, {total_quantity} total pieces")

            # Show large panels in this group
            large_panels = [p for p in group_panels if p.width * p.height > 1000000]  # > 1m²
            if large_panels:
                logger.info(f"  Large panels (>1m²):")
                for p in large_panels:
                    logger.info(f"    {p.id}: {p.width}x{p.height}mm, qty: {p.quantity}")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_material_mapping()