#!/usr/bin/env python3
"""
Debug script to analyze unplaced panels
配置されていないパネルの詳細分析
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.ffd import FirstFitDecreasing
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
from core.material_manager import MaterialInventoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_unplaced_panels():
    """Debug why panels cannot be placed"""
    try:
        logger.info("配置できないパネルの詳細分析開始...")

        # Parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"パネル種類数: {len(panels)}")

        # Apply PI expansion
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"総パネル数: {total_panels}")

        # Get material sheets
        material_manager = MaterialInventoryManager()
        material_sheets = material_manager.get_all_sheets()

        # Analyze each material group
        material_groups = {}
        for panel in panels:
            material = panel.material
            if material not in material_groups:
                material_groups[material] = []
            material_groups[material].append(panel)

        logger.info(f"材料グループ数: {len(material_groups)}")

        # Check each material
        for material, material_panels in material_groups.items():
            logger.info(f"\n=== 材料: {material} ===")

            # Get available sheets for this material
            available_sheets = [sheet for sheet in material_sheets if sheet.material == material]
            if not available_sheets:
                logger.warning(f"材料 {material} のシートが見つかりません!")
                continue

            sheet = available_sheets[0]  # Use first available sheet
            logger.info(f"使用シート: {sheet.width}x{sheet.height}mm ({sheet.material})")

            # Analyze each panel
            panel_count = sum(panel.quantity for panel in material_panels)
            logger.info(f"パネル数: {panel_count}")

            for panel in material_panels:
                # Check if panel can fit in sheet
                can_fit_normal = (panel.cutting_width <= sheet.width and panel.cutting_height <= sheet.height)
                can_fit_rotated = (panel.cutting_height <= sheet.width and panel.cutting_width <= sheet.height)

                if not can_fit_normal and not can_fit_rotated:
                    logger.error(f"  ❌ パネル {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm - シートサイズを超過!")
                elif (not can_fit_normal) and can_fit_rotated:
                    logger.warning(f"  🔄 パネル {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm - 回転が必要")
                else:
                    logger.info(f"  ✅ パネル {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm - 配置可能")

        # Run FFD to see what happens
        logger.info("\n=== FFD最適化実行 ===")
        constraints = OptimizationConstraints(
            material_separation=False,
            max_sheets=1000,
            time_budget=0.0,
            kerf_width=0.0
        )

        engine = OptimizationEngine()
        ffd_algorithm = FirstFitDecreasing()
        engine.register_algorithm(ffd_algorithm)

        results = engine.optimize(panels, constraints, 'FFD')
        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100

        logger.info(f"FFD結果: {total_placed}/{total_panels} ({placement_rate:.1f}%)")
        logger.info(f"使用シート数: {len(results)}")

        # Analyze remaining unplaced panels
        placed_panel_ids = set()
        for result in results:
            for placed_panel in result.panels:
                placed_panel_ids.add(placed_panel.panel_id)

        logger.info("\n=== 配置されなかったパネル ===")
        unplaced_count = 0
        for panel in panels:
            for i in range(panel.quantity):
                panel_instance_id = f"{panel.id}_{i+1}"
                if panel_instance_id not in placed_panel_ids:
                    unplaced_count += 1
                    logger.warning(f"未配置: {panel.id} ({panel.cutting_width}x{panel.cutting_height}mm, {panel.material})")
                    if unplaced_count >= 20:  # Limit output
                        remaining = total_panels - total_placed - unplaced_count
                        if remaining > 0:
                            logger.warning(f"... さらに {remaining} 個のパネルが未配置")
                        break
            if unplaced_count >= 20:
                break

        return total_placed == total_panels

    except Exception as e:
        logger.error(f"デバッグエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_unplaced_panels()