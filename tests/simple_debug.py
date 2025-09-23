#!/usr/bin/env python3
"""
Simple debug script to find the root cause
単純デバッグスクリプト：根本原因を発見
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.pi_manager import PIManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_panel_sizes():
    """Analyze panel sizes vs sheet sizes"""
    try:
        logger.info("パネルサイズとシートサイズの分析開始...")

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

        # Standard sheet sizes (from the system)
        sheet_sizes = {
            'KW300': (1268, 3020),
            'KW90': (1268, 3030),
            'SECC': (1500, 3100),
            'KW400': (1268, 3020)
        }

        logger.info("\n=== シートサイズ ===")
        for material, (width, height) in sheet_sizes.items():
            logger.info(f"{material}: {width}x{height}mm")

        # Group by material
        material_groups = {}
        for panel in panels:
            material = panel.material
            if material not in material_groups:
                material_groups[material] = []
            material_groups[material].append(panel)

        # Check each material group
        total_oversized = 0
        for material, material_panels in material_groups.items():
            logger.info(f"\n=== 材料: {material} ===")

            if material in sheet_sizes:
                sheet_width, sheet_height = sheet_sizes[material]
                logger.info(f"シートサイズ: {sheet_width}x{sheet_height}mm")
            else:
                logger.warning(f"未知の材料: {material}")
                continue

            oversized_panels = 0
            for panel in material_panels:
                # Check if panel fits (with rotation)
                can_fit_normal = (panel.cutting_width <= sheet_width and panel.cutting_height <= sheet_height)
                can_fit_rotated = (panel.cutting_height <= sheet_width and panel.cutting_width <= sheet_height)

                if not can_fit_normal and not can_fit_rotated:
                    logger.error(f"  ❌ {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm (数量:{panel.quantity}) - シートサイズ超過!")
                    oversized_panels += panel.quantity
                    total_oversized += panel.quantity
                elif can_fit_normal and can_fit_rotated:
                    logger.info(f"  ✅ {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm (数量:{panel.quantity}) - 配置可能")
                elif can_fit_rotated:
                    logger.info(f"  🔄 {panel.id}: {panel.cutting_width}x{panel.cutting_height}mm (数量:{panel.quantity}) - 回転で配置可能")

            if oversized_panels > 0:
                logger.error(f"材料 {material}: {oversized_panels} 個のパネルがシートサイズを超過")

        logger.info(f"\n=== 結果サマリー ===")
        logger.info(f"総パネル数: {total_panels}")
        logger.info(f"シートサイズ超過パネル数: {total_oversized}")
        logger.info(f"理論最大配置率: {((total_panels - total_oversized) / total_panels * 100):.1f}%")

        if total_oversized > 0:
            logger.error(f"❌ {total_oversized} 個のパネルは物理的に配置不可能!")
            logger.error("これが100%配置率を阻害している根本原因です")
            return False
        else:
            logger.info("✅ すべてのパネルは理論的に配置可能")
            logger.error("アルゴリズムに問題があります - すべて配置できるはずです")
            return True

    except Exception as e:
        logger.error(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    analyze_panel_sizes()