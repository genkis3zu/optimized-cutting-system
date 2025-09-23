#!/usr/bin/env python3
"""
Correct Unplaced Panel Analysis - Using All Available Sheet Sizes
正しい未配置パネル分析 - 利用可能なすべてのシートサイズ使用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.pi_manager import PIManager
from core.material_manager import MaterialInventoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_with_all_sheets():
    """Analyze unplaced panels using ALL available sheet sizes"""
    try:
        logger.info("正しい未配置パネル分析開始（全シートサイズ使用）...")

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

        # Get material manager
        material_manager = MaterialInventoryManager()

        # Group panels by material
        material_groups = {}
        for panel in panels:
            # Normalize material
            normalized_material = material_manager.normalize_material_code(panel.material)
            if normalized_material not in material_groups:
                material_groups[normalized_material] = []
            material_groups[normalized_material].append(panel)

        logger.info(f"材料グループ数: {len(material_groups)}")

        # Show all available sheets by material
        logger.info("\n=== 利用可能シート一覧 ===")
        for material in material_groups.keys():
            sheets = material_manager.get_sheets_by_type(material)
            if sheets:
                logger.info(f"材料 {material}: {len(sheets)} 種類のシート")
                for sheet in sheets:
                    logger.info(f"  - {sheet.width}x{sheet.height}mm (面積: {sheet.area:.0f}mm²)")
            else:
                logger.error(f"材料 {material}: シートなし")

        # Analyze each material group with ALL sheets
        total_analyzed = 0
        total_unplaceable = 0
        unplaceable_details = []

        for material, material_panels in material_groups.items():
            logger.info(f"\n=== 材料: {material} ===")

            # Analyze each panel with ALL available sheets
            material_unplaceable = 0
            for panel in material_panels:
                # Get panel dimensions
                panel_width = getattr(panel, 'cutting_width', panel.width)
                panel_height = getattr(panel, 'cutting_height', panel.height)
                panel_count = panel.quantity
                total_analyzed += panel_count

                # Find ALL compatible sheets for this panel
                compatible_sheets = material_manager.find_compatible_sheets(
                    material, panel.thickness, panel_width, panel_height
                )

                # Also check with rotation if allowed
                rotated_compatible = []
                if panel.allow_rotation:
                    rotated_compatible = material_manager.find_compatible_sheets(
                        material, panel.thickness, panel_height, panel_width
                    )

                # Combine results
                all_compatible = compatible_sheets + rotated_compatible

                if not all_compatible:
                    # Panel cannot fit in any available sheet
                    material_unplaceable += panel_count
                    total_unplaceable += panel_count

                    # Get all sheets for this material to show available sizes
                    all_sheets = material_manager.get_sheets_by_type(material)
                    if all_sheets:
                        available_sizes = [f"{s.width}x{s.height}mm" for s in all_sheets]
                        available_str = ", ".join(available_sizes)
                    else:
                        available_str = "なし"

                    unplaceable_details.append({
                        'panel_id': panel.id,
                        'material': material,
                        'quantity': panel_count,
                        'reason': 'すべてのシートサイズ超過',
                        'size': f"{panel_width:.1f}x{panel_height:.1f}mm",
                        'available_sheets': available_str
                    })
                    logger.error(f"  ❌ {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count})")
                    logger.error(f"      利用可能シート: {available_str}")

                else:
                    # Panel can fit - show best options
                    if compatible_sheets:
                        best_sheet = compatible_sheets[0]  # Smallest compatible sheet
                        logger.info(f"  ✅ {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - 最適シート:{best_sheet.width}x{best_sheet.height}mm")
                    elif rotated_compatible:
                        best_sheet = rotated_compatible[0]
                        logger.info(f"  🔄 {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - 回転で最適シート:{best_sheet.width}x{best_sheet.height}mm")

            if material_unplaceable > 0:
                logger.error(f"材料 {material}: {material_unplaceable} 個のパネルが配置不可能")
            else:
                logger.info(f"材料 {material}: すべてのパネルが配置可能！")

        # Summary
        logger.info(f"\n=== 分析結果サマリー（全シート考慮） ===")
        logger.info(f"分析対象パネル数: {total_analyzed}")
        logger.info(f"配置不可能パネル数: {total_unplaceable}")

        if total_unplaceable == 0:
            logger.info("🎉 すべてのパネルが配置可能です！")
            placement_rate = 100.0
        else:
            placement_rate = ((total_analyzed - total_unplaceable) / total_analyzed * 100)
            logger.info(f"理論最大配置率: {placement_rate:.1f}%")

        if unplaceable_details:
            logger.info(f"\n=== 真の配置不可能パネル詳細 ===")
            for detail in unplaceable_details:
                logger.error(f"❌ {detail['panel_id']} ({detail['material']}): {detail['size']} - 数量:{detail['quantity']}")
                logger.error(f"   利用可能シート: {detail['available_sheets']}")

        return total_unplaceable == 0, placement_rate

    except Exception as e:
        logger.error(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0

if __name__ == "__main__":
    success, rate = analyze_with_all_sheets()
    if success:
        logger.info(f"🎉 100%配置が可能です！ ({rate:.1f}%)")
    else:
        logger.error(f"❌ 配置率: {rate:.1f}%")
    sys.exit(0 if success else 1)