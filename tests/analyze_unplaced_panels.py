#!/usr/bin/env python3
"""
Analyze unplaced panels to find root cause
未配置パネルの根本原因分析
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

def analyze_unplaced_panels():
    """Analyze why some panels cannot be placed"""
    try:
        logger.info("未配置パネルの詳細分析開始...")

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

        # Analyze each material group
        total_analyzed = 0
        total_unplaceable = 0
        unplaceable_details = []

        for material, material_panels in material_groups.items():
            logger.info(f"\n=== 材料: {material} ===")

            # Get sheet for this material
            sheets = material_manager.get_sheets_by_type(material)
            if not sheets:
                logger.error(f"❌ 材料 {material} のシートが見つかりません!")
                for panel in material_panels:
                    panel_count = panel.quantity
                    total_unplaceable += panel_count
                    unplaceable_details.append({
                        'panel_id': panel.id,
                        'material': material,
                        'quantity': panel_count,
                        'reason': 'シート情報なし',
                        'size': f"{getattr(panel, 'cutting_width', panel.width):.1f}x{getattr(panel, 'cutting_height', panel.height):.1f}mm"
                    })
                continue

            sheet_data = sheets[0]  # Use first available sheet
            sheet_width = sheet_data.width
            sheet_height = sheet_data.height

            logger.info(f"使用シート: {sheet_width}x{sheet_height}mm")

            # Analyze each panel
            material_unplaceable = 0
            for panel in material_panels:
                # Get panel dimensions
                panel_width = getattr(panel, 'cutting_width', panel.width)
                panel_height = getattr(panel, 'cutting_height', panel.height)

                # Check if panel fits
                fits_normal = (panel_width <= sheet_width and panel_height <= sheet_height)
                fits_rotated = (panel_height <= sheet_width and panel_width <= sheet_height) if panel.allow_rotation else False

                panel_count = panel.quantity
                total_analyzed += panel_count

                if not fits_normal and not fits_rotated:
                    # Panel too large
                    material_unplaceable += panel_count
                    total_unplaceable += panel_count
                    unplaceable_details.append({
                        'panel_id': panel.id,
                        'material': material,
                        'quantity': panel_count,
                        'reason': 'シートサイズ超過',
                        'size': f"{panel_width:.1f}x{panel_height:.1f}mm",
                        'sheet_size': f"{sheet_width:.1f}x{sheet_height:.1f}mm"
                    })
                    logger.error(f"  ❌ {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - シートサイズ超過!")
                elif fits_normal and fits_rotated:
                    logger.info(f"  ✅ {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - 配置可能")
                elif fits_rotated:
                    logger.info(f"  🔄 {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - 回転で配置可能")
                else:
                    logger.info(f"  ✅ {panel.id}: {panel_width:.1f}x{panel_height:.1f}mm (数量:{panel_count}) - 配置可能")

            if material_unplaceable > 0:
                logger.error(f"材料 {material}: {material_unplaceable} 個のパネルが配置不可能")

        # Summary
        logger.info(f"\n=== 分析結果サマリー ===")
        logger.info(f"分析対象パネル数: {total_analyzed}")
        logger.info(f"配置不可能パネル数: {total_unplaceable}")
        logger.info(f"理論最大配置率: {((total_analyzed - total_unplaceable) / total_analyzed * 100):.1f}%")

        if unplaceable_details:
            logger.info(f"\n=== 配置不可能パネル詳細 ===")
            for detail in unplaceable_details:
                if 'sheet_size' in detail:
                    logger.error(f"❌ {detail['panel_id']} ({detail['material']}): {detail['size']} > {detail['sheet_size']} - 数量:{detail['quantity']}")
                else:
                    logger.error(f"❌ {detail['panel_id']} ({detail['material']}): {detail['reason']} - 数量:{detail['quantity']}")

        # Check if this explains the 17 missing panels
        expected_missing = total_unplaceable
        actual_missing = total_panels - 456  # From previous test result

        logger.info(f"\n=== 予測と実績の比較 ===")
        logger.info(f"予測配置不可能数: {expected_missing}")
        logger.info(f"実際の未配置数: {actual_missing}")

        if expected_missing == actual_missing:
            logger.info("✅ 理論値と実績が一致！未配置の原因が特定されました")
        else:
            logger.warning(f"⚠️ 理論値と実績に差異: {actual_missing - expected_missing} パネルの追加調査が必要")

        return total_unplaceable == 0

    except Exception as e:
        logger.error(f"未配置パネル分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_unplaced_panels()
    if success:
        logger.info("🎉 すべてのパネルが配置可能です")
    else:
        logger.error("❌ 一部のパネルが配置不可能です")
    sys.exit(0 if success else 1)