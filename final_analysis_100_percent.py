#!/usr/bin/env python3
"""
Final Analysis - Root Cause of Unplaced Panels
最終分析 - 未配置パネルの根本原因

This script provides a definitive answer to the user's question:
なぜ多様なシートが選択できているのに、まだパネルが配置できないのか？
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

def final_analysis():
    """Definitive analysis of why panels can't be placed despite diverse sheet options"""
    try:
        logger.info("最終分析: 未配置パネルの根本原因調査...")

        # Parse data and apply PI expansion
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"総パネル数: {total_panels}")

        # Get material manager and analyze sheets
        material_manager = MaterialInventoryManager()
        all_sheets = material_manager.inventory

        # Find the actual largest sheets
        largest_sheets = sorted(all_sheets, key=lambda s: s.area, reverse=True)[:5]
        logger.info("\\n=== 実際の最大シートサイズ ===")
        for i, sheet in enumerate(largest_sheets):
            logger.info(f"  {i+1}. {sheet.width}x{sheet.height}mm - {sheet.material_type} (面積: {sheet.area:.0f}mm²)")

        absolute_max_w = max(sheet.width for sheet in all_sheets)
        absolute_max_h = max(sheet.height for sheet in all_sheets)
        logger.info(f"\\n絶対最大シート寸法: {absolute_max_w}x{absolute_max_h}mm")

        # Analyze each panel systematically
        logger.info("\\n=== 全パネル配置可能性分析 ===")

        impossible_panels = []
        questionable_panels = []
        placeable_panels = []

        for panel in panels:
            cutting_w = getattr(panel, 'cutting_width', panel.width)
            cutting_h = getattr(panel, 'cutting_height', panel.height)

            # Check against absolute largest sheet
            fits_normal = cutting_w <= absolute_max_w and cutting_h <= absolute_max_h
            fits_rotated = cutting_h <= absolute_max_w and cutting_w <= absolute_max_h if panel.allow_rotation else False

            if not fits_normal and not fits_rotated:
                impossible_panels.append((panel, cutting_w, cutting_h))
            elif cutting_w > 1300 or cutting_h > 1300:  # Borderline large
                questionable_panels.append((panel, cutting_w, cutting_h))
            else:
                placeable_panels.append((panel, cutting_w, cutting_h))

        # Report results
        impossible_qty = sum(panel[0].quantity for panel in impossible_panels)
        questionable_qty = sum(panel[0].quantity for panel in questionable_panels)
        placeable_qty = sum(panel[0].quantity for panel in placeable_panels)

        logger.info(f"\\n=== 配置可能性カテゴリ ===")
        logger.info(f"物理的配置不可能: {len(impossible_panels)}種類, {impossible_qty}個 ({impossible_qty/total_panels*100:.1f}%)")
        logger.info(f"配置困難(1300mm超): {len(questionable_panels)}種類, {questionable_qty}個 ({questionable_qty/total_panels*100:.1f}%)")
        logger.info(f"確実配置可能: {len(placeable_panels)}種類, {placeable_qty}個 ({placeable_qty/total_panels*100:.1f}%)")

        if impossible_panels:
            logger.info("\\n物理的配置不可能パネル:")
            for panel, w, h in impossible_panels:
                logger.error(f"  ❌ {panel.id}: {w:.0f}x{h:.0f}mm - {panel.quantity}個")
                logger.error(f"     理由: 最大シート({absolute_max_w}x{absolute_max_h}mm)より大きい")

        if questionable_panels:
            logger.info("\\n配置困難パネル:")
            for panel, w, h in questionable_panels[:5]:  # Show first 5
                logger.warning(f"  ⚠️ {panel.id}: {w:.0f}x{h:.0f}mm - {panel.quantity}個")

        # Theoretical maximum placement rate
        theoretical_max = (placeable_qty + questionable_qty) / total_panels * 100
        logger.info(f"\\n=== 結論 ===")
        logger.info(f"理論最大配置率: {theoretical_max:.1f}%")

        if theoretical_max >= 100:
            logger.info("✅ 理論上100%配置可能 → アルゴリズム改善で解決可能")
        elif theoretical_max >= 90:
            logger.info("✅ 理論上90%以上配置可能 → アルゴリズム改善で大幅改善可能")
        elif theoretical_max >= 80:
            logger.warning("⚠️ 理論上80%程度配置可能 → 一部物理的制約あり")
        else:
            logger.error("❌ 深刻な物理的制約 → シートサイズ拡張必要")

        # Answer the user's specific question
        logger.info("\\n=== ユーザー質問への回答 ===")
        logger.info("Q: 多様なシートが選択できているのに、なぜパネルが配置できないのか？")

        if impossible_qty > 50:
            logger.info("A: 主要因は**物理的制約** - 一部パネルが最大シートより大きい")
            logger.info("   解決策: より大きなシートサイズの追加または巨大パネルの分割")
        elif impossible_qty + questionable_qty > 100:
            logger.info("A: 主要因は**サイズ制約** - 大型パネルが多く、適切なシートとのマッチングが困難")
            logger.info("   解決策: 大型シート優先選択とアルゴリズム改善")
        else:
            logger.info("A: 主要因は**アルゴリズム効率** - 理論上は配置可能だが、アルゴリズムが最適配置できていない")
            logger.info("   解決策: アルゴリズムの改善（バルク処理、グリッド配置、効率的探索）")

        return theoretical_max >= 90

    except Exception as e:
        logger.error(f"最終分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = final_analysis()
    sys.exit(0 if success else 1)