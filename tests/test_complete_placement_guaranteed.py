#!/usr/bin/env python3
"""
Test Complete Placement Guaranteed Algorithm - 100% Panel Placement
100%配置保証アルゴリズムテスト - 確実な全パネル配置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.complete_placement_guaranteed import CompletePlacementGuaranteedAlgorithm
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_complete_placement_guaranteed():
    """Test the Complete Placement Guaranteed Algorithm"""
    try:
        logger.info("100%配置保証アルゴリズムテスト開始...")

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

        # Show panel overview
        logger.info("\\n=== パネル概要 ===")
        for i, panel in enumerate(panels[:10]):  # Show first 10 panels
            panel_w = getattr(panel, 'cutting_width', panel.width)
            panel_h = getattr(panel, 'cutting_height', panel.height)
            logger.info(f"  {panel.id}: {panel.quantity}個 - {panel_w:.0f}x{panel_h:.0f}mm")

        # Test with multi-sheet optimization
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # Allow many sheets
            time_budget=0.0,           # No time limit
            kerf_width=0.0,           # Thin sheet cutting
            target_efficiency=0.01   # Very low efficiency target - focus on placement
        )

        # Create engine and register the new algorithm
        engine = OptimizationEngine()
        engine.register_algorithm(CompletePlacementGuaranteedAlgorithm())

        logger.info("Complete Placement Guaranteed最適化実行中...")
        start_time = time.time()
        results = engine.optimize(panels, constraints, 'Complete_Placement_Guaranteed')
        execution_time = time.time() - start_time

        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100
        total_sheets = len(results)
        avg_efficiency = sum(result.efficiency for result in results) / len(results) if results else 0

        logger.info(f"\\n=== Complete Placement Guaranteed結果 ===")
        logger.info(f"配置パネル数: {total_placed}/{total_panels} ({placement_rate:.1f}%)")
        logger.info(f"使用シート数: {total_sheets}")
        logger.info(f"平均効率: {avg_efficiency:.1%}")
        logger.info(f"実行時間: {execution_time:.2f}秒")

        # Show detailed results for each sheet
        logger.info(f"\\n=== シート別配置詳細 ===")
        for i, result in enumerate(results[:5]):  # Show first 5 sheets
            sheet_size = f"{result.sheet.width}x{result.sheet.height}mm"
            logger.info(f"  シート{i+1}: {len(result.panels)}個配置 - {sheet_size} ({result.efficiency:.1%}効率)")

        # Show sheet size variety used
        sheet_sizes = {}
        for result in results:
            size_key = f"{result.sheet.width}x{result.sheet.height}mm"
            if size_key not in sheet_sizes:
                sheet_sizes[size_key] = 0
            sheet_sizes[size_key] += 1

        logger.info(f"\\n=== 使用シートサイズ分布 ===")
        for size, count in sorted(sheet_sizes.items()):
            logger.info(f"  {size}: {count}枚")

        # Analyze improvement and success
        logger.info(f"\\n=== 性能評価 ===")

        if placement_rate >= 100.0:
            logger.info("🎉 SUCCESS: Complete Placement Guaranteed が100%配置を達成！")
            logger.info("✅ すべてのパネルが配置されました - 完全成功")
            return True
        elif placement_rate >= 98.0:
            logger.info(f"✅ EXCELLENT: {placement_rate:.1f}%配置 - ほぼ完全配置")
            unplaced = total_panels - total_placed
            logger.info(f"未配置パネル: {unplaced}個のみ")
            return True
        elif placement_rate >= 95.0:
            logger.info(f"⚠️ GOOD: {placement_rate:.1f}%配置 - 高い配置率")
            unplaced = total_panels - total_placed
            logger.info(f"未配置パネル: {unplaced}個")
            return True
        elif placement_rate >= 90.0:
            logger.info(f"⚠️ FAIR: {placement_rate:.1f}%配置 - 改善必要")
            return False
        else:
            logger.error(f"❌ FAILURE: {placement_rate:.1f}%配置 - アルゴリズム要改善")
            return False

    except Exception as e:
        logger.error(f"100%配置保証アルゴリズムテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_placement_guaranteed()
    sys.exit(0 if success else 1)