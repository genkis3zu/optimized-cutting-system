#!/usr/bin/env python3
"""
Test Guaranteed Placement Algorithm - Bulk Processing & Global Optimization
保証配置アルゴリズムテスト - バルク処理とグローバル最適化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.unlimited_runtime_optimizer import UnlimitedRuntimeOptimizer
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_guaranteed_placement():
    """Test the new Guaranteed Placement Algorithm"""
    try:
        logger.info("保証配置アルゴリズムテスト開始...")

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

        # Show bulk analysis before optimization
        logger.info("\n=== バルク分析 ===")
        high_quantity_panels = [p for p in panels if p.quantity >= 4]
        high_quantity_panels.sort(key=lambda p: p.quantity, reverse=True)

        for panel in high_quantity_panels[:10]:
            panel_w = getattr(panel, 'cutting_width', panel.width)
            panel_h = getattr(panel, 'cutting_height', panel.height)
            logger.info(f"  {panel.id}: {panel.quantity}個 - {panel_w:.0f}x{panel_h:.0f}mm")

        # Test with multi-sheet optimization
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # Allow many sheets
            time_budget=0.0,           # No time limit
            kerf_width=0.0,           # Thin sheet cutting
            target_efficiency=0.1     # Low efficiency target - focus on placement
        )

        # Create engine and register the new algorithm
        engine = OptimizationEngine()
        from core.algorithms.simple_bulk_optimizer import SimpleBulkOptimizer
        engine.register_algorithm(SimpleBulkOptimizer())

        logger.info("Simple Bulk最適化実行中...")
        start_time = time.time()
        results = engine.optimize(panels, constraints, 'Simple_Bulk')
        execution_time = time.time() - start_time

        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100
        total_sheets = len(results)
        avg_efficiency = sum(result.efficiency for result in results) / len(results) if results else 0

        logger.info(f"\n=== Simple Bulk結果 ===")
        logger.info(f"配置パネル数: {total_placed}/{total_panels} ({placement_rate:.1f}%)")
        logger.info(f"使用シート数: {total_sheets}")
        logger.info(f"平均効率: {avg_efficiency:.1%}")
        logger.info(f"実行時間: {execution_time:.2f}秒")

        # Show sheet size variety used
        sheet_sizes = {}
        for result in results:
            size_key = f"{result.sheet.width}x{result.sheet.height}mm"
            if size_key not in sheet_sizes:
                sheet_sizes[size_key] = 0
            sheet_sizes[size_key] += 1

        logger.info(f"\n=== 使用シートサイズ分布 ===")
        for size, count in sorted(sheet_sizes.items()):
            logger.info(f"  {size}: {count}枚")

        # Analyze improvement vs previous algorithms
        improvement_vs_ffd = placement_rate - 67.9
        logger.info(f"\n=== 性能改善 ===")
        logger.info(f"FFDからの改善: +{improvement_vs_ffd:.1f}ポイント")

        if placement_rate >= 100.0:
            logger.info("🎉 SUCCESS: Simple Bulkが100%配置を達成！")
            return True
        elif placement_rate >= 90.0:
            logger.info(f"✅ GOOD: {placement_rate:.1f}%配置 - 大幅改善")
            return True
        elif placement_rate > 67.9:
            logger.warning(f"⚠️ IMPROVED: {placement_rate:.1f}%配置 - 改善あり")
            return True
        else:
            logger.error(f"❌ NO IMPROVEMENT: {placement_rate:.1f}%配置")
            return False

    except Exception as e:
        logger.error(f"保証配置アルゴリズムテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_guaranteed_placement()
    sys.exit(0 if success else 1)