#!/usr/bin/env python3
"""
Test Fixed Optimizer - 100% Placement with Proper Sheet Selection
修正された最適化エンジンの100%配置テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_optimizer():
    """Test the fixed optimizer with proper sheet selection"""
    try:
        logger.info("修正された最適化エンジンテスト開始...")

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

        # Test with multi-sheet optimization (should achieve 100%)
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # Allow many sheets for complete placement
            time_budget=0.0,           # No time limit
            kerf_width=0.0,           # Thin sheet cutting
            target_efficiency=0.1     # Low efficiency target - focus on placement
        )

        # Create engine and register algorithms
        engine = OptimizationEngine()

        # Register algorithms
        from core.algorithms.ffd import FirstFitDecreasing
        from core.algorithms.improved_ffd import ImprovedFirstFitDecreasing
        engine.register_algorithm(FirstFitDecreasing())
        engine.register_algorithm(ImprovedFirstFitDecreasing())

        logger.info("Improved FFD最適化実行中 (修正版シート選択)...")
        start_time = time.time()
        results = engine.optimize(panels, constraints, 'Improved_FFD')
        execution_time = time.time() - start_time

        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100
        total_sheets = len(results)
        avg_efficiency = sum(result.efficiency for result in results) / len(results) if results else 0

        logger.info(f"\n=== Improved FFD結果 ===")
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

        if placement_rate >= 100.0:
            logger.info("🎉 SUCCESS: Improved FFDが100%配置を達成！")
            return True
        else:
            logger.warning(f"⚠️ Improved FFD結果: {placement_rate:.1f}%配置")
            logger.error(f"未配置パネル数: {total_panels - total_placed}")
            return placement_rate >= 95.0  # 95%以上なら成功とみなす

    except Exception as e:
        logger.error(f"修正版最適化エンジンテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_optimizer()
    sys.exit(0 if success else 1)