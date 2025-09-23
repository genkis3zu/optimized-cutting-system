#!/usr/bin/env python3
"""
Test Complete Placement Algorithm for 100% guarantee
100%配置保証アルゴリズムのテスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.complete_placement import CompletePlacementAlgorithm
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_complete_placement():
    """Test Complete Placement Algorithm"""
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

        # Constraints: すべてのパネルを配置
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # 無制限シート数
            time_budget=0.0,           # 時間制限なし
            kerf_width=0.0,           # 薄板切断
            target_efficiency=0.1     # 効率は重要でない - 全配置が目標
        )

        # Create engine and test
        engine = OptimizationEngine()
        complete_algorithm = CompletePlacementAlgorithm()
        engine.register_algorithm(complete_algorithm)

        logger.info("Complete Placement実行中...")
        start_time = time.time()
        results = engine.optimize(panels, constraints, 'Complete_Placement')
        execution_time = time.time() - start_time

        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100

        logger.info(f"Complete Placement結果:")
        logger.info(f"  配置パネル数: {total_placed}/{total_panels} ({placement_rate:.1f}%)")
        logger.info(f"  使用シート数: {len(results)}")
        logger.info(f"  実行時間: {execution_time:.2f}秒")

        if placement_rate >= 100.0:
            logger.info("🎉 SUCCESS: Complete Placementが100%配置を達成！")
            return True
        else:
            logger.error(f"❌ FAILURE: Complete Placementが {placement_rate:.1f}% しか配置できませんでした")
            logger.error(f"未配置パネル数: {total_panels - total_placed}")
            return False

    except Exception as e:
        logger.error(f"Complete Placementテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_placement()
    sys.exit(0 if success else 1)