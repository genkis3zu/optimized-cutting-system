#!/usr/bin/env python3
"""
Test Improved FFD Algorithm for 100% placement
改善されたFFDアルゴリズムの100%配置テスト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.improved_ffd import ImprovedFirstFitDecreasing
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_improved_ffd():
    """Test Improved FFD vs Original FFD"""
    try:
        logger.info("改善されたFFDアルゴリズムテスト開始...")

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

        # Constraints: シンプルに全パネル配置を目指す
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # 無制限シート数
            time_budget=0.0,           # 時間制限なし
            kerf_width=0.0,           # 薄板切断
            target_efficiency=0.5     # 目標歩留まり率は低く設定（実際は無関係）
        )

        # Create engines
        engine = OptimizationEngine()

        # Register both algorithms
        original_ffd = FirstFitDecreasing()
        improved_ffd = ImprovedFirstFitDecreasing()

        engine.register_algorithm(original_ffd)
        engine.register_algorithm(improved_ffd)

        # Test Original FFD
        logger.info("=== オリジナルFFDテスト ===")
        start_time = time.time()
        original_results = engine.optimize(panels, constraints, 'FFD')
        original_time = time.time() - start_time
        original_placed = sum(len(result.panels) for result in original_results)
        original_rate = (original_placed / total_panels) * 100

        logger.info(f"オリジナルFFD結果:")
        logger.info(f"  配置パネル数: {original_placed}/{total_panels} ({original_rate:.1f}%)")
        logger.info(f"  使用シート数: {len(original_results)}")
        logger.info(f"  実行時間: {original_time:.2f}秒")

        # Test Improved FFD
        logger.info("=== 改善されたFFDテスト ===")
        start_time = time.time()
        improved_results = engine.optimize(panels, constraints, 'Improved_FFD')
        improved_time = time.time() - start_time
        improved_placed = sum(len(result.panels) for result in improved_results)
        improved_rate = (improved_placed / total_panels) * 100

        logger.info(f"改善されたFFD結果:")
        logger.info(f"  配置パネル数: {improved_placed}/{total_panels} ({improved_rate:.1f}%)")
        logger.info(f"  使用シート数: {len(improved_results)}")
        logger.info(f"  実行時間: {improved_time:.2f}秒")

        # Compare results
        improvement = improved_rate - original_rate
        logger.info(f"\n=== 比較結果 ===")
        logger.info(f"改善度: {improvement:+.1f}%")
        logger.info(f"配置改善: {improved_placed - original_placed:+d} パネル")

        if improved_rate >= 100.0:
            logger.info("🎉 改善されたFFDが100%配置を達成！")
            return True
        elif improved_rate > original_rate:
            logger.info(f"✅ 改善されたFFDが {improvement:.1f}% 向上")
            return True
        elif improved_rate == original_rate:
            logger.info("🤔 同等の結果")
            return True
        else:
            logger.warning("❌ 改善されたFFDが劣化")
            return False

    except Exception as e:
        logger.error(f"改善されたFFDテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_ffd()
    sys.exit(0 if success else 1)