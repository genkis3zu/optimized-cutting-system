#!/usr/bin/env python3
"""
Test single panel placement to debug algorithm
単一パネル配置テストでアルゴリズムをデバッグ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.ffd import FirstFitDecreasing
from core.models import OptimizationConstraints, Panel
from core.pi_manager import PIManager
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_single_panel_placement():
    """Test placing a single panel to verify basic algorithm"""
    try:
        logger.info("単一パネル配置テスト開始...")

        # Parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"パネル種類数: {len(panels)}")

        # Apply PI expansion
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        # Test with a small sample of panels
        test_panels = panels[:5]  # First 5 panel types
        logger.info(f"テストパネル数: {len(test_panels)}")

        for panel in test_panels:
            logger.info(f"パネル: {panel.id}, サイズ: {panel.cutting_width}x{panel.cutting_height}mm, 材料: {panel.material}, 数量: {panel.quantity}")

        # Test constraints
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,           # Allow unlimited sheets
            time_budget=0.0,           # NO TIME LIMIT
            kerf_width=0.0,           # Thin sheet cutting
            target_efficiency=1.0     # Target 100% placement
        )

        # Create engine and test with FFD
        engine = OptimizationEngine()
        ffd_algorithm = FirstFitDecreasing()
        engine.register_algorithm(ffd_algorithm)

        logger.info("FFD最適化実行中...")
        results = engine.optimize(test_panels, constraints, 'FFD')

        total_test_panels = sum(panel.quantity for panel in test_panels)
        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_test_panels) * 100

        logger.info(f"テスト結果:")
        logger.info(f"  対象パネル数: {total_test_panels}")
        logger.info(f"  配置パネル数: {total_placed}")
        logger.info(f"  配置率: {placement_rate:.1f}%")
        logger.info(f"  使用シート数: {len(results)}")

        if placement_rate < 100.0:
            logger.error(f"❌ 単純なテストケースでも100%配置できません!")
            logger.error(f"未配置パネル数: {total_test_panels - total_placed}")

            # Detailed analysis of each sheet
            for i, result in enumerate(results):
                logger.info(f"シート {i+1}: {len(result.panels)} パネル配置, 効率 {result.efficiency:.1%}")
                for placed_panel in result.panels:
                    logger.info(f"  - {placed_panel.panel_id}: {placed_panel.actual_width}x{placed_panel.actual_height}mm")

        else:
            logger.info("✅ 単純なテストケースで100%配置達成!")

        return placement_rate >= 100.0

    except Exception as e:
        logger.error(f"単一パネルテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_very_simple_case():
    """Test with absolutely minimal case - 1 panel type, 1 quantity"""
    try:
        logger.info("\n=== 超シンプルケーステスト ===")

        # Create single panel manually
        test_panel = Panel(
            id="TEST_001",
            width=500.0,  # Small panel
            height=300.0,
            quantity=1,   # Only 1 panel
            material="SECC",
            thickness=0.5,
            priority=1,
            allow_rotation=True
        )

        # Apply PI expansion
        pi_manager = PIManager()
        test_panel.calculate_expanded_dimensions(pi_manager)

        logger.info(f"テストパネル: {test_panel.cutting_width}x{test_panel.cutting_height}mm")

        # Test constraints
        constraints = OptimizationConstraints(
            material_separation=False,
            max_sheets=1000,
            time_budget=0.0,
            kerf_width=0.0,
            target_efficiency=1.0
        )

        # Create engine
        engine = OptimizationEngine()
        ffd_algorithm = FirstFitDecreasing()
        engine.register_algorithm(ffd_algorithm)

        logger.info("超シンプルFFD最適化実行中...")
        results = engine.optimize([test_panel], constraints, 'FFD')

        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / test_panel.quantity) * 100

        logger.info(f"超シンプルテスト結果:")
        logger.info(f"  対象パネル数: {test_panel.quantity}")
        logger.info(f"  配置パネル数: {total_placed}")
        logger.info(f"  配置率: {placement_rate:.1f}%")

        if placement_rate < 100.0:
            logger.error("❌ 1パネルすら配置できません！アルゴリズムに根本的問題があります")
        else:
            logger.info("✅ 1パネルは正常に配置できています")

        return placement_rate >= 100.0

    except Exception as e:
        logger.error(f"超シンプルテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_single_panel_placement()
    success2 = test_very_simple_case()

    if not success1 or not success2:
        logger.error("アルゴリズムに根本的な問題があります")
        sys.exit(1)
    else:
        logger.info("基本テストは成功しました - より複雑な問題があります")
        sys.exit(0)