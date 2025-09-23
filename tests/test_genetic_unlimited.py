#!/usr/bin/env python3
"""
Unlimited Genetic Algorithm Test - Run until 100% placement
無制限遺伝的アルゴリズムテスト - 100%配置まで実行
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.genetic import GeneticAlgorithm
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_genetic_unlimited():
    """Test Genetic Algorithm with unlimited time until 100% placement"""
    try:
        logger.info("無制限遺伝的アルゴリズムテスト開始 - 100%配置まで実行...")

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

        # Create constraints with NO time limit and unlimited sheets
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode for best placement
            max_sheets=1000,           # Allow unlimited sheets
            time_budget=0.0,           # NO TIME LIMIT - run until 100%
            kerf_width=0.0,           # Thin sheet cutting
            target_efficiency=1.0     # Target 100% placement
        )

        logger.info(f"制約条件: 時間制限={constraints.time_budget} (無制限), 最大シート数={constraints.max_sheets}, 目標効率=100%")

        # Create engine
        engine = OptimizationEngine()

        # Register algorithms with optimized parameters for 100% placement
        ffd_algorithm = FirstFitDecreasing()
        genetic_algorithm = GeneticAlgorithm(
            population_size=50,    # Larger population for better exploration
            generations=999999,    # Effectively unlimited generations
            mutation_rate=0.2,     # Higher mutation for better exploration
            crossover_rate=0.85    # High crossover for exploitation
        )

        engine.register_algorithm(ffd_algorithm)
        engine.register_algorithm(genetic_algorithm)

        # Test FFD baseline first
        logger.info("FFDベースライン測定中...")
        start_time = time.time()
        ffd_results = engine.optimize(panels, constraints, 'FFD')
        ffd_time = time.time() - start_time
        ffd_placed = sum(len(result.panels) for result in ffd_results)
        ffd_rate = (ffd_placed / total_panels) * 100

        logger.info(f"FFD結果: {ffd_placed}/{total_panels} ({ffd_rate:.1f}%) 実行時間: {ffd_time:.1f}秒")
        logger.info(f"FFD使用シート数: {len(ffd_results)}")

        # Test Genetic Algorithm - NO TIME LIMIT
        logger.info("=== 遺伝的アルゴリズム実行開始 (時間制限なし - 100%配置まで継続) ===")
        start_time = time.time()

        ga_results = engine.optimize(panels, constraints, 'GA')

        ga_time = time.time() - start_time
        ga_placed = sum(len(result.panels) for result in ga_results)
        ga_rate = (ga_placed / total_panels) * 100

        logger.info(f"=== 遺伝的アルゴリズム結果 ===")
        logger.info(f"実行時間: {ga_time:.1f}秒 ({ga_time/60:.1f}分)")
        logger.info(f"使用シート数: {len(ga_results)}")
        logger.info(f"配置パネル数: {ga_placed}/{total_panels} ({ga_rate:.1f}%)")

        improvement = ga_rate - ffd_rate
        logger.info(f"FFDからの改善: {improvement:+.1f}%")

        # Success criteria
        if ga_rate >= 100.0:
            logger.info("🎉 SUCCESS: 遺伝的アルゴリズムが100%配置を達成しました!")
            return True
        elif ga_rate >= 95.0:
            logger.info(f"🔥 EXCELLENT: 遺伝的アルゴリズムが{ga_rate:.1f}%配置率を達成!")
            return True
        elif ga_rate >= 90.0:
            logger.info(f"✅ VERY GOOD: 遺伝的アルゴリズムが{ga_rate:.1f}%配置率を達成!")
            return True
        elif ga_rate > ffd_rate:
            logger.info(f"🔍 IMPROVEMENT: 遺伝的アルゴリズムがFFDの{ffd_rate:.1f}%から{ga_rate:.1f}%に改善")
            return True
        else:
            logger.warning(f"❌ NO IMPROVEMENT: 遺伝的アルゴリズム {ga_rate:.1f}% vs FFD {ffd_rate:.1f}%")
            return False

    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")
        return False
    except Exception as e:
        logger.error(f"無制限テストでエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genetic_unlimited()
    sys.exit(0 if success else 1)