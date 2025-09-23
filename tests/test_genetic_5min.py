#!/usr/bin/env python3
"""
Test script for Genetic Algorithm with 5-minute time limit
遺伝的アルゴリズムの5分制限テストスクリプト
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

def test_genetic_5min():
    """Test Genetic Algorithm with 5-minute limit for higher placement rate"""
    try:
        logger.info("Testing Genetic Algorithm with 5-minute time limit...")

        # Parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"Parsed {len(panels)} panel types")

        # Apply PI expansion
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"Total individual panels: {total_panels}")

        # Create constraints with 5-minute time limit
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,  # Allow many sheets
            time_budget=300.0,  # 5 minute time limit
            kerf_width=0.0   # Thin sheet cutting
        )

        logger.info(f"Constraints: time_budget={constraints.time_budget}s (5 minutes), max_sheets={constraints.max_sheets}")

        # Create engine
        engine = OptimizationEngine()

        # Register algorithms with optimized parameters
        ffd_algorithm = FirstFitDecreasing()
        genetic_algorithm = GeneticAlgorithm(
            population_size=25,  # Larger population for better diversity
            generations=40,      # More generations for convergence
            mutation_rate=0.15,  # Higher mutation for exploration
            crossover_rate=0.8   # High crossover for exploitation
        )

        engine.register_algorithm(ffd_algorithm)
        engine.register_algorithm(genetic_algorithm)

        # Test FFD first for baseline
        logger.info("Testing FFD (baseline)...")
        start_time = time.time()
        ffd_results = engine.optimize(panels, constraints, 'FFD')
        ffd_time = time.time() - start_time
        ffd_placed = sum(len(result.panels) for result in ffd_results)
        ffd_rate = (ffd_placed / total_panels) * 100

        logger.info(f"FFD Results: {ffd_placed}/{total_panels} ({ffd_rate:.1f}%) in {ffd_time:.1f}s")

        # Test Genetic Algorithm
        logger.info("Testing Genetic Algorithm (5-minute limit)...")
        start_time = time.time()
        ga_results = engine.optimize(panels, constraints, 'GA')
        ga_time = time.time() - start_time
        ga_placed = sum(len(result.panels) for result in ga_results)
        ga_rate = (ga_placed / total_panels) * 100

        logger.info(f"Genetic Algorithm Results:")
        logger.info(f"  Execution time: {ga_time:.1f}s ({ga_time/60:.1f} minutes)")
        logger.info(f"  Sheets used: {len(ga_results)}")
        logger.info(f"  Panels placed: {ga_placed}/{total_panels} ({ga_rate:.1f}%)")

        improvement = ga_rate - ffd_rate
        logger.info(f"Improvement: {improvement:+.1f}% vs FFD")

        if ga_rate >= 100.0:
            logger.info("🎉 SUCCESS: Genetic Algorithm achieved 100% panel placement!")
            return True
        elif ga_rate >= 80.0:
            logger.info(f"🔥 EXCELLENT: Genetic Algorithm achieved {ga_rate:.1f}% placement rate!")
            return True
        elif ga_rate > ffd_rate:
            logger.info(f"✅ IMPROVEMENT: Genetic Algorithm achieved {ga_rate:.1f}% vs FFD's {ffd_rate:.1f}%")
            return True
        elif ga_rate == ffd_rate:
            logger.info(f"🤔 EQUAL: Genetic Algorithm matched FFD's {ffd_rate:.1f}% in {ga_time:.1f}s")
            return True
        else:
            logger.warning(f"❌ REGRESSION: Genetic Algorithm {ga_rate:.1f}% vs FFD {ffd_rate:.1f}%")
            return False

    except Exception as e:
        logger.error(f"5-minute test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genetic_5min()
    sys.exit(0 if success else 1)