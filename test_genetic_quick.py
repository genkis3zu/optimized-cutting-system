#!/usr/bin/env python3
"""
Quick test script for Genetic Algorithm with shorter parameters
遺伝的アルゴリズムの短時間テストスクリプト
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_genetic_quick():
    """Quick test of Genetic Algorithm with reduced parameters"""
    try:
        logger.info("Quick test of Genetic Algorithm...")

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

        # Create constraints with reasonable time limit
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,  # Allow many sheets
            time_budget=300.0,  # 5 minute time limit
            kerf_width=0.0   # Thin sheet cutting
        )

        logger.info(f"Constraints: time_budget={constraints.time_budget}s, max_sheets={constraints.max_sheets}")

        # Create engine
        engine = OptimizationEngine()

        # Register algorithms with optimized parameters
        ffd_algorithm = FirstFitDecreasing()
        genetic_algorithm = GeneticAlgorithm(
            population_size=15,  # Smaller population for speed
            generations=20,      # Fewer generations for speed
            mutation_rate=0.1,   # Standard mutation rate
            crossover_rate=0.7   # Standard crossover rate
        )

        engine.register_algorithm(ffd_algorithm)
        engine.register_algorithm(genetic_algorithm)

        # Compare FFD vs GA
        logger.info("Testing FFD (baseline)...")
        ffd_results = engine.optimize(panels, constraints, 'FFD')
        ffd_placed = sum(len(result.panels) for result in ffd_results)
        ffd_rate = (ffd_placed / total_panels) * 100

        logger.info(f"FFD Results: {ffd_placed}/{total_panels} ({ffd_rate:.1f}%)")

        logger.info("Testing Genetic Algorithm (optimized)...")
        ga_results = engine.optimize(panels, constraints, 'GA')
        ga_placed = sum(len(result.panels) for result in ga_results)
        ga_rate = (ga_placed / total_panels) * 100

        logger.info(f"Genetic Algorithm Results:")
        logger.info(f"  Sheets used: {len(ga_results)}")
        logger.info(f"  Panels placed: {ga_placed}/{total_panels} ({ga_rate:.1f}%)")

        improvement = ga_rate - ffd_rate
        logger.info(f"Improvement: {improvement:+.1f}% vs FFD")

        if ga_rate >= 100.0:
            logger.info("🎉 SUCCESS: Genetic Algorithm achieved 100% panel placement!")
            return True
        elif ga_rate > ffd_rate:
            logger.info(f"✅ IMPROVEMENT: Genetic Algorithm achieved {ga_rate:.1f}% vs FFD's {ffd_rate:.1f}%")
            return True
        else:
            logger.warning(f"❌ NO IMPROVEMENT: Genetic Algorithm {ga_rate:.1f}% vs FFD {ffd_rate:.1f}%")
            return False

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genetic_quick()
    sys.exit(0 if success else 1)