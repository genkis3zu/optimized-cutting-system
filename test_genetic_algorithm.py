#!/usr/bin/env python3
"""
Test script to verify Genetic Algorithm achieves higher placement rate
遺伝的アルゴリズムでより高い配置率を実現することを検証するテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.optimizer import OptimizationEngine
from core.algorithms.ffd import FirstFitDecreasing
from core.algorithms.bfd import BestFitDecreasing
from core.algorithms.genetic import GeneticAlgorithm
from core.models import OptimizationConstraints
from core.pi_manager import PIManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_genetic_algorithm():
    """Test Genetic Algorithm for 100% placement rate"""
    try:
        logger.info("Testing Genetic Algorithm for optimal placement...")

        # Parse data exactly like test_all_placement.py
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"Parsed {len(panels)} panel types")

        # Apply PI expansion exactly like test_all_placement.py
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        total_panels = sum(panel.quantity for panel in panels)
        logger.info(f"Total individual panels: {total_panels}")

        # Create constraints exactly like test_all_placement.py (proven working)
        constraints = OptimizationConstraints(
            material_separation=False,  # Multi-sheet mode
            max_sheets=1000,  # Allow many sheets
            time_budget=0.0,  # No time limit
            kerf_width=0.0   # Thin sheet cutting
        )

        logger.info(f"Constraints: kerf={constraints.kerf_width}, time_budget={constraints.time_budget}, max_sheets={constraints.max_sheets}, material_separation={constraints.material_separation}")

        # Create engine exactly like test_all_placement.py (proven working)
        engine = OptimizationEngine()

        # Register algorithms including Genetic Algorithm
        ffd_algorithm = FirstFitDecreasing()
        bfd_algorithm = BestFitDecreasing()
        genetic_algorithm = GeneticAlgorithm(
            population_size=30,  # Larger population for better results
            generations=50,      # More generations for optimal solution
            mutation_rate=0.15,  # Higher mutation for exploration
            crossover_rate=0.8   # High crossover for exploitation
        )

        engine.register_algorithm(ffd_algorithm)
        engine.register_algorithm(bfd_algorithm)
        engine.register_algorithm(genetic_algorithm)

        # Test Genetic Algorithm
        logger.info("Testing Genetic Algorithm...")
        results = engine.optimize(panels, constraints, 'GA')
        total_placed = sum(len(result.panels) for result in results)
        placement_rate = (total_placed / total_panels) * 100 if total_panels > 0 else 0

        logger.info(f"Genetic Algorithm Results:")
        logger.info(f"  Sheets used: {len(results)}")
        logger.info(f"  Panels placed: {total_placed}/{total_panels} ({placement_rate:.1f}%)")

        if placement_rate >= 100.0:
            logger.info("🎉 SUCCESS: Genetic Algorithm achieved 100% panel placement!")
            return True
        elif placement_rate >= 80.0:
            logger.info(f"✅ GOOD: Genetic Algorithm achieved {placement_rate:.1f}% placement rate (improvement over FFD's 67.7%)")
            return True
        elif placement_rate > 67.7:
            logger.info(f"🔍 IMPROVEMENT: Genetic Algorithm achieved {placement_rate:.1f}% placement rate (better than FFD's 67.7%)")
            return True
        else:
            logger.warning(f"❌ REGRESSION: Genetic Algorithm only achieved {placement_rate:.1f}% placement rate (worse than FFD's 67.7%)")
            return False

    except Exception as e:
        logger.error(f"Genetic Algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_genetic_algorithm()
    sys.exit(0 if success else 1)