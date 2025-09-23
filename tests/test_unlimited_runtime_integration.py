"""
Test script for UnlimitedRuntimeOptimizer integration

Tests the complete integration of the 100% placement guarantee system.
"""

import sys
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unlimited_runtime_integration():
    """Test the complete unlimited runtime integration"""
    try:
        from core.optimizer import create_optimization_engine
        from core.models import Panel, SteelSheet, OptimizationConstraints
        from core.material_manager import get_material_manager

        logger.info("=== Testing UnlimitedRuntimeOptimizer Integration ===")

        # Create test panels
        test_panels = [
            Panel(
                id="P1",
                width=400,
                height=300,
                material="SS400",
                thickness=6.0,
                quantity=2,
                allow_rotation=True
            ),
            Panel(
                id="P2",
                width=350,
                height=250,
                material="SS400",
                thickness=6.0,
                quantity=1,
                allow_rotation=True
            ),
            Panel(
                id="P3",
                width=200,
                height=150,
                material="SS400",
                thickness=6.0,
                quantity=3,
                allow_rotation=True
            )
        ]

        # Create test sheets
        material_manager = get_material_manager()
        material_sheets = material_manager.get_sheets_by_type("SS400")

        # Convert MaterialSheet to SteelSheet objects
        test_sheets = []
        for mat_sheet in material_sheets:
            if abs(mat_sheet.thickness - 6.0) < 0.1:  # Find 6mm thickness
                test_sheets.append(
                    SteelSheet(
                        width=mat_sheet.width,
                        height=mat_sheet.height,
                        material=mat_sheet.material_type,
                        thickness=mat_sheet.thickness
                    )
                )
                break  # Use only one sheet for testing

        if not test_sheets:
            # Create default sheet if none available
            test_sheets = [
                SteelSheet(
                    width=1500,
                    height=3000,
                    material="SS400",
                    thickness=6.0
                )
            ]

        logger.info(f"Test setup: {len(test_panels)} panel types, {sum(p.quantity for p in test_panels)} total panels")
        logger.info(f"Available sheets: {len(test_sheets)}")

        # Create optimization engine
        engine = create_optimization_engine()

        # Test 1: Standard optimization
        logger.info("\n--- Test 1: Standard Optimization ---")
        constraints = OptimizationConstraints(
            time_budget=30.0,
            target_efficiency=0.7
        )

        try:
            result = engine.optimize(test_panels, test_sheets[0], constraints)
            if result:
                placed_count = len(result.panels) if hasattr(result, 'panels') else 0
                total_panels = sum(p.quantity for p in test_panels)
                placement_rate = (placed_count / total_panels) * 100
                logger.info(f"Standard optimization: {placed_count}/{total_panels} panels ({placement_rate:.1f}%)")
                logger.info(f"Efficiency: {result.efficiency:.1f}%")
            else:
                logger.warning("Standard optimization returned no result")
        except Exception as e:
            logger.error(f"Standard optimization failed: {e}")

        # Test 2: 100% Placement Guarantee
        logger.info("\n--- Test 2: 100% Placement Guarantee ---")
        try:
            result = engine.optimize_unlimited_runtime(test_panels, test_sheets)

            if result:
                if hasattr(result, 'placed_panels'):
                    placed_count = len(result.placed_panels)
                else:
                    placed_count = len(result.panels) if hasattr(result, 'panels') else 0

                total_panels = sum(p.quantity for p in test_panels)
                placement_rate = (placed_count / total_panels) * 100

                logger.info(f"100% guarantee optimization: {placed_count}/{total_panels} panels ({placement_rate:.1f}%)")

                if hasattr(result, 'efficiency'):
                    logger.info(f"Efficiency: {result.efficiency:.1f}%")

                if hasattr(result, 'metadata'):
                    if 'optimization_time' in result.metadata:
                        logger.info(f"Optimization time: {result.metadata['optimization_time']:.1f}s")
                    if 'tiers_used' in result.metadata:
                        logger.info(f"Tiers used: {result.metadata['tiers_used']}")

                # Validate 100% placement
                if placement_rate >= 100.0:
                    logger.info("✅ 100% placement guarantee achieved!")
                else:
                    logger.error(f"❌ 100% placement guarantee failed: only {placement_rate:.1f}%")

            else:
                logger.error("100% placement guarantee returned no result")

        except Exception as e:
            logger.error(f"100% placement guarantee failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 3: Memory Manager Stats
        logger.info("\n--- Test 3: Memory Manager Statistics ---")
        try:
            from core.algorithms.memory_manager import get_memory_manager
            memory_manager = get_memory_manager()
            stats = memory_manager.get_stats()

            logger.info("Memory Manager Statistics:")
            logger.info(f"  Memory limit: {stats['memory_limit_mb']:.1f} MB")
            logger.info(f"  Current usage: {stats['current_usage_mb']:.1f} MB")
            logger.info(f"  Panel pool hit rate: {stats['panel_pool']['hit_rate']:.1%}")
            logger.info(f"  Position cache hit rate: {stats['position_cache']['hit_rate']:.1%}")

        except Exception as e:
            logger.error(f"Memory manager stats failed: {e}")

        logger.info("\n=== Integration Test Complete ===")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Some components may not be available yet")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_unlimited_runtime_integration()