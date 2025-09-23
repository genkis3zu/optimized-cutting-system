#!/usr/bin/env python3
"""
Investigate Remaining Unplaced Panels - Deep Analysis
残りの未配置パネルの詳細調査
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
from core.material_manager import MaterialInventoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def investigate_unplaced_panels():
    """Investigate why 32% of panels remain unplaced despite having correct sheet sizes"""
    try:
        logger.info("未配置パネルの詳細調査開始...")

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

        # Run optimization
        constraints = OptimizationConstraints(
            material_separation=False,
            max_sheets=1000,
            time_budget=0.0,
            kerf_width=0.0,
            target_efficiency=0.1
        )

        engine = OptimizationEngine()
        engine.register_algorithm(ImprovedFirstFitDecreasing())

        logger.info("最適化実行...")
        results = engine.optimize(panels, constraints, 'Improved_FFD')

        # Collect placed panel IDs
        placed_panel_ids = set()
        for result in results:
            for placed_panel in result.panels:
                placed_panel_ids.add(placed_panel.panel.id)

        total_placed = len(placed_panel_ids)
        placement_rate = (total_placed / total_panels) * 100

        logger.info(f"配置結果: {total_placed}/{total_panels} ({placement_rate:.1f}%)")

        # Analyze unplaced panels by material and size
        material_manager = MaterialInventoryManager()

        unplaced_analysis = {}
        for panel in panels:
            base_id = panel.id
            panel_width = getattr(panel, 'cutting_width', panel.width)
            panel_height = getattr(panel, 'cutting_height', panel.height)
            normalized_material = material_manager.normalize_material_code(panel.material)

            # Count placed vs total for this panel type
            placed_count = sum(1 for pid in placed_panel_ids if pid.startswith(f"{base_id}_"))
            unplaced_count = panel.quantity - placed_count

            if unplaced_count > 0:
                # Find available sheets for this panel
                compatible_sheets = material_manager.find_compatible_sheets(
                    normalized_material, panel.thickness, panel_width, panel_height
                )

                if panel.allow_rotation:
                    rotated_compatible = material_manager.find_compatible_sheets(
                        normalized_material, panel.thickness, panel_height, panel_width
                    )
                    compatible_sheets.extend(rotated_compatible)

                unplaced_analysis[base_id] = {
                    'panel': panel,
                    'placed_count': placed_count,
                    'unplaced_count': unplaced_count,
                    'total_quantity': panel.quantity,
                    'size': f"{panel_width:.1f}x{panel_height:.1f}mm",
                    'material': normalized_material,
                    'compatible_sheets': len(compatible_sheets),
                    'can_fit_somewhere': len(compatible_sheets) > 0
                }

        # Group analysis by reason
        logger.info(f"\n=== 未配置パネル分析 ===")

        truly_unplaceable = []
        algorithm_failures = []

        for panel_id, analysis in unplaced_analysis.items():
            if not analysis['can_fit_somewhere']:
                truly_unplaceable.append(analysis)
            else:
                algorithm_failures.append(analysis)

        logger.info(f"真の配置不可能パネル: {len(truly_unplaceable)}種類")
        for analysis in truly_unplaceable:
            logger.error(f"  ❌ {analysis['panel'].id}: {analysis['size']} - 物理的配置不可能")

        logger.info(f"\nアルゴリズム配置失敗パネル: {len(algorithm_failures)}種類")
        total_algorithm_failures = sum(a['unplaced_count'] for a in algorithm_failures)

        logger.info(f"アルゴリズム配置失敗数: {total_algorithm_failures}個")

        # Show most problematic panels
        algorithm_failures.sort(key=lambda x: x['unplaced_count'], reverse=True)

        logger.info(f"\n=== 最も配置失敗の多いパネル（TOP 10）===")
        for i, analysis in enumerate(algorithm_failures[:10]):
            logger.warning(f"  {i+1}. {analysis['panel'].id}: {analysis['unplaced_count']}/{analysis['total_quantity']} 未配置")
            logger.warning(f"      サイズ: {analysis['size']}, 材料: {analysis['material']}")
            logger.warning(f"      対応シート数: {analysis['compatible_sheets']}種類")

            # Show which sheets can accommodate this panel
            compatible_sheets = material_manager.find_compatible_sheets(
                analysis['material'], analysis['panel'].thickness,
                analysis['panel'].cutting_width, analysis['panel'].cutting_height
            )

            if compatible_sheets:
                best_sheets = sorted(compatible_sheets, key=lambda s: s.area)[:3]
                sheet_info = ", ".join([f"{s.width}x{s.height}mm" for s in best_sheets])
                logger.warning(f"      最適シート: {sheet_info}")

        # Check for specific patterns
        logger.info(f"\n=== パターン分析 ===")

        # Large panels
        large_panels = [a for a in algorithm_failures if max(a['panel'].cutting_width, a['panel'].cutting_height) > 1000]
        if large_panels:
            logger.warning(f"大型パネル（1000mm超）配置失敗: {len(large_panels)}種類")

        # High quantity panels
        high_qty_panels = [a for a in algorithm_failures if a['panel'].quantity > 10]
        if high_qty_panels:
            logger.warning(f"高数量パネル（10個超）配置失敗: {len(high_qty_panels)}種類")

        # Material-specific issues
        material_issues = {}
        for analysis in algorithm_failures:
            material = analysis['material']
            if material not in material_issues:
                material_issues[material] = 0
            material_issues[material] += analysis['unplaced_count']

        logger.info(f"\n=== 材料別配置失敗数 ===")
        for material, count in sorted(material_issues.items(), key=lambda x: x[1], reverse=True):
            logger.warning(f"  {material}: {count}個")

        return len(truly_unplaceable) == 0 and total_algorithm_failures < 50

    except Exception as e:
        logger.error(f"調査エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = investigate_unplaced_panels()
    sys.exit(0 if success else 1)