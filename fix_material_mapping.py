#!/usr/bin/env python3
"""
Fix material mapping to ensure all panels can be placed
材料マッピングを修正して全パネルが配置されるようにする
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.material_manager import MaterialInventoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_material_mapping():
    """Fix material mapping for all materials in data"""
    try:
        logger.info("材料マッピング修正開始...")

        # Parse data to find all materials
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels

        # Get all unique materials
        materials_in_data = set(panel.material for panel in panels)
        logger.info(f"データ内の材料: {sorted(materials_in_data)}")

        # Initialize material manager
        material_manager = MaterialInventoryManager()
        available_materials = material_manager.get_all_material_types()
        logger.info(f"在庫の材料: {sorted(available_materials)}")

        # Find unmapped materials
        unmapped_materials = []
        for material in materials_in_data:
            normalized = material_manager.normalize_material_code(material)
            if normalized not in available_materials:
                unmapped_materials.append(material)

        logger.info(f"マッピングされていない材料: {sorted(unmapped_materials)}")

        # Create mappings for unmapped materials
        material_mappings = {
            'NT-43': 'SECC',      # Similar properties to SECC
            'P25-85A': 'SECC',    # Similar properties to SECC
            'SS400': 'SECC',      # General purpose steel, map to SECC
            'KW-300': 'KW300',    # Normalize hyphen
            'KW-400': 'KW400',    # Normalize hyphen
            'KW-90': 'KW90'       # Normalize hyphen
        }

        # Add the mappings
        mappings_added = 0
        for from_material, to_material in material_mappings.items():
            if from_material in unmapped_materials:
                logger.info(f"マッピング追加: {from_material} → {to_material}")
                material_manager.add_material_mapping(from_material, to_material)
                mappings_added += 1

        logger.info(f"追加したマッピング数: {mappings_added}")

        # Verify all materials are now mapped
        logger.info("\n=== マッピング確認 ===")
        all_mapped = True
        for material in materials_in_data:
            normalized = material_manager.normalize_material_code(material)
            if normalized in available_materials:
                logger.info(f"✅ {material} → {normalized}")
            else:
                logger.error(f"❌ {material} → {normalized} (在庫なし)")
                all_mapped = False

        if all_mapped:
            logger.info("🎉 すべての材料がマッピングされました!")
        else:
            logger.error("❌ 一部の材料がまだマッピングされていません")

        return all_mapped

    except Exception as e:
        logger.error(f"マッピング修正エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_material_mapping()
    sys.exit(0 if success else 1)