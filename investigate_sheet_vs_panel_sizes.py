#!/usr/bin/env python3
"""
Investigate Sheet vs Panel Sizes - Root Cause Analysis
シートサイズ vs パネルサイズ調査 - 根本原因分析
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.text_parser import parse_cutting_data_file
from core.pi_manager import PIManager
from core.material_manager import MaterialInventoryManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def investigate_sheet_vs_panel_sizes():
    """Investigate why panels can't be placed despite having diverse sheet options"""
    try:
        logger.info("シートサイズ vs パネルサイズ調査開始...")

        # Parse data
        parse_result = parse_cutting_data_file('sample_data/data0923.txt')
        panels = parse_result.panels
        logger.info(f"パネル種類数: {len(panels)}")

        # Apply PI expansion
        pi_manager = PIManager()
        for panel in panels:
            panel.calculate_expanded_dimensions(pi_manager)

        # Get material manager for sheet information
        material_manager = MaterialInventoryManager()

        # Analyze the largest panels
        logger.info("\\n=== 最大パネルサイズ分析 ===")
        largest_panels = []
        for panel in panels:
            cutting_w = getattr(panel, 'cutting_width', panel.width)
            cutting_h = getattr(panel, 'cutting_height', panel.height)
            largest_panels.append((panel, cutting_w, cutting_h, max(cutting_w, cutting_h)))

        # Sort by largest dimension
        largest_panels.sort(key=lambda x: x[3], reverse=True)

        logger.info("TOP 10 最大パネル:")
        for i, (panel, w, h, max_dim) in enumerate(largest_panels[:10]):
            logger.info(f"  {i+1}. {panel.id}: {w:.0f}x{h:.0f}mm (最大辺: {max_dim:.0f}mm) - {panel.quantity}個")

        # Analyze available sheet sizes
        logger.info("\\n=== 利用可能シートサイズ分析 ===")
        all_sheets = material_manager.inventory

        # Group sheets by size
        sheet_sizes = {}
        for sheet in all_sheets:
            size_key = f"{sheet.width}x{sheet.height}mm"
            if size_key not in sheet_sizes:
                sheet_sizes[size_key] = {'count': 0, 'materials': set(), 'max_dim': max(sheet.width, sheet.height)}
            sheet_sizes[size_key]['count'] += 1
            sheet_sizes[size_key]['materials'].add(sheet.material_type)

        # Sort by sheet size
        sorted_sheets = sorted(sheet_sizes.items(), key=lambda x: x[1]['max_dim'], reverse=True)

        logger.info("利用可能シートサイズ (大きい順):")
        for size_key, info in sorted_sheets:
            materials = ', '.join(sorted(info['materials']))
            logger.info(f"  {size_key}: {info['count']}枚 - 材料: {materials}")

        # Find the problematic panels that can't fit in ANY sheet
        logger.info("\\n=== 配置不可能パネル分析 ===")
        largest_sheet_w = max(sheet.width for sheet in all_sheets)
        largest_sheet_h = max(sheet.height for sheet in all_sheets)
        logger.info(f"最大シートサイズ: {largest_sheet_w}x{largest_sheet_h}mm")

        impossible_panels = []
        for panel in panels:
            cutting_w = getattr(panel, 'cutting_width', panel.width)
            cutting_h = getattr(panel, 'cutting_height', panel.height)

            # Check if panel can fit in largest sheet (with rotation)
            fits_normal = cutting_w <= largest_sheet_w and cutting_h <= largest_sheet_h
            fits_rotated = cutting_h <= largest_sheet_w and cutting_w <= largest_sheet_h if panel.allow_rotation else False

            if not fits_normal and not fits_rotated:
                impossible_panels.append((panel, cutting_w, cutting_h))

        if impossible_panels:
            logger.error(f"物理的配置不可能パネル: {len(impossible_panels)}種類")
            total_impossible_quantity = sum(panel[0].quantity for panel in impossible_panels)
            logger.error(f"物理的配置不可能数量: {total_impossible_quantity}個")

            logger.info("\\n配置不可能パネル詳細:")
            for panel, w, h in impossible_panels:
                logger.error(f"  ❌ {panel.id}: {w:.0f}x{h:.0f}mm - {panel.quantity}個")
                logger.error(f"     理由: 最大シート({largest_sheet_w}x{largest_sheet_h}mm)より大きい")
        else:
            logger.info("✅ すべてのパネルは理論上配置可能")

        # Check if this explains the placement failure
        total_panels = sum(panel.quantity for panel in panels)
        if impossible_panels:
            total_impossible = sum(panel[0].quantity for panel in impossible_panels)
            impossible_rate = (total_impossible / total_panels) * 100
            remaining_rate = 100 - impossible_rate

            logger.info(f"\\n=== 配置可能性分析 ===")
            logger.info(f"物理的配置不可能: {impossible_rate:.1f}%")
            logger.info(f"理論上配置可能: {remaining_rate:.1f}%")

            if remaining_rate >= 83:  # If 83%+ should be placeable theoretically
                logger.warning("⚠️ アルゴリズム効率問題: 理論上は83%以上配置可能なはず")
                logger.warning("   → アルゴリズムの改善が必要")
            else:
                logger.info("📊 物理的制約が主要因: シートサイズ拡張を検討")

        # Check specific problematic panel: 968x617mm
        logger.info("\\n=== 特定パネル調査: 968x617mm ===")
        can_fit_968x617 = False
        compatible_sheets = []

        for sheet in all_sheets:
            fits_normal = 968 <= sheet.width and 617 <= sheet.height
            fits_rotated = 617 <= sheet.width and 968 <= sheet.height

            if fits_normal or fits_rotated:
                can_fit_968x617 = True
                orientation = "通常" if fits_normal else "回転"
                compatible_sheets.append(f"{sheet.width}x{sheet.height}mm({orientation})")

        if can_fit_968x617:
            logger.info(f"✅ 968x617mmパネルは配置可能")
            logger.info(f"対応シート: {', '.join(set(compatible_sheets))}")
        else:
            logger.error(f"❌ 968x617mmパネルは物理的配置不可能")

        return impossible_panels is None or len(impossible_panels) == 0

    except Exception as e:
        logger.error(f"調査エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = investigate_sheet_vs_panel_sizes()
    sys.exit(0 if success else 1)