#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed test to understand the 968x712 placement results
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from processing.phase1_individual import IndividualMaterialProcessor
from models.core_models import Panel

def detailed_test():
    """Detailed analysis of 968x712 placement"""
    print("=" * 60)
    print("Detailed Analysis: 968x712 Case")
    print("=" * 60)
    
    # Create test panel
    test_panel = Panel(
        id="test_968x712",
        width=968,
        height=712,
        quantity=12,
        material="SGCC",
        thickness=6.0
    )
    
    processor = IndividualMaterialProcessor()
    result = processor.process_all_materials([test_panel])
    
    # Analyze all results (both high and low efficiency)
    all_results = result.high_efficiency_results + result.low_efficiency_results
    
    if all_results:
        batch = all_results[0]
        print(f"Batch Details:")
        print(f"  Material: {batch.batch.material_type}")
        print(f"  Efficiency: {batch.efficiency:.1%}")
        print(f"  Sheets used: {batch.total_sheets_used}")
        print(f"  Algorithm: {batch.algorithm_used}")
        print()
        
        # Analyze each sheet
        for i, placement in enumerate(batch.placement_results):
            print(f"Sheet {i+1}:")
            print(f"  Panels placed: {len(placement.panels)}")
            print(f"  Sheet efficiency: {placement.efficiency:.1%}")
            print(f"  Algorithm: {placement.algorithm}")
            print(f"  Sheet size: {placement.sheet.width}x{placement.sheet.height}mm")
            print(f"  Sheet area: {placement.sheet.area:,.0f}mm²")
            print(f"  Waste area: {placement.waste_area:,.0f}mm²")
            
            # Calculate panel areas
            used_area = sum(p.actual_width * p.actual_height for p in placement.panels)
            print(f"  Used area: {used_area:,.0f}mm²")
            
            # Show panel positions
            for j, panel in enumerate(placement.panels):
                print(f"    Panel {j+1}: pos=({panel.x:.0f},{panel.y:.0f}) "
                      f"size={panel.actual_width:.0f}x{panel.actual_height:.0f}")
            print()
        
        # Expected vs actual analysis
        sheet_area = 1500 * 3100  # 4,650,000 mm²
        panel_area = 968 * 712    # 689,216 mm²
        
        print(f"Theoretical Analysis:")
        print(f"  Sheet area: {sheet_area:,.0f}mm²")
        print(f"  Panel area: {panel_area:,.0f}mm²")
        print(f"  Panels per sheet (theory): {sheet_area // panel_area}")
        print(f"  Max theoretical efficiency: {(panel_area * 4) / sheet_area:.1%}")
        
        # Check if 4 panels fit with kerf
        kerf = 3.0
        panels_width = int((1500 + kerf) // (968 + kerf))
        panels_height = int((3100 + kerf) // (712 + kerf))
        total_with_kerf = panels_width * panels_height
        
        print(f"  With kerf ({kerf}mm):")
        print(f"    Panels in width: {panels_width}")
        print(f"    Panels in height: {panels_height}")
        print(f"    Total: {total_with_kerf}")
        
        efficiency_with_kerf = (panel_area * total_with_kerf) / sheet_area
        print(f"    Expected efficiency: {efficiency_with_kerf:.1%}")

if __name__ == "__main__":
    detailed_test()