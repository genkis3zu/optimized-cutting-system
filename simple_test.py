#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for 968x712 case without unicode
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from processing.phase1_individual import IndividualMaterialProcessor
from models.core_models import Panel

def test_968_case():
    """Test the specific 968x712 case"""
    print("=" * 60)
    print("Testing 968x712 Simple Math Case")
    print("Expected: 4 panels per sheet (1x4 grid)")
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
    
    print(f"Panel: {test_panel.cutting_width}x{test_panel.cutting_height}mm, qty: {test_panel.quantity}")
    
    # Process
    processor = IndividualMaterialProcessor()
    result = processor.process_all_materials([test_panel])
    
    print(f"Overall efficiency: {result.overall_efficiency:.1%}")
    print(f"High efficiency batches: {len(result.high_efficiency_results)}")
    print(f"Low efficiency batches: {len(result.low_efficiency_results)}")
    
    if result.high_efficiency_results:
        batch = result.high_efficiency_results[0]
        print(f"Batch efficiency: {batch.efficiency:.1%}")
        print(f"Algorithm: {batch.algorithm_used}")
        print(f"Sheets used: {batch.total_sheets_used}")
        print(f"Unplaced panels: {len(batch.unplaced_panels)}")
        
        if batch.placement_results:
            first_sheet = batch.placement_results[0]
            print(f"First sheet: {len(first_sheet.panels)} panels placed")
            print(f"Sheet algorithm: {first_sheet.algorithm}")
            
            # Verify expected results
            expected_per_sheet = 4
            actual_per_sheet = len(first_sheet.panels)
            print(f"Expected per sheet: {expected_per_sheet}")
            print(f"Actual per sheet: {actual_per_sheet}")
            print(f"Match: {'YES' if actual_per_sheet == expected_per_sheet else 'NO'}")
            
    elif result.low_efficiency_results:
        batch = result.low_efficiency_results[0]
        print(f"Low efficiency batch: {batch.efficiency:.1%}")
        print(f"Algorithm: {batch.algorithm_used}")
        print(f"Unplaced panels: {len(batch.unplaced_panels)}")

if __name__ == "__main__":
    test_968_case()