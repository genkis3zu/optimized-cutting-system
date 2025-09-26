#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final test showing the 968x712 case works correctly
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from processing.phase1_individual import IndividualMaterialProcessor
from models.core_models import Panel

def final_test():
    """Final validation test for 968x712 case"""
    print("=" * 60)
    print("FINAL TEST: 968x712 Simple Math Optimization")
    print("=" * 60)
    
    # Create test panel as mentioned by user
    test_panel = Panel(
        id="user_example_968x712",
        width=968,
        height=712,
        quantity=12,  # Test with 12 panels
        material="SGCC",
        thickness=6.0
    )
    
    print(f"Input Panel: {test_panel.cutting_width}x{test_panel.cutting_height}mm")
    print(f"Quantity: {test_panel.quantity}")
    print(f"Material: {test_panel.material}")
    print()
    
    # Process using new system
    processor = IndividualMaterialProcessor()
    result = processor.process_all_materials([test_panel])
    
    # Results
    all_batches = result.high_efficiency_results + result.low_efficiency_results
    if all_batches:
        batch = all_batches[0]
        
        print("RESULTS:")
        print(f"  Algorithm Used: {batch.algorithm_used}")
        print(f"  Overall Efficiency: {batch.efficiency:.1%}")
        print(f"  Sheets Used: {batch.total_sheets_used}")
        print(f"  Total Panels Placed: {sum(len(p.panels) for p in batch.placement_results)}")
        print(f"  Unplaced Panels: {len(batch.unplaced_panels)}")
        print()
        
        # Analyze first sheet (should show the 4-panel pattern)
        if batch.placement_results:
            first_sheet = batch.placement_results[0]
            print("FIRST SHEET DETAILS:")
            print(f"  Panels per sheet: {len(first_sheet.panels)}")
            print(f"  Sheet algorithm: {first_sheet.algorithm}")
            print(f"  Sheet efficiency: {first_sheet.efficiency:.1%}")
            
            # Show panel layout
            print("  Panel Layout:")
            for i, panel in enumerate(first_sheet.panels):
                print(f"    Panel {i+1}: position ({panel.x:.0f}, {panel.y:.0f}) "
                      f"size {panel.actual_width:.0f}x{panel.actual_height:.0f}mm")
            print()
            
        print("VALIDATION:")
        expected_panels_per_sheet = 4
        actual_panels_per_sheet = len(first_sheet.panels) if batch.placement_results else 0
        
        print(f"  Expected panels per sheet: {expected_panels_per_sheet}")
        print(f"  Actual panels per sheet: {actual_panels_per_sheet}")
        
        if actual_panels_per_sheet == expected_panels_per_sheet:
            print("  Result: SUCCESS - User's expectation met!")
        else:
            print(f"  Result: MISMATCH - Got {actual_panels_per_sheet} instead of {expected_panels_per_sheet}")
        
        if "SimpleMath" in batch.algorithm_used:
            print("  Algorithm: SUCCESS - Using simple math as intended")
        else:
            print(f"  Algorithm: ISSUE - Using {batch.algorithm_used} instead of SimpleMath")
        
        # Efficiency analysis
        theoretical_efficiency = (968 * 712 * 4) / (1500 * 3100)
        print(f"  Theoretical max efficiency: {theoretical_efficiency:.1%}")
        print(f"  Actual efficiency: {batch.efficiency:.1%}")
        
        if batch.efficiency >= 0.59:  # Close to theoretical
            print("  Efficiency: GOOD - Close to theoretical maximum")
        else:
            print("  Efficiency: NEEDS REVIEW")
            
    else:
        print("ERROR: No processing results found")

if __name__ == "__main__":
    final_test()