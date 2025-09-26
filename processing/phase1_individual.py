#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Individual material processing for optimal batch optimization
Phase 1: 個別材料処理による最適バッチ最適化
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from models.core_models import Panel, SteelSheet, OptimizationType
from models.batch_models import MaterialBatch, ProcessingResult, BatchOptimizationResult, BatchStatus
from core.batch_optimizer import MaterialBatchOptimizer


@dataclass
class MaterialGroup:
    """
    Group of panels with same material type and thickness
    同一材料種類・厚さのパネルグループ
    """
    material_type: str
    thickness: float
    panels: List[Panel]
    
    @property
    def total_panels(self) -> int:
        return sum(panel.quantity for panel in self.panels)
    
    @property
    def total_area(self) -> float:
        return sum(panel.cutting_area * panel.quantity for panel in self.panels)


class IndividualMaterialProcessor:
    """
    Process each material type individually for optimal efficiency
    各材料種類を個別に処理して最適効率を実現
    """
    
    def __init__(self, sheet_templates: Optional[Dict[str, SteelSheet]] = None):
        self.batch_optimizer = MaterialBatchOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Default sheet templates by material type
        self.sheet_templates = sheet_templates or {
            'SGCC': SteelSheet(width=1500, height=3100, thickness=6.0, material='SGCC'),
            'SPCC': SteelSheet(width=1500, height=3100, thickness=6.0, material='SPCC'),
            'SS400': SteelSheet(width=1500, height=3100, thickness=6.0, material='SS400'),
        }
        
    def group_panels_by_material(self, panels: List[Panel]) -> List[MaterialGroup]:
        """
        Group panels by material type and thickness
        材料種類と厚さでパネルをグループ化
        """
        material_groups = {}
        
        for panel in panels:
            key = (panel.material, panel.thickness)
            if key not in material_groups:
                material_groups[key] = MaterialGroup(
                    material_type=panel.material,
                    thickness=panel.thickness,
                    panels=[]
                )
            material_groups[key].panels.append(panel)
        
        groups = list(material_groups.values())
        
        # Sort by total area (largest first for better optimization)
        groups.sort(key=lambda g: g.total_area, reverse=True)
        
        self.logger.info(f"Created {len(groups)} material groups")
        for group in groups:
            self.logger.info(f"  {group.material_type}-{group.thickness}mm: "
                           f"{group.total_panels} panels, {group.total_area:.0f}mm²")
        
        return groups
    
    def process_material_group(
        self, 
        group: MaterialGroup, 
        kerf_width: float = 3.0
    ) -> ProcessingResult:
        """
        Process a single material group using optimal strategy
        単一材料グループを最適戦略で処理
        """
        # Get appropriate sheet template
        sheet_template = self.sheet_templates.get(
            group.material_type,
            SteelSheet(material=group.material_type, thickness=group.thickness)
        )
        
        # Create material batch
        batch = MaterialBatch(
            material_type=group.material_type,
            thickness=group.thickness,
            panels=group.panels,
            sheet_template=sheet_template
        )
        
        # Optimize the batch
        self.logger.info(f"Processing material group: {group.material_type}-{group.thickness}mm")
        self.logger.info(f"  Panels: {len(group.panels)} types, {group.total_panels} total")
        self.logger.info(f"  Strategy: {batch.determine_optimization_type().value}")
        
        result = self.batch_optimizer.optimize_batch(batch, kerf_width)
        
        self.logger.info(f"  Result: {result.efficiency:.1%} efficiency, "
                        f"{len(result.placement_results)} sheets, "
                        f"{len(result.unplaced_panels)} unplaced")
        
        return result
    
    def process_all_materials(
        self, 
        panels: List[Panel], 
        kerf_width: float = 3.0
    ) -> BatchOptimizationResult:
        """
        Process all materials individually and return comprehensive results
        全材料を個別処理し、包括的結果を返す
        
        Example: For user's case of 968x712 panels:
        - Creates SGCC material group
        - Uses SimpleMathCalculator for vertical stacking
        - Should achieve high efficiency (4 panels per sheet)
        """
        start_time = time.time()
        
        # Group panels by material
        material_groups = self.group_panels_by_material(panels)
        
        # Process each group individually
        processing_results = []
        
        for group in material_groups:
            try:
                result = self.process_material_group(group, kerf_width)
                processing_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process group {group.material_type}-{group.thickness}mm: {e}")
                # Create failed result
                failed_batch = MaterialBatch(
                    material_type=group.material_type,
                    thickness=group.thickness,
                    panels=group.panels,
                    sheet_template=self.sheet_templates.get(group.material_type, SteelSheet())
                )
                failed_result = ProcessingResult(
                    batch=failed_batch,
                    placement_results=[],
                    efficiency=0.0,
                    unplaced_panels=group.panels,
                    algorithm_used="FAILED",
                    status=BatchStatus.FAILED
                )
                processing_results.append(failed_result)
        
        # Create comprehensive result
        optimization_result = BatchOptimizationResult(
            processing_results=processing_results,
            total_processing_time=time.time() - start_time
        )
        
        # Classify results by efficiency
        optimization_result.classify_results(efficiency_threshold=0.7)
        
        self.logger.info(f"Processing complete: {len(optimization_result.high_efficiency_results)} high efficiency, "
                        f"{len(optimization_result.low_efficiency_results)} low efficiency batches")
        
        return optimization_result
    
    def demonstrate_simple_math_case(self, panel_width: float = 968, panel_height: float = 712):
        """
        Demonstrate the simple math case mentioned by user
        ユーザーが言及した単純計算ケースをデモンストレーション
        
        Example: 968x712 panels should fit 4 pieces vertically in 1500x3100 sheet
        """
        # Create test panel
        test_panel = Panel(
            id="demo_968x712",
            width=panel_width,
            height=panel_height,
            quantity=10,  # Try to optimize 10 panels
            material="SGCC",
            thickness=6.0
        )
        
        self.logger.info(f"Demonstrating simple math case: {panel_width}x{panel_height} panels")
        
        # Process through the system
        result = self.process_all_materials([test_panel])
        
        if result.high_efficiency_results:
            batch_result = result.high_efficiency_results[0]
            self.logger.info(f"Demo result: {batch_result.efficiency:.1%} efficiency")
            
            for placement in batch_result.placement_results:
                panels_per_sheet = len(placement.panels)
                self.logger.info(f"  Sheet {placement.sheet_id}: {panels_per_sheet} panels placed")
                self.logger.info(f"  Algorithm: {placement.algorithm}")
                
                # Show panel positions
                for i, placed_panel in enumerate(placement.panels):
                    self.logger.info(f"    Panel {i+1}: ({placed_panel.x:.0f}, {placed_panel.y:.0f}) "
                                   f"size: {placed_panel.actual_width:.0f}x{placed_panel.actual_height:.0f}")
        
        return result