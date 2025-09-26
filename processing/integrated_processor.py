#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated processing system implementing the full workflow
完全なワークフローを実装する統合処理システム

Processing flow:
inputデータ → PIコード突き合わせ → 展開寸法計算 → 材料別バッチ最適化 → 低効率組み合わせ最適化
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from models.database_models import MaterialDatabase
from models.core_models import Panel, SteelSheet
from models.batch_models import MaterialBatch, BatchOptimizationResult, ProcessingResult
from processing.dimension_processor import DimensionProcessor, DimensionInput, ProcessedPanel
from processing.phase1_individual import IndividualMaterialProcessor
from core.batch_optimizer import MaterialBatchOptimizer


@dataclass
class IntegratedProcessingResult:
    """
    Complete result of integrated processing workflow
    統合処理ワークフローの完全な結果
    """
    input_summary: Dict[str, any]
    dimension_processing_summary: Dict[str, any]
    individual_optimization_result: BatchOptimizationResult
    residual_optimization_result: Optional[BatchOptimizationResult] = None
    overall_efficiency: float = 0.0
    total_sheets_used: int = 0
    total_waste_reduction: float = 0.0
    processing_time: float = 0.0


class IntegratedProcessor:
    """
    Integrated processor implementing the complete workflow with zero kerf/safety margins
    kerfやsafety cutを0とした完全ワークフローの統合プロセッサ
    """

    def __init__(self, db_path: str = "data/materials.db"):
        self.db = MaterialDatabase(db_path)
        self.dimension_processor = DimensionProcessor(self.db)
        self.individual_processor = IndividualMaterialProcessor()
        self.batch_optimizer = MaterialBatchOptimizer()
        self.logger = logging.getLogger(__name__)

        # User specified: kerfやsafety cutは考慮しなくて良いです。0でOKです。
        self.kerf_width = 0.0
        self.safety_margin = 0.0

    def process_complete_workflow(self, input_data: List[DimensionInput]) -> IntegratedProcessingResult:
        """
        Process complete workflow from input to optimized results
        入力から最適化結果までの完全ワークフロー処理

        Processing steps:
        1. Input data validation and PI code dimension expansion
        2. Material-based batching and individual optimization
        3. Efficiency evaluation and residual material identification
        4. Combined optimization for improved yield
        """
        start_time = time.time()

        self.logger.info(f"Starting integrated processing workflow for {len(input_data)} input items")

        # Step 1: Dimension processing with PI code expansion
        processed_panels = self.dimension_processor.process_input_data(input_data)
        dimension_summary = self.dimension_processor.validate_processing_results(processed_panels)

        self.logger.info(f"Dimension processing completed: {dimension_summary['valid_panels']} valid panels")

        # Step 2: Extract panels and group by material
        panels = self.dimension_processor.get_panels_for_optimization(processed_panels)
        if not panels:
            self.logger.error("No valid panels after dimension processing")
            return IntegratedProcessingResult(
                input_summary={'error': 'No valid panels after dimension processing'},
                dimension_processing_summary=dimension_summary,
                individual_optimization_result=BatchOptimizationResult(),
                processing_time=time.time() - start_time
            )

        # Step 3: Individual material optimization (Phase 1)
        individual_result = self.individual_processor.process_all_materials(panels)
        individual_result.classify_results()

        self.logger.info(f"Individual optimization: {len(individual_result.high_efficiency_results)} high efficiency, "
                        f"{len(individual_result.low_efficiency_results)} low efficiency batches")

        # Step 4: Residual material combination optimization if needed
        residual_result = None
        if individual_result.low_efficiency_results:
            residual_result = self._optimize_residual_materials(individual_result.low_efficiency_results)
            self.logger.info(f"Residual optimization completed with {residual_result.overall_efficiency:.1%} efficiency")

        # Step 5: Calculate overall metrics
        overall_result = self._calculate_overall_metrics(
            individual_result, residual_result, dimension_summary
        )

        processing_time = time.time() - start_time
        overall_result.processing_time = processing_time

        self.logger.info(f"Integrated processing completed in {processing_time:.2f}s")
        return overall_result

    def _optimize_residual_materials(self, low_efficiency_results: List[ProcessingResult]) -> BatchOptimizationResult:
        """
        Optimize residual materials from low efficiency results
        低効率結果からの余り材最適化
        """
        # Extract unplaced panels and create combined batches
        all_unplaced_panels = []

        for result in low_efficiency_results:
            all_unplaced_panels.extend(result.unplaced_panels)

            # Also extract panels from low efficiency placements that could be re-optimized
            for placement_result in result.placement_results:
                if placement_result.efficiency < 0.6:  # Very low efficiency
                    # Extract panels for re-optimization
                    for placed_panel in placement_result.panels:
                        panel = Panel(
                            id=placed_panel.panel.id + "_reopt",
                            width=placed_panel.panel.width,
                            height=placed_panel.panel.height,
                            quantity=1,
                            material=placed_panel.panel.material,
                            thickness=placed_panel.panel.thickness,
                            priority=placed_panel.panel.priority,
                            allow_rotation=placed_panel.panel.allow_rotation,
                            block_order=placed_panel.panel.block_order,
                            pi_code=placed_panel.panel.pi_code,
                            expanded_width=placed_panel.panel.expanded_width,
                            expanded_height=placed_panel.panel.expanded_height
                        )
                        all_unplaced_panels.append(panel)

        if not all_unplaced_panels:
            return BatchOptimizationResult()

        # Re-process with combined approach
        return self.individual_processor.process_all_materials(all_unplaced_panels)

    def _calculate_overall_metrics(
        self,
        individual_result: BatchOptimizationResult,
        residual_result: Optional[BatchOptimizationResult],
        dimension_summary: Dict[str, any]
    ) -> IntegratedProcessingResult:
        """
        Calculate overall processing metrics
        全体処理メトリクスの計算
        """
        # Combine all results
        all_results = individual_result.processing_results.copy()
        if residual_result:
            all_results.extend(residual_result.processing_results)

        # Calculate totals
        total_sheets = sum(len(result.placement_results) for result in all_results)
        total_placed_area = sum(
            sum(
                sum(panel.actual_width * panel.actual_height for panel in placement.panels)
                for placement in result.placement_results
            )
            for result in all_results
        )
        total_sheet_area = sum(
            sum(placement.sheet.area for placement in result.placement_results)
            for result in all_results
        )

        overall_efficiency = total_placed_area / total_sheet_area if total_sheet_area > 0 else 0.0

        # Calculate waste reduction compared to no residual optimization
        base_efficiency = individual_result.overall_efficiency
        waste_reduction = 0.0
        if residual_result and residual_result.overall_efficiency > base_efficiency:
            waste_reduction = residual_result.overall_efficiency - base_efficiency

        # Create input summary
        input_summary = {
            'total_input_items': dimension_summary['total_panels'],
            'valid_panels': dimension_summary['valid_panels'],
            'material_groups': len(dimension_summary['material_groups']),
            'pi_expansions': dimension_summary['panels_with_pi_expansion'],
            'warnings': len(dimension_summary['warnings']),
            'errors': len(dimension_summary['errors'])
        }

        return IntegratedProcessingResult(
            input_summary=input_summary,
            dimension_processing_summary=dimension_summary,
            individual_optimization_result=individual_result,
            residual_optimization_result=residual_result,
            overall_efficiency=overall_efficiency,
            total_sheets_used=total_sheets,
            total_waste_reduction=waste_reduction
        )

    def create_standard_sheet_template(self, material_type: str, thickness: float) -> SteelSheet:
        """
        Create standard sheet template (1500x3100mm)
        標準シートテンプレート作成（1500x3100mm）
        """
        return SteelSheet(
            width=1500.0,
            height=3100.0,
            thickness=thickness,
            material=material_type,
            cost_per_sqm=100.0  # Default cost
        )

    def process_csv_input(self, csv_data: List[Dict[str, str]]) -> IntegratedProcessingResult:
        """
        Process CSV-formatted input data
        CSV形式入力データの処理
        """
        input_data = self.dimension_processor.create_input_from_csv_data(csv_data)
        return self.process_complete_workflow(input_data)

    def process_manual_input(
        self,
        panels_data: List[Dict[str, any]],
        default_pi_code: str = ""
    ) -> IntegratedProcessingResult:
        """
        Process manually entered panel data
        手動入力パネルデータの処理
        """
        input_data = []

        for i, panel_data in enumerate(panels_data):
            try:
                input_item = DimensionInput(
                    panel_id=panel_data.get('id', f"panel_{i+1}"),
                    pi_code=panel_data.get('pi_code', default_pi_code),
                    finished_width=float(panel_data['width']),
                    finished_height=float(panel_data['height']),
                    quantity=int(panel_data.get('quantity', 1)),
                    material_type=panel_data['material'],
                    thickness=float(panel_data['thickness']),
                    priority=int(panel_data.get('priority', 1)),
                    block_order=int(panel_data.get('block_order', 1))
                )
                input_data.append(input_item)

            except (KeyError, ValueError) as e:
                self.logger.warning(f"Failed to process panel data {i}: {e}")
                continue

        return self.process_complete_workflow(input_data)

    def get_processing_report(self, result: IntegratedProcessingResult) -> Dict[str, any]:
        """
        Generate comprehensive processing report
        包括的な処理レポートを生成
        """
        return {
            'summary': {
                'overall_efficiency': f"{result.overall_efficiency:.1%}",
                'total_sheets_used': result.total_sheets_used,
                'processing_time': f"{result.processing_time:.2f}s",
                'waste_reduction': f"{result.total_waste_reduction:.1%}" if result.total_waste_reduction > 0 else "N/A"
            },
            'input_processing': result.input_summary,
            'dimension_expansion': result.dimension_processing_summary,
            'individual_optimization': {
                'high_efficiency_batches': len(result.individual_optimization_result.high_efficiency_results),
                'low_efficiency_batches': len(result.individual_optimization_result.low_efficiency_results),
                'base_efficiency': f"{result.individual_optimization_result.overall_efficiency:.1%}"
            },
            'residual_optimization': {
                'performed': result.residual_optimization_result is not None,
                'efficiency_improvement': f"{result.total_waste_reduction:.1%}" if result.total_waste_reduction > 0 else "0%"
            } if result.residual_optimization_result else None
        }