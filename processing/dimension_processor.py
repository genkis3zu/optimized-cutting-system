#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimension processing system with PI code expansion
PIコード展開による寸法処理システム
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import logging

from models.core_models import Panel
from models.database_models import MaterialDatabase, PiCodeData, MaterialData


@dataclass
class DimensionInput:
    """
    Input data for dimension processing
    寸法処理用の入力データ
    """
    panel_id: str
    pi_code: str
    finished_width: float      # 完成W寸法
    finished_height: float     # 完成H寸法
    quantity: int
    material_type: str
    thickness: float
    priority: int = 1
    block_order: int = 1


@dataclass
class ProcessedPanel:
    """
    Processed panel with expanded dimensions
    展開寸法付きの処理済みパネル
    """
    input_data: DimensionInput
    pi_code_data: Optional[PiCodeData]
    material_data: Optional[MaterialData]
    expanded_width: float
    expanded_height: float
    panel: Panel
    processing_notes: List[str]


class DimensionProcessor:
    """
    Process panel dimensions with PI code expansion
    PIコード展開による寸法処理
    """

    def __init__(self, db: MaterialDatabase):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def process_input_data(self, input_data: List[DimensionInput]) -> List[ProcessedPanel]:
        """
        Process input data with PI code expansion
        PIコード展開による入力データ処理

        Processing flow:
        1. Look up PI code for dimension expansion
        2. Calculate expanded dimensions
        3. Validate material compatibility
        4. Create Panel objects with expanded dimensions
        """
        processed_panels = []

        for input_item in input_data:
            processed = self._process_single_input(input_item)
            if processed:
                processed_panels.append(processed)

        return processed_panels

    def _process_single_input(self, input_item: DimensionInput) -> Optional[ProcessedPanel]:
        """Process single input item"""
        notes = []

        # Look up PI code
        pi_code_data = self.db.get_pi_code(input_item.pi_code)
        if not pi_code_data:
            self.logger.warning(f"PI code not found: {input_item.pi_code}")
            notes.append(f"PI code {input_item.pi_code} not found - using finished dimensions")
            expanded_w = input_item.finished_width
            expanded_h = input_item.finished_height
        else:
            # Calculate expanded dimensions
            expanded_w, expanded_h = pi_code_data.expand_dimensions(
                input_item.finished_width,
                input_item.finished_height
            )
            notes.append(f"PI expansion: {input_item.finished_width}x{input_item.finished_height} -> {expanded_w:.1f}x{expanded_h:.1f}")

        # Look up material data
        material_data = None
        material_types = self.db.get_material_types()
        if input_item.material_type not in material_types:
            self.logger.warning(f"Material type not found in database: {input_item.material_type}")
            notes.append(f"Material {input_item.material_type} not in database - using provided specs")
        else:
            # Try to find matching material
            all_materials = self.db.get_all_materials()
            for mat in all_materials:
                if (mat.material_type == input_item.material_type and
                    abs(mat.thickness - input_item.thickness) < 0.01):
                    material_data = mat
                    break

        # Validate dimensions
        if expanded_w < 50 or expanded_h < 50:
            self.logger.error(f"Panel {input_item.panel_id}: Dimensions too small ({expanded_w}x{expanded_h})")
            notes.append("ERROR: Dimensions below 50mm minimum")
            return None

        if expanded_w > 1500 or expanded_h > 3100:
            self.logger.warning(f"Panel {input_item.panel_id}: Large dimensions ({expanded_w}x{expanded_h})")
            notes.append("WARNING: Dimensions exceed standard sheet size")

        # Create Panel object
        panel = Panel(
            id=input_item.panel_id,
            width=input_item.finished_width,
            height=input_item.finished_height,
            quantity=input_item.quantity,
            material=input_item.material_type,
            thickness=input_item.thickness,
            priority=input_item.priority,
            allow_rotation=True,  # Enable rotation by default
            block_order=input_item.block_order,
            pi_code=input_item.pi_code,
            expanded_width=expanded_w,
            expanded_height=expanded_h
        )

        return ProcessedPanel(
            input_data=input_item,
            pi_code_data=pi_code_data,
            material_data=material_data,
            expanded_width=expanded_w,
            expanded_height=expanded_h,
            panel=panel,
            processing_notes=notes
        )

    def create_input_from_csv_data(self, csv_data: List[Dict[str, str]]) -> List[DimensionInput]:
        """
        Create input data from CSV-like data
        CSV形式データから入力データを作成

        Expected columns:
        - panel_id or id
        - pi_code
        - finished_width or width
        - finished_height or height
        - quantity
        - material_type or material
        - thickness
        """
        input_data = []

        for row in csv_data:
            try:
                # Flexible column mapping
                panel_id = row.get('panel_id', row.get('id', f"panel_{len(input_data)+1}"))
                pi_code = row.get('pi_code', row.get('PI_code', ''))
                finished_width = float(row.get('finished_width', row.get('width', 0)))
                finished_height = float(row.get('finished_height', row.get('height', 0)))
                quantity = int(row.get('quantity', 1))
                material_type = row.get('material_type', row.get('material', 'SGCC'))
                thickness = float(row.get('thickness', 6.0))
                priority = int(row.get('priority', 1))
                block_order = int(row.get('block_order', 1))

                if finished_width <= 0 or finished_height <= 0:
                    self.logger.warning(f"Invalid dimensions for {panel_id}: {finished_width}x{finished_height}")
                    continue

                input_item = DimensionInput(
                    panel_id=panel_id,
                    pi_code=pi_code,
                    finished_width=finished_width,
                    finished_height=finished_height,
                    quantity=quantity,
                    material_type=material_type,
                    thickness=thickness,
                    priority=priority,
                    block_order=block_order
                )

                input_data.append(input_item)

            except (ValueError, KeyError) as e:
                self.logger.error(f"Failed to parse row: {row} - {e}")
                continue

        return input_data

    def validate_processing_results(self, processed_panels: List[ProcessedPanel]) -> Dict[str, any]:
        """
        Validate processing results and generate summary
        処理結果の検証とサマリー生成
        """
        summary = {
            'total_panels': len(processed_panels),
            'valid_panels': 0,
            'panels_with_pi_expansion': 0,
            'panels_with_material_data': 0,
            'material_groups': {},
            'warnings': [],
            'errors': []
        }

        for processed in processed_panels:
            summary['valid_panels'] += 1

            # Count PI expansions
            if processed.pi_code_data:
                summary['panels_with_pi_expansion'] += 1

            # Count material data matches
            if processed.material_data:
                summary['panels_with_material_data'] += 1

            # Group by material
            material_key = f"{processed.panel.material}_{processed.panel.thickness}t"
            if material_key not in summary['material_groups']:
                summary['material_groups'][material_key] = []
            summary['material_groups'][material_key].append(processed.panel.id)

            # Collect warnings and errors
            for note in processed.processing_notes:
                if 'WARNING' in note:
                    summary['warnings'].append(f"{processed.panel.id}: {note}")
                elif 'ERROR' in note:
                    summary['errors'].append(f"{processed.panel.id}: {note}")

        return summary

    def get_panels_for_optimization(self, processed_panels: List[ProcessedPanel]) -> List[Panel]:
        """
        Extract Panel objects for optimization
        最適化用のPanelオブジェクトを抽出
        """
        return [p.panel for p in processed_panels if p.panel]

    def get_material_groups(self, processed_panels: List[ProcessedPanel]) -> Dict[str, List[Panel]]:
        """
        Group panels by material type for batch processing
        材料別バッチ処理用のパネルグループ化
        """
        groups = {}

        for processed in processed_panels:
            panel = processed.panel
            material_key = f"{panel.material}_{panel.thickness}"

            if material_key not in groups:
                groups[material_key] = []
            groups[material_key].append(panel)

        return groups