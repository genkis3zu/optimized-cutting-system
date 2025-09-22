"""
Work Instruction Generation for Steel Cutting Operations
鋼板切断作業用作業指示生成

Generates step-by-step cutting instructions with guillotine constraints
ギロチン制約下でのステップバイステップ切断指示を生成
"""

import time
import logging
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.models import PlacementResult, PlacedPanel, SteelSheet


class CutType(Enum):
    """Cut type enumeration"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    INITIAL = "initial"  # First cut to establish working area


class SafetyLevel(Enum):
    """Safety level for cutting operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CuttingInstruction:
    """
    Individual cutting instruction step
    個別切断指示ステップ
    """
    step_number: int
    cut_type: CutType
    start_point: Tuple[float, float]  # (x, y) in mm
    end_point: Tuple[float, float]    # (x, y) in mm
    dimension: float                   # Cut length in mm
    description: str                   # Human-readable description
    safety_level: SafetyLevel
    estimated_time: float             # Estimated time in minutes
    remaining_pieces: List[str]       # Description of remaining pieces
    target_panels: List[str]          # Panel IDs this cut will create
    warnings: List[str] = field(default_factory=list)

    @property
    def cut_length(self) -> float:
        """Calculate actual cut length"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    @property
    def is_edge_to_edge(self) -> bool:
        """Check if cut goes from edge to edge (guillotine constraint)"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point

        # For guillotine cuts, either x1==x2 (vertical) or y1==y2 (horizontal)
        return abs(x1 - x2) < 0.1 or abs(y1 - y2) < 0.1


@dataclass
class WorkInstruction:
    """
    Complete work instruction document
    完全な作業指示書
    """
    sheet_id: int
    material_type: str
    sheet_dimensions: Tuple[float, float]  # (width, height)
    total_steps: int
    cutting_sequence: List[CuttingInstruction]
    quality_checkpoints: List[str]
    safety_notes: List[str]
    machine_settings: Dict[str, Any]
    estimated_total_time: float        # Total time in minutes
    kerf_width: float                  # Cutting allowance
    generated_at: datetime
    generated_by: str = "Optimization System"

    @property
    def total_cut_length(self) -> float:
        """Calculate total cutting length"""
        return sum(inst.cut_length for inst in self.cutting_sequence)

    @property
    def complexity_score(self) -> float:
        """Calculate complexity score (0-1, higher is more complex)"""
        if not self.cutting_sequence:
            return 0.0

        # Factors affecting complexity
        step_count_factor = min(len(self.cutting_sequence) / 20, 1.0)
        cut_length_factor = min(self.total_cut_length / 10000, 1.0)
        safety_factor = sum(1 for inst in self.cutting_sequence
                          if inst.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]) / len(self.cutting_sequence)

        return (step_count_factor * 0.4 + cut_length_factor * 0.3 + safety_factor * 0.3)


class WorkInstructionGenerator:
    """
    Generator for work instructions from placement results
    配置結果からの作業指示生成器
    """

    def __init__(self, kerf_width: float = 3.5):
        self.kerf_width = kerf_width
        self.logger = logging.getLogger(__name__)

        # Cutting time estimation (minutes per meter)
        self.cutting_speed = {
            'SS400': 2.0,    # Carbon steel
            'SUS304': 3.0,   # Stainless steel (slower)
            'SUS316': 3.5,   # Stainless steel (slower)
            'AL6061': 1.5,   # Aluminum (faster)
            'AL5052': 1.5    # Aluminum (faster)
        }

        # Safety considerations by material
        self.material_safety = {
            'SS400': SafetyLevel.MEDIUM,
            'SUS304': SafetyLevel.HIGH,    # Harder material
            'SUS316': SafetyLevel.HIGH,
            'AL6061': SafetyLevel.LOW,     # Easier to cut
            'AL5052': SafetyLevel.LOW
        }

    def generate_work_instruction(
        self,
        placement_result: PlacementResult,
        sheet: SteelSheet,
        include_safety_notes: bool = True,
        optimize_sequence: bool = True
    ) -> WorkInstruction:
        """
        Generate complete work instruction from placement result
        配置結果から完全な作業指示を生成
        """
        start_time = time.time()

        self.logger.info(f"Generating work instruction for sheet {placement_result.sheet_id}")

        # Generate cutting sequence
        cutting_sequence = self._generate_cutting_sequence(
            placement_result.panels,
            sheet,
            optimize_sequence
        )

        # Generate quality checkpoints
        quality_checkpoints = self._generate_quality_checkpoints(cutting_sequence)

        # Generate safety notes
        safety_notes = self._generate_safety_notes(sheet.material, cutting_sequence) if include_safety_notes else []

        # Generate machine settings
        machine_settings = self._generate_machine_settings(sheet.material, sheet.thickness)

        # Calculate estimated time
        estimated_time = self._calculate_total_time(cutting_sequence, sheet.material)

        # Create work instruction
        work_instruction = WorkInstruction(
            sheet_id=placement_result.sheet_id,
            material_type=sheet.material,
            sheet_dimensions=(sheet.width, sheet.height),
            total_steps=len(cutting_sequence),
            cutting_sequence=cutting_sequence,
            quality_checkpoints=quality_checkpoints,
            safety_notes=safety_notes,
            machine_settings=machine_settings,
            estimated_total_time=estimated_time,
            kerf_width=self.kerf_width,
            generated_at=datetime.now()
        )

        generation_time = time.time() - start_time
        self.logger.info(
            f"Work instruction generated: {len(cutting_sequence)} steps, "
            f"estimated time: {estimated_time:.1f}min, "
            f"generation time: {generation_time:.3f}s"
        )

        return work_instruction

    def _generate_cutting_sequence(
        self,
        placed_panels: List[PlacedPanel],
        sheet: SteelSheet,
        optimize: bool = True
    ) -> List[CuttingInstruction]:
        """
        Generate optimized cutting sequence with guillotine constraints
        ギロチン制約下での最適化された切断順序を生成
        """
        if not placed_panels:
            return []

        # Collect all cutting lines
        cutting_lines = self._collect_cutting_lines(placed_panels, sheet)

        # Generate cutting instructions
        instructions = self._create_cutting_instructions(cutting_lines, sheet)

        # Optimize sequence if requested
        if optimize:
            instructions = self._optimize_cutting_order(instructions, sheet)

        # Add step numbers and descriptions
        for i, instruction in enumerate(instructions, 1):
            instruction.step_number = i
            instruction.description = self._generate_step_description(instruction, i)

        return instructions

    def _collect_cutting_lines(
        self,
        placed_panels: List[PlacedPanel],
        sheet: SteelSheet
    ) -> Dict[str, List[float]]:
        """
        Collect all required cutting lines for guillotine cutting
        ギロチン切断に必要なすべての切断線を収集
        """
        horizontal_lines = set()
        vertical_lines = set()

        # Add sheet boundaries
        horizontal_lines.update([0, sheet.height])
        vertical_lines.update([0, sheet.width])

        # Add panel boundaries
        for panel in placed_panels:
            x1, y1, x2, y2 = panel.bounds
            horizontal_lines.update([y1, y2])
            vertical_lines.update([x1, x2])

        # Remove duplicate lines and sort
        return {
            'horizontal': sorted(list(horizontal_lines)),
            'vertical': sorted(list(vertical_lines))
        }

    def _create_cutting_instructions(
        self,
        cutting_lines: Dict[str, List[float]],
        sheet: SteelSheet
    ) -> List[CuttingInstruction]:
        """
        Create cutting instructions from cutting lines
        切断線から切断指示を作成
        """
        instructions = []

        # Generate horizontal cuts (top to bottom)
        for i, y in enumerate(cutting_lines['horizontal'][1:-1], 1):  # Skip boundaries
            instruction = CuttingInstruction(
                step_number=0,  # Will be set later
                cut_type=CutType.HORIZONTAL,
                start_point=(0, y),
                end_point=(sheet.width, y),
                dimension=sheet.width,
                description="",  # Will be set later
                safety_level=self._determine_safety_level(sheet.material, sheet.width),
                estimated_time=self._estimate_cut_time(sheet.width, sheet.material),
                remaining_pieces=[],
                target_panels=[]
            )
            instructions.append(instruction)

        # Generate vertical cuts (left to right)
        for i, x in enumerate(cutting_lines['vertical'][1:-1], 1):  # Skip boundaries
            instruction = CuttingInstruction(
                step_number=0,  # Will be set later
                cut_type=CutType.VERTICAL,
                start_point=(x, 0),
                end_point=(x, sheet.height),
                dimension=sheet.height,
                description="",  # Will be set later
                safety_level=self._determine_safety_level(sheet.material, sheet.height),
                estimated_time=self._estimate_cut_time(sheet.height, sheet.material),
                remaining_pieces=[],
                target_panels=[]
            )
            instructions.append(instruction)

        return instructions

    def _optimize_cutting_order(
        self,
        instructions: List[CuttingInstruction],
        sheet: SteelSheet
    ) -> List[CuttingInstruction]:
        """
        Optimize cutting order for efficiency and safety
        効率と安全性のための切断順序最適化
        """
        # Guillotine cutting strategy: horizontal cuts first, then vertical
        # This follows the Japanese manufacturing preference: top-to-bottom processing

        horizontal_cuts = [inst for inst in instructions if inst.cut_type == CutType.HORIZONTAL]
        vertical_cuts = [inst for inst in instructions if inst.cut_type == CutType.VERTICAL]

        # Sort horizontal cuts from top to bottom (largest y first)
        horizontal_cuts.sort(key=lambda cut: cut.start_point[1], reverse=True)

        # Sort vertical cuts from left to right (smallest x first)
        vertical_cuts.sort(key=lambda cut: cut.start_point[0])

        # Combine: all horizontal cuts first, then vertical cuts
        optimized = horizontal_cuts + vertical_cuts

        self.logger.debug(
            f"Optimized cutting order: {len(horizontal_cuts)} horizontal + "
            f"{len(vertical_cuts)} vertical cuts"
        )

        return optimized

    def _determine_safety_level(self, material: str, cut_length: float) -> SafetyLevel:
        """
        Determine safety level based on material and cut length
        材質と切断長に基づく安全レベル決定
        """
        base_safety = self.material_safety.get(material, SafetyLevel.MEDIUM)

        # Increase safety level for long cuts
        if cut_length > 2000:  # > 2 meters
            if base_safety == SafetyLevel.LOW:
                return SafetyLevel.MEDIUM
            elif base_safety == SafetyLevel.MEDIUM:
                return SafetyLevel.HIGH
            else:
                return SafetyLevel.CRITICAL

        return base_safety

    def _estimate_cut_time(self, cut_length: float, material: str) -> float:
        """
        Estimate cutting time in minutes
        切断時間を分単位で見積もり
        """
        speed = self.cutting_speed.get(material, 2.0)  # minutes per meter
        length_meters = cut_length / 1000  # Convert mm to meters

        # Base cutting time
        cutting_time = length_meters * speed

        # Add setup time (positioning, measurement)
        setup_time = 0.5  # 30 seconds

        return cutting_time + setup_time

    def _generate_step_description(
        self,
        instruction: CuttingInstruction,
        step_number: int
    ) -> str:
        """
        Generate human-readable step description
        人間が読める作業ステップ説明を生成
        """
        cut_type_jp = "横切断" if instruction.cut_type == CutType.HORIZONTAL else "縦切断"
        cut_type_en = "Horizontal cut" if instruction.cut_type == CutType.HORIZONTAL else "Vertical cut"

        x1, y1 = instruction.start_point
        x2, y2 = instruction.end_point

        description = (
            f"Step {step_number}: {cut_type_jp} / {cut_type_en}\n"
            f"開始点 / Start: ({x1:.1f}, {y1:.1f})mm\n"
            f"終了点 / End: ({x2:.1f}, {y2:.1f})mm\n"
            f"長さ / Length: {instruction.dimension:.1f}mm\n"
            f"予想時間 / Est. time: {instruction.estimated_time:.1f}min"
        )

        return description

    def _generate_quality_checkpoints(
        self,
        cutting_sequence: List[CuttingInstruction]
    ) -> List[str]:
        """
        Generate quality checkpoints for the cutting process
        切断プロセスの品質チェックポイントを生成
        """
        checkpoints = [
            "材料確認 / Material verification - Confirm material type and thickness",
            "機械設定確認 / Machine setup verification - Check cutting parameters",
            "初期寸法測定 / Initial dimension measurement - Verify sheet dimensions"
        ]

        # Add checkpoint every 5 steps for long sequences
        if len(cutting_sequence) > 10:
            for i in range(5, len(cutting_sequence), 5):
                checkpoints.append(
                    f"中間検査 {i} / Intermediate inspection {i} - "
                    f"Verify cut accuracy and remaining material"
                )

        checkpoints.extend([
            "最終寸法確認 / Final dimension verification - Check all panel dimensions",
            "エッジ品質検査 / Edge quality inspection - Verify cut edge quality",
            "数量確認 / Quantity verification - Confirm all panels are cut"
        ])

        return checkpoints

    def _generate_safety_notes(
        self,
        material: str,
        cutting_sequence: List[CuttingInstruction]
    ) -> List[str]:
        """
        Generate safety notes specific to material and cutting process
        材質と切断プロセスに固有の安全注意事項を生成
        """
        safety_notes = [
            "保護具着用必須 / PPE required - Safety glasses, gloves, and hearing protection",
            "切断エリア確認 / Cutting area verification - Ensure clear workspace",
            "緊急停止位置確認 / Emergency stop location - Know emergency stop procedures"
        ]

        # Material-specific safety notes
        material_notes = {
            'SUS304': [
                "ステンレス鋼注意 / Stainless steel caution - Higher cutting force required",
                "冷却液使用 / Coolant use - Use appropriate coolant for heat management"
            ],
            'SUS316': [
                "高合金鋼注意 / High alloy steel caution - Slower cutting speeds required",
                "工具寿命注意 / Tool life attention - Monitor cutting tool condition"
            ],
            'AL6061': [
                "アルミ切粉注意 / Aluminum chip caution - Manage aluminum chips properly",
                "潤滑剤使用 / Lubricant use - Use appropriate cutting fluid"
            ]
        }

        if material in material_notes:
            safety_notes.extend(material_notes[material])

        # Add notes for complex cutting sequences
        if len(cutting_sequence) > 15:
            safety_notes.append(
                "複雑作業注意 / Complex operation attention - "
                "Take breaks every 30 minutes to maintain focus"
            )

        # Check for long cuts requiring special attention
        long_cuts = [inst for inst in cutting_sequence if inst.dimension > 2000]
        if long_cuts:
            safety_notes.append(
                "長尺切断注意 / Long cut attention - "
                "Use proper support for long cuts to prevent material sagging"
            )

        return safety_notes

    def _generate_machine_settings(
        self,
        material: str,
        thickness: float
    ) -> Dict[str, Any]:
        """
        Generate machine settings based on material and thickness
        材質と板厚に基づく機械設定を生成
        """
        # Base settings for different materials
        base_settings = {
            'SS400': {
                'cutting_speed': 1500,  # mm/min
                'feed_rate': 800,       # mm/min
                'power': 85,            # % of max power
                'gas_pressure': 1.2     # bar
            },
            'SUS304': {
                'cutting_speed': 1200,
                'feed_rate': 600,
                'power': 95,
                'gas_pressure': 1.5
            },
            'SUS316': {
                'cutting_speed': 1000,
                'feed_rate': 500,
                'power': 100,
                'gas_pressure': 1.8
            },
            'AL6061': {
                'cutting_speed': 2000,
                'feed_rate': 1000,
                'power': 70,
                'gas_pressure': 0.8
            },
            'AL5052': {
                'cutting_speed': 1800,
                'feed_rate': 900,
                'power': 75,
                'gas_pressure': 0.9
            }
        }

        settings = base_settings.get(material, base_settings['SS400']).copy()

        # Adjust for thickness
        thickness_factor = thickness / 6.0  # 6mm is baseline
        settings['cutting_speed'] = int(settings['cutting_speed'] / thickness_factor)
        settings['power'] = min(100, int(settings['power'] * thickness_factor))

        # Add material-specific settings
        settings.update({
            'material': material,
            'thickness': thickness,
            'kerf_width': self.kerf_width,
            'coolant_required': material.startswith('SUS'),
            'tool_type': 'carbide' if material.startswith('SUS') else 'standard'
        })

        return settings

    def _calculate_total_time(
        self,
        cutting_sequence: List[CuttingInstruction],
        material: str
    ) -> float:
        """
        Calculate total estimated time for cutting sequence
        切断順序の総見積もり時間を計算
        """
        if not cutting_sequence:
            return 0.0

        # Sum individual cutting times
        cutting_time = sum(inst.estimated_time for inst in cutting_sequence)

        # Add setup and preparation time
        setup_time = 5.0  # 5 minutes initial setup

        # Add material handling time (based on material type)
        handling_multiplier = {
            'SS400': 1.0,
            'SUS304': 1.2,  # Heavier, more careful handling
            'SUS316': 1.3,
            'AL6061': 0.8,  # Lighter
            'AL5052': 0.8
        }

        handling_factor = handling_multiplier.get(material, 1.0)
        handling_time = len(cutting_sequence) * 0.5 * handling_factor  # 30s per cut adjusted

        total_time = setup_time + cutting_time + handling_time

        self.logger.debug(
            f"Time calculation: setup={setup_time:.1f}min, "
            f"cutting={cutting_time:.1f}min, handling={handling_time:.1f}min, "
            f"total={total_time:.1f}min"
        )

        return total_time


def create_work_instruction_generator(kerf_width: float = 3.5) -> WorkInstructionGenerator:
    """Create work instruction generator instance"""
    return WorkInstructionGenerator(kerf_width)