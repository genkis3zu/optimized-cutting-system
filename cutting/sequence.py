"""
Cutting Sequence Optimization for Steel Processing
鋼板加工用切断順序最適化

Optimizes cutting sequence under guillotine constraints with Japanese manufacturing preferences
ギロチン制約下での日本製造業の嗜好に基づく切断順序最適化
"""

import time
import logging
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import math

from cutting.instruction import CuttingInstruction, CutType, SafetyLevel, WorkInstruction
from core.models import PlacedPanel, SteelSheet


class SequenceStrategy(Enum):
    """Sequence optimization strategy"""
    TOP_TO_BOTTOM = "top_to_bottom"      # Japanese standard: 上から下へ
    MINIMIZE_TOOL_CHANGES = "min_tools"   # Minimize tool/setting changes
    MINIMIZE_HANDLING = "min_handling"    # Minimize material handling
    SAFETY_FIRST = "safety_first"        # Prioritize safety over efficiency
    HYBRID = "hybrid"                     # Balanced approach


@dataclass
class SequenceConstraint:
    """Constraints for sequence optimization"""
    max_handling_weight: float = 50.0    # kg - Maximum safe handling weight
    min_stability_size: float = 100.0    # mm - Minimum size for stable handling
    tool_change_penalty: float = 2.0     # minutes - Time penalty for tool changes
    material_flip_penalty: float = 5.0   # minutes - Time penalty for flipping material
    safety_buffer: float = 50.0          # mm - Safety buffer around cuts


@dataclass
class CuttingOperation:
    """
    Enhanced cutting operation with optimization metadata
    最適化メタデータ付き拡張切断操作
    """
    instruction: CuttingInstruction
    dependency_cuts: List[int] = None     # Cut IDs that must be completed first
    creates_instability: bool = False     # Whether this cut creates unstable pieces
    requires_support: bool = False        # Whether cut requires additional support
    tool_requirements: Dict[str, Any] = None  # Tool/setting requirements
    accessibility_score: float = 1.0     # How accessible the cut is (0-1)
    stability_impact: float = 0.0        # Impact on remaining material stability

    def __post_init__(self):
        if self.dependency_cuts is None:
            self.dependency_cuts = []
        if self.tool_requirements is None:
            self.tool_requirements = {}


class CuttingSequenceOptimizer:
    """
    Advanced cutting sequence optimizer with Japanese manufacturing preferences
    日本製造業の嗜好に基づく高度な切断順序最適化器
    """

    def __init__(self, strategy: SequenceStrategy = SequenceStrategy.HYBRID):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

        # Japanese manufacturing preferences
        self.jp_preferences = {
            'prefer_horizontal_first': True,     # 横切りを先に
            'top_to_bottom_order': True,         # 上から下への順序
            'minimize_material_handling': True,  # 材料取り扱い最小化
            'safety_margin_factor': 1.2,        # 安全マージン係数
            'quality_over_speed': True           # 品質重視
        }

        # Optimization weights for different criteria
        self.optimization_weights = {
            'time_efficiency': 0.3,
            'safety': 0.3,
            'quality': 0.2,
            'material_handling': 0.2
        }

    def optimize_sequence(
        self,
        work_instruction: WorkInstruction,
        constraints: Optional[SequenceConstraint] = None
    ) -> WorkInstruction:
        """
        Optimize cutting sequence with advanced considerations
        高度な考慮事項による切断順序最適化
        """
        start_time = time.time()

        if constraints is None:
            constraints = SequenceConstraint()

        self.logger.info(
            f"Optimizing cutting sequence with {len(work_instruction.cutting_sequence)} operations "
            f"using {self.strategy.value} strategy"
        )

        # Convert to cutting operations with metadata
        cutting_ops = self._create_cutting_operations(work_instruction, constraints)

        # Apply strategy-specific optimization
        if self.strategy == SequenceStrategy.TOP_TO_BOTTOM:
            optimized_ops = self._optimize_top_to_bottom(cutting_ops)
        elif self.strategy == SequenceStrategy.MINIMIZE_TOOL_CHANGES:
            optimized_ops = self._optimize_tool_changes(cutting_ops)
        elif self.strategy == SequenceStrategy.MINIMIZE_HANDLING:
            optimized_ops = self._optimize_material_handling(cutting_ops)
        elif self.strategy == SequenceStrategy.SAFETY_FIRST:
            optimized_ops = self._optimize_safety_first(cutting_ops)
        else:  # HYBRID
            optimized_ops = self._optimize_hybrid(cutting_ops, constraints)

        # Validate sequence and resolve dependencies
        optimized_ops = self._resolve_dependencies(optimized_ops)

        # Update work instruction with optimized sequence
        optimized_instruction = self._update_work_instruction(work_instruction, optimized_ops)

        optimization_time = time.time() - start_time
        self.logger.info(
            f"Sequence optimization completed in {optimization_time:.3f}s, "
            f"estimated time change: {optimized_instruction.estimated_total_time - work_instruction.estimated_total_time:.1f}min"
        )

        return optimized_instruction

    def _create_cutting_operations(
        self,
        work_instruction: WorkInstruction,
        constraints: SequenceConstraint
    ) -> List[CuttingOperation]:
        """
        Convert cutting instructions to enhanced operations with metadata
        切断指示を拡張操作（メタデータ付き）に変換
        """
        operations = []

        for i, instruction in enumerate(work_instruction.cutting_sequence):
            operation = CuttingOperation(
                instruction=instruction,
                tool_requirements=self._determine_tool_requirements(instruction, work_instruction.material_type),
                accessibility_score=self._calculate_accessibility_score(instruction, work_instruction),
                stability_impact=self._calculate_stability_impact(instruction, work_instruction)
            )

            # Analyze cut dependencies and characteristics
            operation.creates_instability = self._check_creates_instability(instruction, constraints)
            operation.requires_support = self._check_requires_support(instruction, constraints)
            operation.dependency_cuts = self._find_dependency_cuts(instruction, work_instruction.cutting_sequence, i)

            operations.append(operation)

        return operations

    def _determine_tool_requirements(
        self,
        instruction: CuttingInstruction,
        material: str
    ) -> Dict[str, Any]:
        """
        Determine tool requirements for cutting operation
        切断操作のツール要件を決定
        """
        requirements = {
            'tool_type': 'standard',
            'cutting_speed': 'normal',
            'coolant_required': False,
            'special_handling': False
        }

        # Material-specific requirements
        if material.startswith('SUS'):  # Stainless steel
            requirements.update({
                'tool_type': 'carbide',
                'cutting_speed': 'slow',
                'coolant_required': True,
                'special_handling': True
            })
        elif material.startswith('AL'):  # Aluminum
            requirements.update({
                'tool_type': 'aluminum',
                'cutting_speed': 'fast',
                'coolant_required': False
            })

        # Cut-specific requirements
        if instruction.dimension > 2000:  # Long cuts
            requirements['special_handling'] = True
            requirements['support_required'] = True

        if instruction.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
            requirements['special_handling'] = True

        return requirements

    def _calculate_accessibility_score(
        self,
        instruction: CuttingInstruction,
        work_instruction: WorkInstruction
    ) -> float:
        """
        Calculate how accessible the cut is (0-1, higher is better)
        切断の到達性を計算（0-1、高い方が良い）
        """
        x1, y1 = instruction.start_point
        x2, y2 = instruction.end_point
        sheet_w, sheet_h = work_instruction.sheet_dimensions

        # Base score - cuts near edges are more accessible
        edge_distance = min(x1, y1, sheet_w - x2, sheet_h - y2)
        edge_score = max(0, min(1, edge_distance / 200))  # Within 200mm of edge gets full score

        # Reduce score for cuts that require reaching over large pieces
        reach_penalty = 0.0
        if instruction.cut_type == CutType.HORIZONTAL and y1 > sheet_h * 0.5:
            reach_penalty = 0.2  # Cutting at top requires reach-over
        elif instruction.cut_type == CutType.VERTICAL and x1 > sheet_w * 0.5:
            reach_penalty = 0.2  # Cutting at right requires reach-over

        return max(0, 1.0 - edge_score - reach_penalty)

    def _calculate_stability_impact(
        self,
        instruction: CuttingInstruction,
        work_instruction: WorkInstruction
    ) -> float:
        """
        Calculate impact on material stability (0-1, higher impact is worse)
        材料安定性への影響を計算（0-1、影響大は悪い）
        """
        # Long cuts have higher stability impact
        length_factor = min(1.0, instruction.dimension / 3000)

        # Cuts that divide material into smaller pieces increase instability
        position_factor = 0.0
        sheet_w, sheet_h = work_instruction.sheet_dimensions

        if instruction.cut_type == CutType.HORIZONTAL:
            # Horizontal cuts in the middle create most instability
            y_position = instruction.start_point[1]
            center_distance = abs(y_position - sheet_h / 2) / (sheet_h / 2)
            position_factor = 1.0 - center_distance
        else:  # Vertical cut
            x_position = instruction.start_point[0]
            center_distance = abs(x_position - sheet_w / 2) / (sheet_w / 2)
            position_factor = 1.0 - center_distance

        return (length_factor * 0.6 + position_factor * 0.4)

    def _check_creates_instability(
        self,
        instruction: CuttingInstruction,
        constraints: SequenceConstraint
    ) -> bool:
        """
        Check if cut creates unstable pieces
        切断が不安定な部品を作るかチェック
        """
        # Very long cuts can create instability
        if instruction.dimension > 2500:
            return True

        # Cuts that create narrow strips
        if instruction.cut_type == CutType.VERTICAL and instruction.dimension > 1000:
            # Check if this creates a narrow vertical strip
            return True

        return False

    def _check_requires_support(
        self,
        instruction: CuttingInstruction,
        constraints: SequenceConstraint
    ) -> bool:
        """
        Check if cut requires additional support
        切断が追加サポートを必要とするかチェック
        """
        # Long cuts require support
        if instruction.dimension > 1500:
            return True

        # High safety level cuts may require support
        if instruction.safety_level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
            return True

        return False

    def _find_dependency_cuts(
        self,
        instruction: CuttingInstruction,
        all_instructions: List[CuttingInstruction],
        current_index: int
    ) -> List[int]:
        """
        Find cuts that must be completed before this cut
        この切断の前に完了すべき切断を特定
        """
        dependencies = []

        # In guillotine cutting, some cuts enable others
        # For example, horizontal cuts must be done before related vertical cuts

        x1, y1 = instruction.start_point
        x2, y2 = instruction.end_point

        for i, other_instruction in enumerate(all_instructions):
            if i >= current_index:  # Only consider previous cuts
                continue

            other_x1, other_y1 = other_instruction.start_point
            other_x2, other_y2 = other_instruction.end_point

            # Check if this cut intersects with or depends on the other cut
            if self._cuts_intersect_or_depend(instruction, other_instruction):
                dependencies.append(i)

        return dependencies

    def _cuts_intersect_or_depend(
        self,
        cut1: CuttingInstruction,
        cut2: CuttingInstruction
    ) -> bool:
        """
        Check if two cuts intersect or have dependencies
        2つの切断が交差または依存関係にあるかチェック
        """
        x1_1, y1_1 = cut1.start_point
        x2_1, y2_1 = cut1.end_point
        x1_2, y1_2 = cut2.start_point
        x2_2, y2_2 = cut2.end_point

        # Check for intersection
        if cut1.cut_type != cut2.cut_type:
            # Horizontal and vertical cuts - check intersection
            if cut1.cut_type == CutType.HORIZONTAL:
                # cut1 is horizontal, cut2 is vertical
                if (min(x1_1, x2_1) <= x1_2 <= max(x1_1, x2_1) and
                    min(y1_2, y2_2) <= y1_1 <= max(y1_2, y2_2)):
                    return True
            else:
                # cut1 is vertical, cut2 is horizontal
                if (min(y1_1, y2_1) <= y1_2 <= max(y1_1, y2_1) and
                    min(x1_2, x2_2) <= x1_1 <= max(x1_2, x2_2)):
                    return True

        return False

    def _optimize_top_to_bottom(self, operations: List[CuttingOperation]) -> List[CuttingOperation]:
        """
        Optimize using top-to-bottom Japanese manufacturing preference
        日本製造業の上から下への順序嗜好を使用した最適化
        """
        horizontal_ops = [op for op in operations if op.instruction.cut_type == CutType.HORIZONTAL]
        vertical_ops = [op for op in operations if op.instruction.cut_type == CutType.VERTICAL]

        # Sort horizontal cuts from top to bottom (highest Y first)
        horizontal_ops.sort(key=lambda op: op.instruction.start_point[1], reverse=True)

        # Sort vertical cuts from left to right (lowest X first)
        vertical_ops.sort(key=lambda op: op.instruction.start_point[0])

        # Japanese preference: horizontal cuts first, then vertical
        return horizontal_ops + vertical_ops

    def _optimize_tool_changes(self, operations: List[CuttingOperation]) -> List[CuttingOperation]:
        """
        Optimize to minimize tool changes
        ツール変更を最小化する最適化
        """
        # Group by tool requirements
        tool_groups = {}
        for op in operations:
            tool_type = op.tool_requirements.get('tool_type', 'standard')
            speed = op.tool_requirements.get('cutting_speed', 'normal')
            key = f"{tool_type}_{speed}"

            if key not in tool_groups:
                tool_groups[key] = []
            tool_groups[key].append(op)

        # Sort each group by position for efficient cutting
        optimized = []
        for group in tool_groups.values():
            # Sort by position within each tool group
            group.sort(key=lambda op: (
                op.instruction.start_point[1] if op.instruction.cut_type == CutType.HORIZONTAL
                else op.instruction.start_point[0]
            ))
            optimized.extend(group)

        return optimized

    def _optimize_material_handling(self, operations: List[CuttingOperation]) -> List[CuttingOperation]:
        """
        Optimize to minimize material handling
        材料取り扱いを最小化する最適化
        """
        # Prioritize cuts that don't require material repositioning
        stable_cuts = [op for op in operations if not op.requires_support and not op.creates_instability]
        unstable_cuts = [op for op in operations if op.requires_support or op.creates_instability]

        # Sort stable cuts by accessibility
        stable_cuts.sort(key=lambda op: op.accessibility_score, reverse=True)

        # Sort unstable cuts by safety level and then position
        unstable_cuts.sort(key=lambda op: (
            op.instruction.safety_level.value,
            op.stability_impact
        ))

        return stable_cuts + unstable_cuts

    def _optimize_safety_first(self, operations: List[CuttingOperation]) -> List[CuttingOperation]:
        """
        Optimize prioritizing safety over efficiency
        効率より安全性を優先した最適化
        """
        # Group by safety level
        safety_groups = {level: [] for level in SafetyLevel}

        for op in operations:
            safety_groups[op.instruction.safety_level].append(op)

        # Process in order of increasing safety risk
        safety_order = [SafetyLevel.LOW, SafetyLevel.MEDIUM, SafetyLevel.HIGH, SafetyLevel.CRITICAL]
        optimized = []

        for level in safety_order:
            group = safety_groups[level]
            # Within each safety level, sort by stability impact
            group.sort(key=lambda op: op.stability_impact)
            optimized.extend(group)

        return optimized

    def _optimize_hybrid(
        self,
        operations: List[CuttingOperation],
        constraints: SequenceConstraint
    ) -> List[CuttingOperation]:
        """
        Hybrid optimization balancing multiple criteria
        複数基準をバランスするハイブリッド最適化
        """
        # Calculate composite score for each operation
        scored_operations = []

        for op in operations:
            # Time efficiency score (lower time is better)
            time_score = 1.0 - min(1.0, op.instruction.estimated_time / 10.0)

            # Safety score (lower safety risk is better)
            safety_values = {SafetyLevel.LOW: 1.0, SafetyLevel.MEDIUM: 0.7,
                           SafetyLevel.HIGH: 0.4, SafetyLevel.CRITICAL: 0.1}
            safety_score = safety_values[op.instruction.safety_level]

            # Quality score (higher accessibility and lower stability impact is better)
            quality_score = op.accessibility_score * (1.0 - op.stability_impact)

            # Material handling score (less handling is better)
            handling_score = 1.0 if not (op.requires_support or op.creates_instability) else 0.5

            # Weighted composite score
            composite_score = (
                time_score * self.optimization_weights['time_efficiency'] +
                safety_score * self.optimization_weights['safety'] +
                quality_score * self.optimization_weights['quality'] +
                handling_score * self.optimization_weights['material_handling']
            )

            scored_operations.append((composite_score, op))

        # Sort by composite score (highest first)
        scored_operations.sort(key=lambda x: x[0], reverse=True)

        # Apply Japanese preference for horizontal-first within similar scores
        if self.jp_preferences['prefer_horizontal_first']:
            scored_operations = self._apply_horizontal_preference(scored_operations)

        return [op for score, op in scored_operations]

    def _apply_horizontal_preference(
        self,
        scored_operations: List[Tuple[float, CuttingOperation]]
    ) -> List[Tuple[float, CuttingOperation]]:
        """
        Apply Japanese preference for horizontal cuts first
        横切りを先にする日本の嗜好を適用
        """
        # Group operations with similar scores (within 0.1)
        score_groups = []
        current_group = []
        last_score = None

        for score, op in scored_operations:
            if last_score is None or abs(score - last_score) <= 0.1:
                current_group.append((score, op))
            else:
                if current_group:
                    score_groups.append(current_group)
                current_group = [(score, op)]
            last_score = score

        if current_group:
            score_groups.append(current_group)

        # Within each group, prioritize horizontal cuts
        reordered = []
        for group in score_groups:
            horizontal = [(s, op) for s, op in group if op.instruction.cut_type == CutType.HORIZONTAL]
            vertical = [(s, op) for s, op in group if op.instruction.cut_type == CutType.VERTICAL]

            # Sort horizontal cuts top to bottom
            horizontal.sort(key=lambda x: x[1].instruction.start_point[1], reverse=True)
            # Sort vertical cuts left to right
            vertical.sort(key=lambda x: x[1].instruction.start_point[0])

            reordered.extend(horizontal + vertical)

        return reordered

    def _resolve_dependencies(self, operations: List[CuttingOperation]) -> List[CuttingOperation]:
        """
        Resolve cutting dependencies and ensure valid sequence
        切断依存関係を解決し有効な順序を保証
        """
        # Create dependency graph
        resolved = []
        remaining = operations.copy()
        max_iterations = len(operations) * 2  # Prevent infinite loops

        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1
            progress_made = False

            for i, op in enumerate(remaining):
                # Check if all dependencies are satisfied
                dependencies_satisfied = all(
                    any(resolved_op.instruction.step_number == dep_id for resolved_op in resolved)
                    for dep_id in op.dependency_cuts
                )

                if dependencies_satisfied:
                    resolved.append(op)
                    remaining.pop(i)
                    progress_made = True
                    break

            if not progress_made and remaining:
                # Force resolution of circular dependencies
                self.logger.warning("Circular dependency detected, forcing resolution")
                remaining[0].dependency_cuts = []

        if remaining:
            self.logger.warning(f"Could not resolve all dependencies, {len(remaining)} operations remain")
            resolved.extend(remaining)

        return resolved

    def _update_work_instruction(
        self,
        original: WorkInstruction,
        optimized_ops: List[CuttingOperation]
    ) -> WorkInstruction:
        """
        Update work instruction with optimized sequence
        最適化された順序で作業指示を更新
        """
        # Extract optimized cutting sequence
        optimized_sequence = []
        total_time = 0

        for i, op in enumerate(optimized_ops, 1):
            instruction = op.instruction
            instruction.step_number = i
            instruction.description = self._update_step_description(instruction, i, op)
            optimized_sequence.append(instruction)
            total_time += instruction.estimated_time

        # Create new work instruction with optimized sequence
        return WorkInstruction(
            sheet_id=original.sheet_id,
            material_type=original.material_type,
            sheet_dimensions=original.sheet_dimensions,
            total_steps=len(optimized_sequence),
            cutting_sequence=optimized_sequence,
            quality_checkpoints=original.quality_checkpoints,
            safety_notes=self._update_safety_notes(original.safety_notes, optimized_ops),
            machine_settings=original.machine_settings,
            estimated_total_time=total_time + 5.0,  # Add setup time
            kerf_width=original.kerf_width,
            generated_at=original.generated_at,
            generated_by=f"{original.generated_by} (Optimized)"
        )

    def _update_step_description(
        self,
        instruction: CuttingInstruction,
        step_number: int,
        operation: CuttingOperation
    ) -> str:
        """
        Update step description with optimization information
        最適化情報で作業ステップ説明を更新
        """
        base_description = instruction.description

        # Add optimization notes
        optimization_notes = []

        if operation.requires_support:
            optimization_notes.append("要サポート / Requires support")

        if operation.creates_instability:
            optimization_notes.append("安定性注意 / Stability caution")

        if operation.tool_requirements.get('special_handling'):
            optimization_notes.append("特別取扱 / Special handling required")

        if optimization_notes:
            base_description += f"\n注意事項 / Notes: {', '.join(optimization_notes)}"

        return base_description

    def _update_safety_notes(
        self,
        original_notes: List[str],
        optimized_ops: List[CuttingOperation]
    ) -> List[str]:
        """
        Update safety notes based on optimized sequence
        最適化された順序に基づく安全注意事項の更新
        """
        updated_notes = original_notes.copy()

        # Add sequence-specific safety notes
        unstable_cuts = [op for op in optimized_ops if op.creates_instability]
        if unstable_cuts:
            updated_notes.append(
                f"不安定化切断注意 / Instability warning - "
                f"{len(unstable_cuts)} cuts may create unstable pieces"
            )

        support_cuts = [op for op in optimized_ops if op.requires_support]
        if support_cuts:
            updated_notes.append(
                f"サポート必要 / Support required - "
                f"{len(support_cuts)} cuts require additional support"
            )

        return updated_notes


def create_sequence_optimizer(strategy: SequenceStrategy = SequenceStrategy.HYBRID) -> CuttingSequenceOptimizer:
    """Create cutting sequence optimizer instance"""
    return CuttingSequenceOptimizer(strategy)