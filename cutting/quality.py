"""
Quality Management System for Steel Cutting Operations
鋼板切断作業用品質管理システム

Manages quality checkpoints, standards, and validation throughout the cutting process
切断プロセス全体の品質チェックポイント、標準、バリデーションを管理
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from cutting.instruction import WorkInstruction, CuttingInstruction


class CheckpointType(Enum):
    """Quality checkpoint types"""
    DIMENSIONAL = "dimensional"      # 寸法検査
    VISUAL = "visual"               # 外観検査
    MATERIAL = "material"           # 材質確認
    EDGE_QUALITY = "edge_quality"   # エッジ品質
    SAFETY = "safety"               # 安全確認
    SETUP = "setup"                 # セットアップ確認
    FINAL = "final"                 # 最終検査


class InspectionMethod(Enum):
    """Inspection methods"""
    VISUAL_CHECK = "visual"         # 目視確認
    CALIPER = "caliper"            # キャリパー測定
    RULER = "ruler"                # 定規測定
    GAUGE = "gauge"                # ゲージ測定
    TEMPLATE = "template"          # テンプレート照合
    AUTOMATED = "automated"        # 自動測定


class QualityLevel(Enum):
    """Quality requirement levels"""
    BASIC = "basic"                # 基本品質
    STANDARD = "standard"          # 標準品質
    PRECISION = "precision"        # 精密品質
    CRITICAL = "critical"          # 重要品質


@dataclass
class QualityCheckpoint:
    """
    Individual quality checkpoint
    個別品質チェックポイント
    """
    checkpoint_id: str
    step_number: int
    checkpoint_type: CheckpointType
    description: str
    japanese_description: str
    inspection_method: InspectionMethod
    target_value: Optional[float] = None       # Target measurement
    tolerance: Optional[float] = None          # Acceptable tolerance
    upper_limit: Optional[float] = None        # Upper specification limit
    lower_limit: Optional[float] = None        # Lower specification limit
    critical_flag: bool = False                # Critical to quality
    estimated_time: float = 1.0                # Minutes required
    required_tools: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    pass_criteria: str = ""
    fail_actions: List[str] = field(default_factory=list)

    @property
    def is_in_tolerance(self) -> bool:
        """Check if target value is within tolerance"""
        if self.target_value is None or self.tolerance is None:
            return True

        if self.upper_limit is not None and self.target_value > self.upper_limit:
            return False
        if self.lower_limit is not None and self.target_value < self.lower_limit:
            return False

        return True


@dataclass
class QualityRecord:
    """
    Quality inspection record
    品質検査記録
    """
    checkpoint_id: str
    inspector: str
    timestamp: datetime
    measured_value: Optional[float] = None
    pass_status: bool = True
    notes: str = ""
    corrective_actions: List[str] = field(default_factory=list)
    verification_required: bool = False


@dataclass
class QualityStandard:
    """
    Quality standards for materials and processes
    材質・プロセス用品質標準
    """
    material_type: str
    thickness_range: Tuple[float, float]
    dimensional_tolerance: float = 0.5         # ±mm
    edge_roughness_max: float = 3.2           # μm Ra
    perpendicularity_tolerance: float = 0.3   # mm
    parallelism_tolerance: float = 0.2        # mm
    burn_free_edge_required: bool = True
    deburring_required: bool = True
    surface_finish_requirements: Dict[str, Any] = field(default_factory=dict)


class QualityManager:
    """
    Quality management system for steel cutting operations
    鋼板切断作業用品質管理システム
    """

    def __init__(self, quality_level: QualityLevel = QualityLevel.STANDARD):
        self.quality_level = quality_level
        self.logger = logging.getLogger(__name__)

        # Quality standards database
        self.quality_standards = self._initialize_quality_standards()

        # Inspection templates
        self.inspection_templates = self._initialize_inspection_templates()

        # Quality records storage
        self.quality_records: List[QualityRecord] = []

    def generate_quality_plan(
        self,
        work_instruction: WorkInstruction
    ) -> List[QualityCheckpoint]:
        """
        Generate comprehensive quality plan for work instruction
        作業指示の包括的品質計画を生成
        """
        start_time = time.time()

        self.logger.info(
            f"Generating quality plan for {work_instruction.material_type} "
            f"with {work_instruction.total_steps} cutting steps"
        )

        checkpoints = []

        # Pre-cutting checkpoints
        checkpoints.extend(self._generate_pre_cutting_checkpoints(work_instruction))

        # Process checkpoints
        checkpoints.extend(self._generate_process_checkpoints(work_instruction))

        # Post-cutting checkpoints
        checkpoints.extend(self._generate_post_cutting_checkpoints(work_instruction))

        # Final inspection checkpoint
        checkpoints.extend(self._generate_final_checkpoints(work_instruction))

        generation_time = time.time() - start_time
        self.logger.info(
            f"Generated {len(checkpoints)} quality checkpoints in {generation_time:.3f}s"
        )

        return checkpoints

    def _generate_pre_cutting_checkpoints(
        self,
        work_instruction: WorkInstruction
    ) -> List[QualityCheckpoint]:
        """Generate pre-cutting quality checkpoints"""
        checkpoints = []

        # Material verification
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="PRE_001",
            step_number=0,
            checkpoint_type=CheckpointType.MATERIAL,
            description="Material type and thickness verification",
            japanese_description="材質・板厚確認",
            inspection_method=InspectionMethod.CALIPER,
            target_value=work_instruction.sheet_dimensions[0],
            tolerance=0.1,
            critical_flag=True,
            estimated_time=2.0,
            required_tools=["caliper", "material_certificate"],
            pass_criteria="Material matches specification and certificate",
            fail_actions=["Stop process", "Replace material", "Update specifications"]
        ))

        # Sheet dimension verification
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="PRE_002",
            step_number=0,
            checkpoint_type=CheckpointType.DIMENSIONAL,
            description="Sheet dimension verification",
            japanese_description="シート寸法確認",
            inspection_method=InspectionMethod.RULER,
            target_value=work_instruction.sheet_dimensions[0],
            tolerance=2.0,
            critical_flag=True,
            estimated_time=1.5,
            required_tools=["steel_ruler", "measuring_tape"],
            pass_criteria="Dimensions within tolerance",
            fail_actions=["Adjust cutting plan", "Mark actual dimensions"]
        ))

        # Machine setup verification
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="PRE_003",
            step_number=0,
            checkpoint_type=CheckpointType.SETUP,
            description="Machine setup and calibration verification",
            japanese_description="機械セットアップ・校正確認",
            inspection_method=InspectionMethod.VISUAL_CHECK,
            critical_flag=True,
            estimated_time=3.0,
            required_tools=["setup_checklist", "calibration_tools"],
            safety_notes=["Verify emergency stops", "Check cutting parameters"],
            pass_criteria="All machine parameters match work instruction",
            fail_actions=["Recalibrate machine", "Contact maintenance", "Update parameters"]
        ))

        return checkpoints

    def _generate_process_checkpoints(
        self,
        work_instruction: WorkInstruction
    ) -> List[QualityCheckpoint]:
        """Generate process quality checkpoints"""
        checkpoints = []

        # Determine checkpoint frequency based on cutting sequence length
        total_steps = len(work_instruction.cutting_sequence)

        if total_steps <= 5:
            checkpoint_interval = 2  # Every 2 steps for short sequences
        elif total_steps <= 15:
            checkpoint_interval = 5  # Every 5 steps for medium sequences
        else:
            checkpoint_interval = 10  # Every 10 steps for long sequences

        # Generate intermediate checkpoints
        for i in range(checkpoint_interval, total_steps, checkpoint_interval):
            checkpoints.append(QualityCheckpoint(
                checkpoint_id=f"PROC_{i:03d}",
                step_number=i,
                checkpoint_type=CheckpointType.DIMENSIONAL,
                description=f"Intermediate dimension check at step {i}",
                japanese_description=f"ステップ {i} 中間寸法チェック",
                inspection_method=InspectionMethod.CALIPER,
                tolerance=0.5,
                estimated_time=1.0,
                required_tools=["caliper"],
                pass_criteria="Cut dimensions within tolerance",
                fail_actions=["Adjust cutting parameters", "Re-cut if necessary"]
            ))

        # Edge quality checkpoints for critical materials
        if work_instruction.material_type.startswith('SUS'):
            checkpoints.append(QualityCheckpoint(
                checkpoint_id="PROC_EDGE",
                step_number=total_steps // 2,
                checkpoint_type=CheckpointType.EDGE_QUALITY,
                description="Edge quality inspection for stainless steel",
                japanese_description="ステンレス鋼エッジ品質検査",
                inspection_method=InspectionMethod.VISUAL_CHECK,
                estimated_time=2.0,
                required_tools=["magnifying_glass", "edge_gauge"],
                pass_criteria="No burn marks, smooth edge finish",
                fail_actions=["Adjust cutting speed", "Check coolant flow", "Deburr if necessary"]
            ))

        return checkpoints

    def _generate_post_cutting_checkpoints(
        self,
        work_instruction: WorkInstruction
    ) -> List[QualityCheckpoint]:
        """Generate post-cutting quality checkpoints"""
        checkpoints = []

        # Panel count verification
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="POST_001",
            step_number=work_instruction.total_steps + 1,
            checkpoint_type=CheckpointType.VISUAL,
            description="Panel count and identification verification",
            japanese_description="パネル数・識別確認",
            inspection_method=InspectionMethod.VISUAL_CHECK,
            critical_flag=True,
            estimated_time=2.0,
            required_tools=["panel_list", "marking_tools"],
            pass_criteria="All panels present and correctly identified",
            fail_actions=["Re-count panels", "Check for missing pieces", "Update records"]
        ))

        # Dimensional verification sampling
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="POST_002",
            step_number=work_instruction.total_steps + 2,
            checkpoint_type=CheckpointType.DIMENSIONAL,
            description="Random dimensional verification (20% sampling)",
            japanese_description="ランダム寸法確認（20%サンプリング）",
            inspection_method=InspectionMethod.CALIPER,
            tolerance=0.3,
            estimated_time=5.0,
            required_tools=["precision_caliper", "measurement_sheet"],
            pass_criteria="Sampled panels within tolerance",
            fail_actions=["Increase sample size", "Check cutting accuracy", "Adjust process"]
        ))

        return checkpoints

    def _generate_final_checkpoints(
        self,
        work_instruction: WorkInstruction
    ) -> List[QualityCheckpoint]:
        """Generate final inspection checkpoints"""
        checkpoints = []

        # Final inspection based on quality level
        if self.quality_level in [QualityLevel.PRECISION, QualityLevel.CRITICAL]:
            # Comprehensive final inspection
            checkpoints.append(QualityCheckpoint(
                checkpoint_id="FINAL_001",
                step_number=work_instruction.total_steps + 10,
                checkpoint_type=CheckpointType.FINAL,
                description="Comprehensive final inspection",
                japanese_description="包括的最終検査",
                inspection_method=InspectionMethod.TEMPLATE,
                critical_flag=True,
                estimated_time=10.0,
                required_tools=["go_no_go_gauges", "surface_roughness_tester", "inspection_checklist"],
                pass_criteria="All specifications met according to quality standard",
                fail_actions=["Rework non-conforming parts", "Document deviations", "Get approval"]
            ))
        else:
            # Standard final inspection
            checkpoints.append(QualityCheckpoint(
                checkpoint_id="FINAL_001",
                step_number=work_instruction.total_steps + 10,
                checkpoint_type=CheckpointType.FINAL,
                description="Standard final inspection",
                japanese_description="標準最終検査",
                inspection_method=InspectionMethod.VISUAL_CHECK,
                estimated_time=5.0,
                required_tools=["inspection_checklist"],
                pass_criteria="Visual inspection confirms quality requirements",
                fail_actions=["Document any defects", "Determine disposition"]
            ))

        # Documentation and release
        checkpoints.append(QualityCheckpoint(
            checkpoint_id="FINAL_002",
            step_number=work_instruction.total_steps + 11,
            checkpoint_type=CheckpointType.FINAL,
            description="Documentation completion and release authorization",
            japanese_description="文書化完了・リリース承認",
            inspection_method=InspectionMethod.VISUAL_CHECK,
            critical_flag=True,
            estimated_time=3.0,
            required_tools=["quality_records", "release_forms"],
            pass_criteria="All documentation complete and signed",
            fail_actions=["Complete missing documentation", "Get required approvals"]
        ))

        return checkpoints

    def _initialize_quality_standards(self) -> Dict[str, QualityStandard]:
        """Initialize quality standards for different materials"""
        standards = {}

        # Carbon Steel (SS400)
        standards['SS400'] = QualityStandard(
            material_type='SS400',
            thickness_range=(1.6, 25.0),
            dimensional_tolerance=0.5,
            edge_roughness_max=6.3,
            perpendicularity_tolerance=0.5,
            parallelism_tolerance=0.3,
            burn_free_edge_required=False,
            deburring_required=True
        )

        # Stainless Steel (SUS304)
        standards['SUS304'] = QualityStandard(
            material_type='SUS304',
            thickness_range=(0.8, 20.0),
            dimensional_tolerance=0.3,
            edge_roughness_max=3.2,
            perpendicularity_tolerance=0.3,
            parallelism_tolerance=0.2,
            burn_free_edge_required=True,
            deburring_required=True,
            surface_finish_requirements={
                'no_discoloration': True,
                'no_heat_marks': True,
                'passivation_required': True
            }
        )

        # Stainless Steel (SUS316)
        standards['SUS316'] = QualityStandard(
            material_type='SUS316',
            thickness_range=(0.8, 20.0),
            dimensional_tolerance=0.3,
            edge_roughness_max=1.6,
            perpendicularity_tolerance=0.2,
            parallelism_tolerance=0.15,
            burn_free_edge_required=True,
            deburring_required=True,
            surface_finish_requirements={
                'no_discoloration': True,
                'no_heat_marks': True,
                'passivation_required': True,
                'cleanroom_handling': True
            }
        )

        # Aluminum (AL6061)
        standards['AL6061'] = QualityStandard(
            material_type='AL6061',
            thickness_range=(1.0, 15.0),
            dimensional_tolerance=0.3,
            edge_roughness_max=1.6,
            perpendicularity_tolerance=0.2,
            parallelism_tolerance=0.15,
            burn_free_edge_required=True,
            deburring_required=True,
            surface_finish_requirements={
                'no_melting': True,
                'smooth_finish': True
            }
        )

        # Aluminum (AL5052)
        standards['AL5052'] = QualityStandard(
            material_type='AL5052',
            thickness_range=(0.8, 12.0),
            dimensional_tolerance=0.3,
            edge_roughness_max=1.6,
            perpendicularity_tolerance=0.2,
            parallelism_tolerance=0.15,
            burn_free_edge_required=True,
            deburring_required=True,
            surface_finish_requirements={
                'no_melting': True,
                'corrosion_resistance': True
            }
        )

        return standards

    def _initialize_inspection_templates(self) -> Dict[str, List[str]]:
        """Initialize inspection templates for different checkpoint types"""
        templates = {
            CheckpointType.DIMENSIONAL.value: [
                "Measure width with caliper",
                "Measure height with caliper",
                "Check diagonal measurements",
                "Verify corner squareness",
                "Record measurements on inspection sheet"
            ],
            CheckpointType.VISUAL.value: [
                "Inspect for surface defects",
                "Check for cracks or damage",
                "Verify marking/identification",
                "Confirm quantity count",
                "Document any anomalies"
            ],
            CheckpointType.EDGE_QUALITY.value: [
                "Inspect cut edge smoothness",
                "Check for burn marks",
                "Verify edge perpendicularity",
                "Assess need for deburring",
                "Measure edge roughness if required"
            ],
            CheckpointType.MATERIAL.value: [
                "Verify material certificate",
                "Check material marking",
                "Confirm thickness with caliper",
                "Validate material properties",
                "Document material traceability"
            ],
            CheckpointType.SETUP.value: [
                "Check machine calibration",
                "Verify cutting parameters",
                "Test emergency stops",
                "Confirm tool condition",
                "Validate safety systems"
            ]
        }
        return templates

    def create_inspection_checklist(
        self,
        checkpoints: List[QualityCheckpoint]
    ) -> Dict[str, Any]:
        """
        Create printable inspection checklist
        印刷可能な検査チェックリストを作成
        """
        checklist = {
            'title': '品質検査チェックリスト / Quality Inspection Checklist',
            'generated_at': datetime.now().isoformat(),
            'total_checkpoints': len(checkpoints),
            'estimated_time': sum(cp.estimated_time for cp in checkpoints),
            'checkpoints': []
        }

        for checkpoint in checkpoints:
            checkpoint_data = {
                'id': checkpoint.checkpoint_id,
                'step': checkpoint.step_number,
                'type': checkpoint.checkpoint_type.value,
                'description': checkpoint.description,
                'japanese_description': checkpoint.japanese_description,
                'method': checkpoint.inspection_method.value,
                'critical': checkpoint.critical_flag,
                'estimated_time': checkpoint.estimated_time,
                'required_tools': checkpoint.required_tools,
                'pass_criteria': checkpoint.pass_criteria,
                'signature_line': '検査者署名 / Inspector Signature: _______________',
                'date_line': '検査日時 / Inspection Date: _______________'
            }

            if checkpoint.target_value is not None:
                checkpoint_data['target_value'] = checkpoint.target_value
                checkpoint_data['tolerance'] = checkpoint.tolerance

            checklist['checkpoints'].append(checkpoint_data)

        return checklist

    def record_inspection_result(
        self,
        checkpoint_id: str,
        inspector: str,
        pass_status: bool,
        measured_value: Optional[float] = None,
        notes: str = ""
    ) -> QualityRecord:
        """
        Record inspection result
        検査結果を記録
        """
        record = QualityRecord(
            checkpoint_id=checkpoint_id,
            inspector=inspector,
            timestamp=datetime.now(),
            measured_value=measured_value,
            pass_status=pass_status,
            notes=notes
        )

        self.quality_records.append(record)

        self.logger.info(
            f"Recorded inspection result for {checkpoint_id}: "
            f"{'PASS' if pass_status else 'FAIL'} by {inspector}"
        )

        return record

    def generate_quality_report(
        self,
        work_instruction: WorkInstruction,
        checkpoints: List[QualityCheckpoint]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        包括的品質レポートを生成
        """
        completed_records = [r for r in self.quality_records
                           if r.checkpoint_id in [cp.checkpoint_id for cp in checkpoints]]

        total_checkpoints = len(checkpoints)
        completed_checkpoints = len(completed_records)
        passed_checkpoints = len([r for r in completed_records if r.pass_status])

        pass_rate = (passed_checkpoints / completed_checkpoints * 100) if completed_checkpoints > 0 else 0

        report = {
            'work_instruction_id': work_instruction.sheet_id,
            'material_type': work_instruction.material_type,
            'quality_level': self.quality_level.value,
            'summary': {
                'total_checkpoints': total_checkpoints,
                'completed_checkpoints': completed_checkpoints,
                'passed_checkpoints': passed_checkpoints,
                'pass_rate': pass_rate,
                'completion_rate': (completed_checkpoints / total_checkpoints * 100) if total_checkpoints > 0 else 0
            },
            'checkpoint_results': [],
            'recommendations': [],
            'generated_at': datetime.now().isoformat()
        }

        # Add individual checkpoint results
        for checkpoint in checkpoints:
            record = next((r for r in completed_records if r.checkpoint_id == checkpoint.checkpoint_id), None)

            result = {
                'checkpoint_id': checkpoint.checkpoint_id,
                'description': checkpoint.description,
                'critical': checkpoint.critical_flag,
                'status': 'completed' if record else 'pending',
                'pass_status': record.pass_status if record else None,
                'measured_value': record.measured_value if record else None,
                'target_value': checkpoint.target_value,
                'tolerance': checkpoint.tolerance,
                'inspector': record.inspector if record else None,
                'timestamp': record.timestamp.isoformat() if record else None,
                'notes': record.notes if record else ""
            }

            report['checkpoint_results'].append(result)

        # Generate recommendations
        if pass_rate < 90:
            report['recommendations'].append(
                "パス率が90%未満です。プロセス改善を検討してください。 / "
                "Pass rate below 90%. Consider process improvements."
            )

        failed_critical = [r for r in completed_records
                         if not r.pass_status and
                         any(cp.critical_flag for cp in checkpoints if cp.checkpoint_id == r.checkpoint_id)]

        if failed_critical:
            report['recommendations'].append(
                f"{len(failed_critical)}個の重要チェックポイントが失敗しました。即座の対応が必要です。 / "
                f"{len(failed_critical)} critical checkpoints failed. Immediate action required."
            )

        return report

    def get_quality_standard(self, material_type: str) -> Optional[QualityStandard]:
        """Get quality standard for material type"""
        return self.quality_standards.get(material_type)

    def export_quality_data(self, format_type: str = 'json') -> str:
        """
        Export quality data in specified format
        指定形式で品質データをエクスポート
        """
        data = {
            'quality_records': [
                {
                    'checkpoint_id': r.checkpoint_id,
                    'inspector': r.inspector,
                    'timestamp': r.timestamp.isoformat(),
                    'measured_value': r.measured_value,
                    'pass_status': r.pass_status,
                    'notes': r.notes,
                    'corrective_actions': r.corrective_actions,
                    'verification_required': r.verification_required
                }
                for r in self.quality_records
            ],
            'quality_standards': {
                material: {
                    'material_type': std.material_type,
                    'thickness_range': std.thickness_range,
                    'dimensional_tolerance': std.dimensional_tolerance,
                    'edge_roughness_max': std.edge_roughness_max,
                    'perpendicularity_tolerance': std.perpendicularity_tolerance,
                    'parallelism_tolerance': std.parallelism_tolerance,
                    'burn_free_edge_required': std.burn_free_edge_required,
                    'deburring_required': std.deburring_required,
                    'surface_finish_requirements': std.surface_finish_requirements
                }
                for material, std in self.quality_standards.items()
            },
            'exported_at': datetime.now().isoformat()
        }

        if format_type.lower() == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


def create_quality_manager(quality_level: QualityLevel = QualityLevel.STANDARD) -> QualityManager:
    """Create quality manager instance"""
    return QualityManager(quality_level)