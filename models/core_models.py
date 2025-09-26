#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core data models for the new steel cutting optimization system
新しい鋼板切断最適化システムのコアデータモデル
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum


@dataclass(unsafe_hash=True)
class Panel:
    """
    Panel data structure with validation
    パネルデータ構造（検証付き）
    """
    id: str
    width: float  # mm (50-1500) - 完成寸法 / Finished dimension
    height: float  # mm (50-3100) - 完成寸法 / Finished dimension
    quantity: int
    material: str  # Material block key
    thickness: float  # mm
    priority: int = 1  # 1-10
    allow_rotation: bool = True
    block_order: int = 0  # Order within material block
    pi_code: str = ""  # PIコード for dimension expansion / PI code for dimension expansion

    # 展開寸法 (PIコードによる計算結果) / Expanded dimensions (calculated by PI code)
    expanded_width: Optional[float] = None
    expanded_height: Optional[float] = None

    def __post_init__(self):
        """Validate panel dimensions after initialization"""
        self.validate_size()

    def validate_size(self) -> bool:
        """
        Validate panel size constraints
        パネルサイズ制約の検証
        """
        if not (50 <= self.width <= 1500):
            raise ValueError(f"Panel width {self.width}mm must be between 50-1500mm")
        if not (50 <= self.height <= 3100):
            raise ValueError(f"Panel height {self.height}mm must be between 50-3100mm")
        if self.quantity <= 0:
            raise ValueError(f"Panel quantity must be positive, got {self.quantity}")
        if self.thickness <= 0:
            raise ValueError(f"Panel thickness must be positive, got {self.thickness}")
        return True

    @property
    def area(self) -> float:
        """Calculate panel area in mm² (using finished dimensions)"""
        return self.width * self.height

    @property
    def cutting_area(self) -> float:
        """Calculate cutting area in mm² (using expanded dimensions if available)"""
        if self.expanded_width is not None and self.expanded_height is not None:
            return self.expanded_width * self.expanded_height
        return self.area

    @property
    def cutting_width(self) -> float:
        """Get cutting width (expanded or original)"""
        return self.expanded_width if self.expanded_width is not None else self.width

    @property
    def cutting_height(self) -> float:
        """Get cutting height (expanded or original)"""
        return self.expanded_height if self.expanded_height is not None else self.height


@dataclass
class SteelSheet:
    """
    Steel sheet (mother material) data structure
    鋼板（母材）データ構造
    """
    width: float = 1500.0  # mm - standard width
    height: float = 3100.0  # mm - standard height
    thickness: float = 6.0  # mm
    material: str = "SGCC"  # Steel grade
    cost_per_sheet: float = 15000.0  # JPY
    availability: int = 100  # Stock count
    priority: int = 1  # Usage priority

    def __post_init__(self):
        """Validate sheet dimensions"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Sheet dimensions must be positive")
        if self.thickness <= 0:
            raise ValueError("Sheet thickness must be positive")

    @property
    def area(self) -> float:
        """Calculate sheet area in mm²"""
        return self.width * self.height


@dataclass
class PlacedPanel:
    """
    Panel with placement coordinates
    配置座標付きパネル
    """
    panel: Panel
    x: float  # Bottom-left x coordinate
    y: float  # Bottom-left y coordinate
    rotated: bool = False

    @property
    def actual_width(self) -> float:
        """Get actual cutting width considering rotation"""
        return self.panel.cutting_height if self.rotated else self.panel.cutting_width

    @property
    def actual_height(self) -> float:
        """Get actual cutting height considering rotation"""
        return self.panel.cutting_width if self.rotated else self.panel.cutting_height

    @property
    def expanded_width(self) -> float:
        """Get expanded cutting width considering rotation (clearer naming)"""
        return self.actual_width

    @property
    def expanded_height(self) -> float:
        """Get expanded cutting height considering rotation (clearer naming)"""
        return self.actual_height


@dataclass
class PlacementResult:
    """
    Result of panel placement on a single sheet
    単一シートでのパネル配置結果
    """
    sheet_id: int
    material_block: str
    sheet: SteelSheet
    panels: List[PlacedPanel] = field(default_factory=list)
    efficiency: float = 0.0  # 0-1
    waste_area: float = 0.0  # mm²
    cut_length: float = 0.0  # mm
    cost: float = 0.0  # JPY
    algorithm: str = ""
    processing_time: float = 0.0  # seconds
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.panels:
            used_area = sum(panel.actual_width * panel.actual_height for panel in self.panels)
            self.waste_area = self.sheet.area - used_area
            self.efficiency = used_area / self.sheet.area if self.sheet.area > 0 else 0.0


@dataclass
class OptimizationConstraints:
    """
    Constraints for optimization process
    最適化処理の制約条件
    """
    enable_gpu: bool = True
    gpu_memory_limit: int = 2048  # MB
    max_processing_time: float = 60.0  # seconds
    kerf_width: float = 3.0  # mm - cutting blade width
    safety_margin: float = 5.0  # mm - safety margin
    min_efficiency_threshold: float = 0.6  # 60% minimum efficiency
    allow_mixed_materials: bool = False  # Material mixing not allowed
    max_sheets_per_material: int = 10  # Maximum sheets per material type


class OptimizationType(Enum):
    """
    Type of optimization to perform
    実行する最適化のタイプ
    """
    SIMPLE_MATH = "simple_math"  # Simple mathematical calculation
    COMPLEX_ALGORITHM = "complex_algorithm"  # Complex bin packing algorithm
    HYBRID = "hybrid"  # Combination of both