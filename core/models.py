"""
Core data models for steel cutting optimization system
鋼板切断最適化システムのコアデータモデル
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, validator, Field
import uuid
from datetime import datetime


@dataclass
class Panel:
    """
    Panel data structure with validation
    パネルデータ構造（検証付き）
    """
    id: str
    width: float  # mm (50-1500)
    height: float  # mm (50-3100)
    quantity: int
    material: str  # Material block key
    thickness: float  # mm
    priority: int = 1  # 1-10
    allow_rotation: bool = True
    block_order: int = 0  # Order within material block
    
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
        """Calculate panel area in mm²"""
        return self.width * self.height
    
    @property
    def rotated(self) -> 'Panel':
        """Return rotated version of panel if rotation allowed"""
        if not self.allow_rotation:
            return self
        return Panel(
            id=f"{self.id}_rotated",
            width=self.height,
            height=self.width,
            quantity=self.quantity,
            material=self.material,
            thickness=self.thickness,
            priority=self.priority,
            allow_rotation=self.allow_rotation,
            block_order=self.block_order
        )
    
    def fits_in_sheet(self, sheet_width: float, sheet_height: float) -> bool:
        """Check if panel fits in given sheet dimensions"""
        fits_normal = self.width <= sheet_width and self.height <= sheet_height
        if self.allow_rotation:
            fits_rotated = self.height <= sheet_width and self.width <= sheet_height
            return fits_normal or fits_rotated
        return fits_normal


@dataclass
class SteelSheet:
    """
    Steel sheet (mother material) data structure
    鋼板（母材）データ構造
    """
    width: float = 1500.0  # mm - standard width
    height: float = 3100.0  # mm - standard height
    thickness: float = 6.0  # mm
    material: str = "SS400"  # Steel grade
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
        """Get actual width considering rotation"""
        return self.panel.height if self.rotated else self.panel.width
    
    @property
    def actual_height(self) -> float:
        """Get actual height considering rotation"""
        return self.panel.width if self.rotated else self.panel.height
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) bounds"""
        return (
            self.x,
            self.y,
            self.x + self.actual_width,
            self.y + self.actual_height
        )
    
    def overlaps_with(self, other: 'PlacedPanel') -> bool:
        """Check if this panel overlaps with another"""
        x1, y1, x2, y2 = self.bounds
        ox1, oy1, ox2, oy2 = other.bounds
        
        return not (x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2)


@dataclass
class PlacementResult:
    """
    Result of optimization with placement details
    最適化結果と配置詳細
    """
    sheet_id: int
    material_block: str
    sheet: SteelSheet
    panels: List[PlacedPanel]
    efficiency: float  # 0-1
    waste_area: float  # mm²
    cut_length: float  # mm - total cutting length
    cost: float  # JPY
    algorithm: str = "Unknown"
    processing_time: float = 0.0  # seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def used_area(self) -> float:
        """Calculate total used area"""
        return sum(panel.panel.area for panel in self.panels)
    
    @property
    def total_area(self) -> float:
        """Get total sheet area"""
        return self.sheet.area
    
    def calculate_efficiency(self) -> float:
        """Recalculate efficiency based on placed panels"""
        if self.total_area == 0:
            return 0.0
        self.efficiency = self.used_area / self.total_area
        return self.efficiency
    
    def validate_no_overlaps(self) -> bool:
        """Validate that no panels overlap"""
        for i, panel1 in enumerate(self.panels):
            for panel2 in self.panels[i+1:]:
                if panel1.overlaps_with(panel2):
                    raise ValueError(f"Panels {panel1.panel.id} and {panel2.panel.id} overlap")
        return True
    
    def validate_within_bounds(self) -> bool:
        """Validate all panels are within sheet bounds"""
        for placed_panel in self.panels:
            x1, y1, x2, y2 = placed_panel.bounds
            if x1 < 0 or y1 < 0 or x2 > self.sheet.width or y2 > self.sheet.height:
                raise ValueError(f"Panel {placed_panel.panel.id} exceeds sheet bounds")
        return True


@dataclass
class CuttingInstruction:
    """
    Individual cutting instruction
    個別切断指示
    """
    step_number: int
    cut_type: str  # 'horizontal' or 'vertical'
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    dimension: float  # mm
    description: str
    remaining_pieces: List[str] = field(default_factory=list)
    
    @property
    def cut_length(self) -> float:
        """Calculate cutting length"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


@dataclass
class WorkInstruction:
    """
    Complete work instruction for cutting operation
    切断作業の完全な作業指示
    """
    sheet_id: int
    material_type: str
    total_steps: int
    cutting_sequence: List[CuttingInstruction]
    quality_checkpoints: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    machine_constraints: Dict[str, Any] = field(default_factory=dict)
    estimated_time: float = 0.0  # minutes
    
    def __post_init__(self):
        """Initialize default safety notes"""
        if not self.safety_notes:
            self.safety_notes = [
                "安全メガネ・手袋を着用してください",
                "切断前に材料を確実に固定してください", 
                "切断後の鋭利な縁にご注意ください",
                "作業エリアを清潔に保ってください"
            ]
    
    @property
    def total_cut_length(self) -> float:
        """Calculate total cutting length"""
        return sum(instruction.cut_length for instruction in self.cutting_sequence)


@dataclass
class OptimizationConstraints:
    """
    Constraints for optimization process
    最適化プロセスの制約条件
    """
    max_sheets: int = 10  # Maximum number of sheets to use
    kerf_width: float = 3.5  # mm - cutting allowance
    min_waste_piece: float = 50.0  # mm - minimum usable waste piece
    allow_rotation: bool = True
    material_separation: bool = True  # Separate materials into blocks
    time_budget: float = 30.0  # seconds
    target_efficiency: float = 0.75  # 75%
    
    def validate(self) -> bool:
        """Validate constraint values"""
        if self.max_sheets <= 0:
            raise ValueError("Max sheets must be positive")
        if self.kerf_width < 0:
            raise ValueError("Kerf width cannot be negative")
        if self.min_waste_piece < 0:
            raise ValueError("Min waste piece cannot be negative")
        if not (0 <= self.target_efficiency <= 1):
            raise ValueError("Target efficiency must be between 0 and 1")
        return True


# Pydantic models for API validation
class PanelAPI(BaseModel):
    """Pydantic model for API validation"""
    id: str = Field(..., min_length=1, max_length=50)
    width: float = Field(..., ge=50, le=1500, description="Width in mm (50-1500)")
    height: float = Field(..., ge=50, le=3100, description="Height in mm (50-3100)")
    quantity: int = Field(..., ge=1, description="Quantity must be positive")
    material: str = Field(..., min_length=1, max_length=20)
    thickness: float = Field(..., gt=0, description="Thickness must be positive")
    priority: int = Field(1, ge=1, le=10)
    allow_rotation: bool = True
    
    def to_panel(self) -> Panel:
        """Convert to Panel dataclass"""
        return Panel(
            id=self.id,
            width=self.width,
            height=self.height,
            quantity=self.quantity,
            material=self.material,
            thickness=self.thickness,
            priority=self.priority,
            allow_rotation=self.allow_rotation
        )


class OptimizationRequest(BaseModel):
    """API request model for optimization"""
    panels: List[PanelAPI]
    constraints: Optional[Dict[str, Any]] = None
    algorithm: Optional[str] = "AUTO"
    time_budget: float = Field(30.0, ge=1.0, le=300.0)