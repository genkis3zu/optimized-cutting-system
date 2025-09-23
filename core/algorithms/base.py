"""
Base classes for optimization algorithms
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from core.models import Panel, SteelSheet, PlacementResult, OptimizationConstraints


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms
    """

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def optimize(
        self,
        panels: List[Panel],
        sheets: List[SteelSheet],
        constraints: Optional[Dict[str, Any]] = None
    ) -> PlacementResult:
        """
        Main optimization method
        """
        pass

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time in seconds
        """
        return 0.0

    def calculate_complexity(self, panels: List[Panel]) -> float:
        """
        Calculate problem complexity (0-1)
        """
        if not panels:
            return 0.0

        # Size diversity factor
        unique_sizes = len(set((p.width, p.height) for p in panels))
        size_diversity = min(1.0, unique_sizes / len(panels))

        # Quantity factor
        total_quantity = sum(p.quantity for p in panels)
        quantity_factor = min(1.0, total_quantity / 100)

        # Rotation complexity
        rotation_factor = sum(1 for p in panels if p.allow_rotation) / len(panels)

        # Material diversity
        unique_materials = len(set(p.material for p in panels))
        material_factor = min(1.0, unique_materials / 10)

        # Combined complexity
        complexity = (
            size_diversity * 0.4 +
            quantity_factor * 0.3 +
            rotation_factor * 0.2 +
            material_factor * 0.1
        )

        return min(1.0, complexity)

    def validate_placement(self, placement: PlacementResult) -> bool:
        """
        Validate placement result
        """
        try:
            # Basic validation
            if not placement:
                return False
            return True
        except Exception:
            return False