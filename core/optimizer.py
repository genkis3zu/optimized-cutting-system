"""
Optimization engine for steel cutting with algorithm selection
鋼板切断最適化エンジン（アルゴリズム選択機能付き）
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import threading

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel, 
    OptimizationConstraints
)


@dataclass
class OptimizationMetrics:
    """Performance metrics for optimization"""
    algorithm: str
    processing_time: float
    efficiency: float
    panels_placed: int
    sheets_used: int
    memory_usage: float = 0.0
    iterations: int = 0


class OptimizationAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms
    最適化アルゴリズムの抽象基底クラス
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def optimize(
        self, 
        panels: List[Panel], 
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> PlacementResult:
        """
        Main optimization method
        メイン最適化メソッド
        """
        pass
    
    @abstractmethod
    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time in seconds
        処理時間の見積もり（秒）
        """
        pass
    
    def calculate_complexity(self, panels: List[Panel]) -> float:
        """
        Calculate problem complexity (0-1)
        問題の複雑度を計算（0-1）
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
    
    def group_by_material(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """
        Group panels by material type
        材質別にパネルをグループ化
        """
        material_groups = defaultdict(list)
        for panel in panels:
            # Expand panels based on quantity
            for i in range(panel.quantity):
                individual_panel = Panel(
                    id=f"{panel.id}_{i+1}" if panel.quantity > 1 else panel.id,
                    width=panel.width,
                    height=panel.height,
                    quantity=1,
                    material=panel.material,
                    thickness=panel.thickness,
                    priority=panel.priority,
                    allow_rotation=panel.allow_rotation,
                    block_order=panel.block_order
                )
                material_groups[panel.material].append(individual_panel)
        
        return dict(material_groups)
    
    def validate_placement(self, placement: PlacementResult) -> bool:
        """
        Validate placement result
        配置結果の検証
        """
        try:
            placement.validate_no_overlaps()
            placement.validate_within_bounds()
            return True
        except ValueError as e:
            self.logger.error(f"Placement validation failed: {e}")
            return False


class OptimizationEngine:
    """
    Main optimization engine with algorithm selection
    アルゴリズム選択機能付きメイン最適化エンジン
    """
    
    def __init__(self):
        self.algorithms = {}
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        self.timeout_manager = TimeoutManager()
        
    def register_algorithm(self, algorithm: OptimizationAlgorithm):
        """Register an optimization algorithm"""
        self.algorithms[algorithm.name] = algorithm
        self.logger.info(f"Registered algorithm: {algorithm.name}")
    
    def select_algorithm(
        self, 
        panels: List[Panel], 
        constraints: OptimizationConstraints
    ) -> str:
        """
        Select optimal algorithm based on problem characteristics
        問題特性に基づく最適アルゴリズムの選択
        """
        if not panels:
            return "FFD"  # Default for empty input
        
        # Calculate problem complexity
        complexity = self._calculate_problem_complexity(panels)
        time_budget = constraints.time_budget
        panel_count = sum(p.quantity for p in panels)
        
        # Algorithm selection logic based on specification
        if complexity < 0.3 and time_budget >= 1.0:
            return "FFD"  # Fast, 70-75% efficiency
        elif complexity < 0.7 and time_budget >= 5.0:
            return "BFD"  # Balanced, 80-85% efficiency
        elif time_budget >= 30.0:
            return "HYBRID"  # Optimization-focused, 85%+ efficiency
        else:
            return "FFD_TIMEOUT"  # Fast with timeout handling
    
    def _calculate_problem_complexity(self, panels: List[Panel]) -> float:
        """Calculate normalized problem complexity"""
        if not panels:
            return 0.0
        
        # Use first algorithm to calculate complexity
        if self.algorithms:
            first_algo = next(iter(self.algorithms.values()))
            return first_algo.calculate_complexity(panels)
        
        # Fallback calculation
        unique_sizes = len(set((p.width, p.height) for p in panels))
        total_panels = sum(p.quantity for p in panels)
        diversity = unique_sizes / len(panels) if panels else 0
        
        return min(1.0, (total_panels * diversity) / 1000)
    
    def optimize(
        self,
        panels: List[Panel],
        constraints: Optional[OptimizationConstraints] = None,
        algorithm_hint: Optional[str] = None
    ) -> List[PlacementResult]:
        """
        Main optimization method
        メイン最適化メソッド
        """
        if not panels:
            self.logger.warning("No panels provided for optimization")
            return []
        
        # Set default constraints
        if constraints is None:
            constraints = OptimizationConstraints()
        
        constraints.validate()
        
        # Select algorithm
        algorithm_name = algorithm_hint or self.select_algorithm(panels, constraints)
        
        if algorithm_name not in self.algorithms:
            self.logger.warning(f"Algorithm {algorithm_name} not found, using FFD")
            algorithm_name = "FFD"
        
        algorithm = self.algorithms[algorithm_name]
        
        self.logger.info(
            f"Starting optimization with {algorithm_name} "
            f"for {len(panels)} panel types, "
            f"total quantity: {sum(p.quantity for p in panels)}"
        )
        
        # Start performance monitoring
        start_time = time.time()
        self.performance_monitor.start_monitoring()
        
        try:
            # Group panels by material if required
            if constraints.material_separation:
                material_groups = algorithm.group_by_material(panels)
                results = []
                
                for material, material_panels in material_groups.items():
                    self.logger.info(f"Processing material block: {material} ({len(material_panels)} panels)")
                    
                    # Create appropriate sheet for material
                    sheet = SteelSheet(material=material)
                    
                    # Optimize material block
                    result = self._optimize_with_timeout(
                        algorithm, material_panels, sheet, constraints
                    )
                    
                    if result:
                        result.material_block = material
                        results.append(result)
                
                return results
            else:
                # Single optimization run
                sheet = SteelSheet()
                result = self._optimize_with_timeout(
                    algorithm, panels, sheet, constraints
                )
                return [result] if result else []
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return []
        
        finally:
            processing_time = time.time() - start_time
            self.performance_monitor.stop_monitoring()
            
            self.logger.info(f"Optimization completed in {processing_time:.2f} seconds")
    
    def _optimize_with_timeout(
        self,
        algorithm: OptimizationAlgorithm,
        panels: List[Panel],
        sheet: SteelSheet,
        constraints: OptimizationConstraints
    ) -> Optional[PlacementResult]:
        """Execute optimization with timeout handling"""
        
        def run_optimization():
            return algorithm.optimize(panels, sheet, constraints)
        
        try:
            # Use timeout manager for time-limited execution
            result = self.timeout_manager.execute_with_timeout(
                run_optimization,
                timeout=constraints.time_budget
            )
            
            if result and algorithm.validate_placement(result):
                # Calculate final metrics
                result.algorithm = algorithm.name
                result.processing_time = time.time() - result.timestamp.timestamp()
                result.calculate_efficiency()
                
                self.logger.info(
                    f"Algorithm {algorithm.name} achieved {result.efficiency:.1%} efficiency "
                    f"with {len(result.panels)} panels placed"
                )
                
                return result
            else:
                self.logger.warning(f"Algorithm {algorithm.name} produced invalid result")
                return None
        
        except FutureTimeoutError:
            self.logger.warning(f"Algorithm {algorithm.name} timed out after {constraints.time_budget}s")
            return None
        except Exception as e:
            self.logger.error(f"Algorithm {algorithm.name} failed: {e}")
            return None


class PerformanceMonitor:
    """Monitor optimization performance and resource usage"""
    
    def __init__(self):
        self.start_time = None
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.monitoring = True
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def check_resource_limits(self, memory_limit_mb: float = 512) -> bool:
        """Check if resource usage is within limits"""
        if not self.monitoring:
            return True
            
        memory_usage = self.get_memory_usage()
        return memory_usage < memory_limit_mb


class TimeoutManager:
    """Manage algorithm timeouts and recovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TimeoutManager")
    
    def execute_with_timeout(self, func, timeout: float):
        """Execute function with timeout"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            self.logger.warning(f"Function timed out after {timeout} seconds")
            # Note: Cannot actually kill thread in Python
            raise FutureTimeoutError()
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


# Factory function for creating optimization engine
def create_optimization_engine() -> OptimizationEngine:
    """
    Create and configure optimization engine with default algorithms
    デフォルトアルゴリズム付き最適化エンジンの作成
    """
    engine = OptimizationEngine()
    
    # Algorithms will be imported and registered separately
    # to avoid circular imports
    
    return engine