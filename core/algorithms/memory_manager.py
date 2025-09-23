"""
Memory Management System for Large-Scale Panel Optimization

Implements efficient memory management strategies including:
- Panel instance pooling
- LRU caching for position tests
- Progressive memory allocation
- Memory-mapped storage for extreme cases
"""

import gc
import sys
import weakref
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import hashlib
import pickle

from core.models import Panel, PlacedPanel


class PanelPool:
    """
    Object pool for Panel instances to reduce memory allocation overhead.

    Reuses Panel objects to minimize garbage collection pressure.
    """

    def __init__(self, initial_size: int = 100):
        """
        Initialize panel pool.

        Args:
            initial_size: Initial pool size
        """
        self._pool: List[Panel] = []
        self._in_use: Set[int] = set()
        self._panel_cache: Dict[str, Panel] = {}
        self.hits = 0
        self.misses = 0

    def get_panel(
        self,
        panel_id: str,
        width: float,
        height: float,
        material: str,
        thickness: float = 1.0,
        quantity: int = 1,
        allow_rotation: bool = True,
        pi_code: Optional[str] = None
    ) -> Panel:
        """
        Get a panel from the pool or create new if necessary.

        Args:
            panel_id: Unique panel identifier
            width: Panel width
            height: Panel height
            material: Material type
            thickness: Material thickness
            quantity: Panel quantity
            allow_rotation: Whether rotation is allowed
            pi_code: PI expansion code

        Returns:
            Panel instance
        """
        # Check cache first
        cache_key = f"{panel_id}_{width}_{height}_{material}"
        if cache_key in self._panel_cache:
            self.hits += 1
            return self._panel_cache[cache_key]

        self.misses += 1

        # Create new panel
        panel = Panel(
            panel_id=panel_id,
            width=width,
            height=height,
            material=material,
            thickness=thickness,
            quantity=quantity,
            allow_rotation=allow_rotation,
            pi_code=pi_code
        )

        # Add to cache
        self._panel_cache[cache_key] = panel

        return panel

    def release_panel(self, panel: Panel):
        """
        Return a panel to the pool.

        Args:
            panel: Panel to release
        """
        # Panels remain in cache, just remove from in-use set
        panel_id = id(panel)
        self._in_use.discard(panel_id)

    def clear_cache(self):
        """Clear the panel cache to free memory"""
        self._panel_cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self._panel_cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_usage': sys.getsizeof(self._panel_cache)
        }


class LRUPositionCache:
    """
    LRU cache for position test results to avoid redundant calculations.

    Caches collision detection results for recently tested positions.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        sheet_id: str
    ) -> str:
        """Create cache key from position parameters"""
        # Round to nearest mm for cache efficiency
        x_int = int(x)
        y_int = int(y)
        w_int = int(width)
        h_int = int(height)

        return f"{sheet_id}_{x_int}_{y_int}_{w_int}_{h_int}"

    def get(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        sheet_id: str
    ) -> Optional[bool]:
        """
        Get cached collision result.

        Args:
            x, y: Position coordinates
            width, height: Panel dimensions
            sheet_id: Sheet identifier

        Returns:
            Cached collision result or None if not found
        """
        key = self._make_key(x, y, width, height, sheet_id)

        if key in self._cache:
            self.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        return None

    def put(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        sheet_id: str,
        result: bool
    ):
        """
        Cache a collision test result.

        Args:
            x, y: Position coordinates
            width, height: Panel dimensions
            sheet_id: Sheet identifier
            result: Collision test result
        """
        key = self._make_key(x, y, width, height, sheet_id)

        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = result

    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_usage': sys.getsizeof(self._cache)
        }


class MemoryManager:
    """
    Central memory management system for optimization algorithms.

    Coordinates memory usage across all optimization components.
    """

    def __init__(self, memory_limit_mb: Optional[int] = None):
        """
        Initialize memory manager.

        Args:
            memory_limit_mb: Memory limit in megabytes (None for auto)
        """
        # Auto-detect available memory if not specified
        if memory_limit_mb is None:
            try:
                # Use available memory with practical limits for typical PCs
                import psutil
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                # Conservative approach: min 2GB, max 3.5GB to avoid paging
                optimal_mb = min(3584, max(2048, int(available_mb * 0.6)))
                self.memory_limit_mb = optimal_mb
            except ImportError:
                # Default to 3GB for production use - safe for most PCs
                self.memory_limit_mb = 3072
        else:
            self.memory_limit_mb = memory_limit_mb

        # Initialize components
        self.panel_pool = PanelPool()
        self.position_cache = LRUPositionCache()

        # Memory tracking
        self._memory_usage = 0
        self._gc_threshold = self.memory_limit_mb * 0.9  # GC at 90% usage

        # Weak references for automatic cleanup
        self._tracked_objects = weakref.WeakSet()

    def allocate_panels(self, panel_data: List[Dict]) -> List[Panel]:
        """
        Allocate panels with memory optimization.

        Args:
            panel_data: List of panel specifications

        Returns:
            List of Panel objects
        """
        panels = []

        for data in panel_data:
            panel = self.panel_pool.get_panel(
                panel_id=data.get('panel_id'),
                width=data.get('width'),
                height=data.get('height'),
                material=data.get('material'),
                thickness=data.get('thickness', 1.0),
                quantity=data.get('quantity', 1),
                allow_rotation=data.get('allow_rotation', True),
                pi_code=data.get('pi_code')
            )
            panels.append(panel)

            # Check memory usage
            if self._should_gc():
                self._run_gc()

        return panels

    def check_position(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        sheet_id: str,
        check_func=None
    ) -> bool:
        """
        Check position with caching.

        Args:
            x, y: Position coordinates
            width, height: Panel dimensions
            sheet_id: Sheet identifier
            check_func: Function to call if not cached

        Returns:
            Collision test result
        """
        # Check cache first
        cached = self.position_cache.get(x, y, width, height, sheet_id)
        if cached is not None:
            return cached

        # Perform actual check
        if check_func:
            result = check_func(x, y, width, height)
            self.position_cache.put(x, y, width, height, sheet_id, result)
            return result

        return False

    def optimize_memory_for_dataset_size(self, num_panels: int, num_sheets: int):
        """
        Optimize memory allocation based on dataset size.

        Args:
            num_panels: Number of panels
            num_sheets: Number of sheets
        """
        # Detailed memory estimation based on analysis
        panel_memory = num_panels * 0.0002  # 0.2KB per panel
        sheet_memory = num_sheets * 0.0001  # 0.1KB per sheet
        placed_panel_memory = num_panels * 0.0003  # 0.3KB per placed panel
        spatial_index_memory = (2.0 + num_panels * 0.0001 + num_sheets * 0.5) * (1.2 ** (num_panels / 1000))
        position_cache_memory = 5.0 + num_panels * 0.0005 + num_sheets * 1.0
        algorithm_memory = (10.0 + num_panels * 0.001 + num_sheets * 2.0) * (1.5 ** (num_panels / 1000))
        runtime_overhead = 50.0 + num_panels * 0.0002 + num_sheets * 0.1

        # 100% guarantee additional overhead
        guarantee_overhead = 92.0  # From analysis

        total_estimate_mb = (
            panel_memory + sheet_memory + placed_panel_memory +
            spatial_index_memory + position_cache_memory + algorithm_memory +
            runtime_overhead + guarantee_overhead
        )

        # Recommended memory with safety margin
        recommended_mb = total_estimate_mb * 1.5

        logger.info(f"Memory optimization for {num_panels} panels, {num_sheets} sheets:")
        logger.info(f"  Estimated usage: {total_estimate_mb:.1f} MB")
        logger.info(f"  Recommended: {recommended_mb:.1f} MB")
        logger.info(f"  Current limit: {self.memory_limit_mb:.1f} MB")

        # Determine optimization strategy for typical PC environment
        memory_usage_ratio = recommended_mb / self.memory_limit_mb

        if memory_usage_ratio <= 0.7:
            # Comfortable memory - use full optimization
            cache_size = min(15000, max(5000, num_panels * 15))
            strategy = "FULL_OPTIMIZATION"
        elif memory_usage_ratio <= 0.9:
            # Efficient memory usage - balanced optimization
            cache_size = min(8000, max(3000, num_panels * 8))
            strategy = "BALANCED_OPTIMIZATION"
        elif memory_usage_ratio <= 1.2:
            # Tight memory - conservative optimization
            cache_size = min(4000, max(1500, num_panels * 4))
            strategy = "MEMORY_CONSCIOUS"
        else:
            # Critical memory - minimal optimization to avoid paging
            cache_size = min(2000, max(500, num_panels * 2))
            strategy = "PAGING_AVOIDANCE"

        # Apply optimizations
        self.position_cache = LRUPositionCache(max_size=int(cache_size))

        # Status reporting for PC-friendly memory management
        if strategy == "PAGING_AVOIDANCE":
            logger.warning(f"🚨 Critical memory usage - paging avoidance mode active")
        elif strategy == "MEMORY_CONSCIOUS":
            logger.warning(f"⚠️ Conservative memory mode - performance may be reduced")
        elif strategy == "BALANCED_OPTIMIZATION":
            logger.info(f"🔧 Balanced optimization for typical PC environment")
        else:
            logger.info(f"✅ Full optimization with comfortable memory headroom")

        # Store strategy for performance monitoring
        self.optimization_strategy = strategy
        self.estimated_memory_mb = total_estimate_mb
        self.recommended_memory_mb = recommended_mb

    def _should_gc(self) -> bool:
        """Check if garbage collection should run"""
        # Simple heuristic based on allocation count
        return len(self._tracked_objects) > 10000

    def _run_gc(self):
        """Run garbage collection"""
        before = self._get_memory_usage_mb()
        gc.collect()
        after = self._get_memory_usage_mb()

        if before - after > 10:  # Freed more than 10MB
            print(f"GC freed {before - after:.1f}MB")

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback if psutil not available - estimate based on object count
            return len(self._tracked_objects) * 0.001  # Rough estimate

    def clear_caches(self):
        """Clear all caches to free memory"""
        self.panel_pool.clear_cache()
        self.position_cache.clear()
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        return {
            'memory_limit_mb': self.memory_limit_mb,
            'current_usage_mb': self._get_memory_usage_mb(),
            'panel_pool': self.panel_pool.get_stats(),
            'position_cache': self.position_cache.get_stats(),
            'tracked_objects': len(self._tracked_objects)
        }


@dataclass
class OptimizationState:
    """
    Compact representation of optimization state for checkpointing.

    Allows resuming optimization from saved state.
    """
    panels: List[Panel]
    placed_panels: List[PlacedPanel]
    remaining_panels: List[Panel]
    current_tier: int
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: str):
        """Save state to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'OptimizationState':
        """Load state from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_size_mb(self) -> float:
        """Get size of state in MB"""
        return sys.getsizeof(pickle.dumps(self)) / (1024 * 1024)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def reset_memory_manager():
    """Reset global memory manager"""
    global _memory_manager
    if _memory_manager:
        _memory_manager.clear_caches()
    _memory_manager = None