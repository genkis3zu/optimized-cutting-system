#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models package for the new steel cutting optimization system
新しい鋼板切断最適化システムのモデルパッケージ
"""

from .core_models import Panel, SteelSheet, PlacedPanel, PlacementResult, OptimizationConstraints
from .batch_models import MaterialBatch, ProcessingResult, BatchOptimizationResult
from .residual_models import ResidualMaterial, CombinationResult
from .optimization_models import OptimizationStrategy, PlacementMetrics

__all__ = [
    # Core models
    'Panel', 'SteelSheet', 'PlacedPanel', 'PlacementResult', 'OptimizationConstraints',

    # Batch models
    'MaterialBatch', 'ProcessingResult', 'BatchOptimizationResult',

    # Residual models
    'ResidualMaterial', 'CombinationResult',

    # Optimization models
    'OptimizationStrategy', 'PlacementMetrics'
]