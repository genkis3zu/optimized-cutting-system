#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session State Management for Streamlit App
Streamlit アプリのセッション状態管理
"""

import streamlit as st
from typing import List, Optional, Dict, Any
from dataclasses import asdict
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.core_models import Panel, SteelSheet, OptimizationConstraints
from models.batch_models import BatchOptimizationResult

def initialize_session_state():
    """
    Initialize all session state variables with default values
    セッション状態変数を初期値で初期化
    """
    
    # Panel data management
    if 'panels' not in st.session_state:
        st.session_state.panels = []
    
    if 'current_panel_id' not in st.session_state:
        st.session_state.current_panel_id = 1
    
    # Material and sheet templates
    if 'sheet_templates' not in st.session_state:
        st.session_state.sheet_templates = {
            'SGCC': SteelSheet(width=1500, height=3100, thickness=6.0, material='SGCC', cost_per_sheet=15000.0),
            'SPCC': SteelSheet(width=1500, height=3100, thickness=6.0, material='SPCC', cost_per_sheet=14000.0),
            'SS400': SteelSheet(width=1500, height=3100, thickness=6.0, material='SS400', cost_per_sheet=16000.0),
        }
    
    # Optimization settings
    if 'optimization_constraints' not in st.session_state:
        st.session_state.optimization_constraints = OptimizationConstraints(
            kerf_width=3.0,
            safety_margin=5.0,
            min_efficiency_threshold=0.6,
            max_processing_time=60.0
        )
    
    # Results storage
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    
    # UI state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False
    
    # Advanced settings
    if 'show_advanced_settings' not in st.session_state:
        st.session_state.show_advanced_settings = False
    
    # File upload state
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

def add_panel(panel: Panel) -> bool:
    """
    Add a panel to the session state
    パネルをセッション状態に追加
    """
    try:
        # Validate panel before adding
        panel.validate_size()
        
        # Check for duplicate IDs
        existing_ids = [p.id for p in st.session_state.panels]
        if panel.id in existing_ids:
            panel.id = f"{panel.id}_{st.session_state.current_panel_id}"
            st.session_state.current_panel_id += 1
        
        st.session_state.panels.append(panel)
        return True
    except Exception as e:
        st.error(f"Failed to add panel: {str(e)}")
        return False

def remove_panel(panel_id: str) -> bool:
    """
    Remove a panel from session state
    セッション状態からパネルを削除
    """
    try:
        st.session_state.panels = [p for p in st.session_state.panels if p.id != panel_id]
        return True
    except Exception as e:
        st.error(f"Failed to remove panel: {str(e)}")
        return False

def update_panel(panel_id: str, updated_panel: Panel) -> bool:
    """
    Update an existing panel in session state
    セッション状態の既存パネルを更新
    """
    try:
        updated_panel.validate_size()
        
        for i, panel in enumerate(st.session_state.panels):
            if panel.id == panel_id:
                st.session_state.panels[i] = updated_panel
                return True
        
        st.error(f"Panel {panel_id} not found")
        return False
    except Exception as e:
        st.error(f"Failed to update panel: {str(e)}")
        return False

def get_panels_by_material(material: str) -> List[Panel]:
    """
    Get all panels of a specific material
    特定材料のパネルを全て取得
    """
    return [p for p in st.session_state.panels if p.material == material]

def get_material_summary() -> Dict[str, Dict[str, Any]]:
    """
    Get summary statistics by material
    材料別サマリー統計を取得
    """
    summary = {}
    
    for panel in st.session_state.panels:
        if panel.material not in summary:
            summary[panel.material] = {
                'panel_count': 0,
                'total_quantity': 0,
                'total_area': 0.0,
                'thickness': panel.thickness
            }
        
        summary[panel.material]['panel_count'] += 1
        summary[panel.material]['total_quantity'] += panel.quantity
        summary[panel.material]['total_area'] += panel.cutting_area * panel.quantity
    
    return summary

def clear_all_panels():
    """
    Clear all panels from session state
    セッション状態から全パネルをクリア
    """
    st.session_state.panels = []
    st.session_state.current_panel_id = 1

def save_optimization_results(results: BatchOptimizationResult):
    """
    Save optimization results to session state and history
    最適化結果をセッション状態と履歴に保存
    """
    st.session_state.optimization_results = results
    
    # Add to history with timestamp
    import datetime
    history_entry = {
        'timestamp': datetime.datetime.now(),
        'panel_count': len(st.session_state.panels),
        'material_count': len(get_material_summary()),
        'efficiency': results.overall_efficiency,
        'sheets_used': sum(len(r.placement_results) for r in results.processing_results),
        'processing_time': results.total_processing_time
    }
    
    st.session_state.processing_history.append(history_entry)
    
    # Keep only last 10 entries
    if len(st.session_state.processing_history) > 10:
        st.session_state.processing_history = st.session_state.processing_history[-10:]

def get_optimization_constraints() -> OptimizationConstraints:
    """Get current optimization constraints"""
    return st.session_state.optimization_constraints

def update_optimization_constraints(**kwargs):
    """Update optimization constraints"""
    current = st.session_state.optimization_constraints
    
    # Create updated constraints
    updated = OptimizationConstraints(
        kerf_width=kwargs.get('kerf_width', current.kerf_width),
        safety_margin=kwargs.get('safety_margin', current.safety_margin),
        min_efficiency_threshold=kwargs.get('min_efficiency_threshold', current.min_efficiency_threshold),
        max_processing_time=kwargs.get('max_processing_time', current.max_processing_time),
        allow_mixed_materials=kwargs.get('allow_mixed_materials', current.allow_mixed_materials),
        max_sheets_per_material=kwargs.get('max_sheets_per_material', current.max_sheets_per_material)
    )
    
    st.session_state.optimization_constraints = updated

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_material_options() -> List[str]:
    """Get cached material options for dropdowns"""
    return ['SGCC', 'SPCC', 'SS400', 'SUS304', 'SUS430', 'SECC']

@st.cache_data(ttl=300)
def get_cached_thickness_options() -> List[float]:
    """Get cached thickness options"""
    return [1.0, 1.2, 1.6, 2.0, 2.3, 3.2, 4.5, 6.0, 8.0, 9.0, 12.0]

def export_session_data() -> Dict[str, Any]:
    """
    Export current session data for download
    現在のセッションデータをダウンロード用にエクスポート
    """
    return {
        'panels': [asdict(panel) for panel in st.session_state.panels],
        'optimization_constraints': asdict(st.session_state.optimization_constraints),
        'sheet_templates': {k: asdict(v) for k, v in st.session_state.sheet_templates.items()},
        'export_timestamp': str(datetime.datetime.now())
    }