#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization Execution Page - Steel Cutting Optimizer
最適化実行ページ - 鋼板切断最適化システム
"""

import streamlit as st
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.core_models import Panel
from models.batch_models import BatchOptimizationResult, ProcessingResult
from processing.phase1_individual import IndividualMaterialProcessor
from utils.session_manager import (
    initialize_session_state, 
    get_material_summary,
    get_optimization_constraints,
    save_optimization_results
)
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Optimization - Steel Cutting Optimizer",
    page_icon="🎯",
    layout="wide"
)

# Initialize session
initialize_session_state()

def main():
    """Main optimization page"""
    
    # Render sidebar
    render_sidebar()
    
    # Custom CSS
    st.markdown("""
    <style>
    .optimization-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #FF9800;
    }
    
    .batch-preview {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .status-running { border-left-color: #FF9800; }
    .status-success { border-left-color: #4CAF50; }
    .status-error { border-left-color: #F44336; }
    
    .metric-large {
        font-size: 2rem;
        font-weight: bold;
        color: #2196F3;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FF9800, #F57C00); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🎯 Optimization</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Execute material-based optimization with intelligent strategy selection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have panels
    if not st.session_state.panels:
        st.warning("⚠️ No panels found. Please add panels in the Panel Input page first.")
        if st.button("➡️ Go to Panel Input"):
            st.switch_page("pages/01_🔧_Panel_Input.py")
        return
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Pre-Analysis", "🚀 Execute Optimization", "📈 Real-time Monitor"])
    
    with tab1:
        render_pre_analysis()
    
    with tab2:
        render_optimization_execution()
    
    with tab3:
        render_monitoring()

def render_pre_analysis():
    """Render pre-optimization analysis"""
    st.markdown("### 🔍 Pre-Optimization Analysis")
    
    summary = get_material_summary()
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_panels = len(st.session_state.panels)
    total_quantity = sum(p.quantity for p in st.session_state.panels)
    total_area = sum(p.cutting_area * p.quantity for p in st.session_state.panels) / 1_000_000
    material_count = len(summary)
    
    with col1:
        st.metric("Panel Types", total_panels)
    with col2:
        st.metric("Total Pieces", total_quantity)
    with col3:
        st.metric("Total Area", f"{total_area:.1f} m²")
    with col4:
        st.metric("Materials", material_count)
    
    # Strategy prediction for each material
    st.markdown("#### 🧠 Strategy Prediction by Material")
    
    processor = IndividualMaterialProcessor()
    
    for material, data in summary.items():
        with st.expander(f"📐 {material} - Strategy Analysis", expanded=True):
            material_panels = [p for p in st.session_state.panels if p.material == material]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Analyze each panel type for this material
                strategy_breakdown = {}
                
                for panel in material_panels:
                    # Create a single-panel batch to test strategy
                    from models.batch_models import MaterialBatch
                    test_batch = MaterialBatch(
                        material_type=material,
                        thickness=panel.thickness,
                        panels=[panel],
                        sheet_template=st.session_state.sheet_templates.get(material)
                    )
                    
                    strategy = test_batch.determine_optimization_type()
                    strategy_key = strategy.value
                    
                    if strategy_key not in strategy_breakdown:
                        strategy_breakdown[strategy_key] = []
                    
                    strategy_breakdown[strategy_key].append({
                        'panel': panel,
                        'reason': get_strategy_reason(panel, test_batch.sheet_template)
                    })
                
                # Display strategy breakdown
                for strategy, panel_list in strategy_breakdown.items():
                    st.markdown(f"**{strategy.replace('_', ' ').title()}**: {len(panel_list)} panel types")
                    
                    for item in panel_list:
                        panel = item['panel']
                        reason = item['reason']
                        st.text(f"  • {panel.id} ({panel.cutting_width}×{panel.cutting_height}mm) - {reason}")
            
            with col2:
                # Expected performance metrics
                if len(material_panels) == 1 and material_panels[0].quantity >= 4:
                    # Show expected results for single-panel cases
                    panel = material_panels[0]
                    sheet = st.session_state.sheet_templates.get(material)
                    
                    if sheet:
                        # Calculate expected panels per sheet
                        kerf = get_optimization_constraints().kerf_width
                        panels_w = int((sheet.width + kerf) // (panel.cutting_width + kerf))
                        panels_h = int((sheet.height + kerf) // (panel.cutting_height + kerf))
                        panels_per_sheet = panels_w * panels_h
                        
                        expected_efficiency = (panel.cutting_area * panels_per_sheet) / sheet.area
                        
                        st.metric("Expected/Sheet", f"{panels_per_sheet} panels")
                        st.metric("Expected Efficiency", f"{expected_efficiency:.1%}")
                        
                        if panels_per_sheet == 4 and panel.cutting_width == 968 and panel.cutting_height == 712:
                            st.success("🎯 User's 968×712 case detected!")

def get_strategy_reason(panel: Panel, sheet) -> str:
    """Get human-readable reason for strategy selection"""
    cutting_w = panel.cutting_width
    cutting_h = panel.cutting_height
    
    if not sheet:
        return "No sheet template"
    
    sheet_w = sheet.width
    sheet_h = sheet.height
    
    # Check conditions in order
    if abs(cutting_w - sheet_w) < 1.0:
        return "Panel width matches sheet width exactly"
    
    if abs(cutting_w * 2 - sheet_w) < 1.0:
        return "Two panels fit exactly in width"
    
    if abs(cutting_w * 3 - sheet_w) < 1.0:
        return "Three panels fit exactly in width"
    
    panels_w = int(sheet_w // cutting_w)
    panels_h = int(sheet_h // cutting_h)
    
    if panels_w >= 1 and panels_h >= 1:
        return f"Simple grid: {panels_w}×{panels_h} panels"
    
    return "Complex packing required"

def render_optimization_execution():
    """Render optimization execution interface"""
    st.markdown("### 🚀 Execute Optimization")
    
    # Optimization settings
    with st.container():
        st.markdown('<div class="batch-preview">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Optimization Settings")
        
        constraints = get_optimization_constraints()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Kerf Width", f"{constraints.kerf_width} mm")
            st.metric("Safety Margin", f"{constraints.safety_margin} mm")
        
        with col2:
            st.metric("Min Efficiency", f"{constraints.min_efficiency_threshold:.0%}")
            st.metric("Max Time", f"{constraints.max_processing_time} sec")
        
        with col3:
            st.metric("Material Mixing", "🚫 Disabled")
            st.metric("Legacy Code", "✅ Clean")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Material batch preview
    summary = get_material_summary()
    
    st.markdown("#### 📦 Material Batches to Process")
    
    for material, data in summary.items():
        with st.container():
            st.markdown('<div class="batch-preview">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"**{material}**")
                st.text(f"{data['thickness']}mm thick")
            
            with col2:
                st.metric("Panel Types", data['panel_count'])
            
            with col3:
                st.metric("Total Pieces", data['total_quantity'])
            
            with col4:
                st.metric("Total Area", f"{data['total_area']/1_000_000:.1f} m²")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Execute button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🚀 Start Optimization", 
                    type="primary", 
                    use_container_width=True,
                    help="Begin individual material processing"):
            
            # Store execution flag in session state
            st.session_state.optimization_running = True
            st.session_state.optimization_start_time = time.time()
            st.experimental_rerun()

def render_monitoring():
    """Render real-time monitoring during optimization"""
    st.markdown("### 📈 Real-time Monitor")
    
    # Check if optimization is running
    if hasattr(st.session_state, 'optimization_running') and st.session_state.optimization_running:
        execute_optimization_with_monitoring()
    
    elif st.session_state.optimization_results:
        display_completed_results()
    
    else:
        st.info("🔄 No optimization running. Start optimization in the Execute tab to see real-time monitoring.")

def execute_optimization_with_monitoring():
    """Execute optimization with real-time monitoring"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    results_placeholder = st.empty()
    
    try:
        processor = IndividualMaterialProcessor()
        
        # Update status
        status_placeholder.info("🔄 Starting optimization...")
        progress_bar.progress(0.1)
        
        # Execute optimization
        with st.spinner("Processing materials..."):
            start_time = time.time()
            result = processor.process_all_materials(st.session_state.panels)
            processing_time = time.time() - start_time
        
        # Update progress
        progress_bar.progress(0.8)
        status_placeholder.success("✅ Optimization completed!")
        
        # Save results
        save_optimization_results(result)
        
        # Complete
        progress_bar.progress(1.0)
        
        # Display immediate results
        with results_placeholder.container():
            display_optimization_summary(result)
        
        # Reset running flag
        st.session_state.optimization_running = False
        
        # Auto-navigate to results after a delay
        time.sleep(2)
        st.success("🎯 Optimization completed! Redirecting to Results page...")
        time.sleep(1)
        st.switch_page("pages/03_📊_Results.py")
        
    except Exception as e:
        progress_bar.progress(0.0)
        status_placeholder.error(f"❌ Optimization failed: {str(e)}")
        st.session_state.optimization_running = False
        st.error(f"Error details: {str(e)}")

def display_optimization_summary(result: BatchOptimizationResult):
    """Display quick optimization summary"""
    st.markdown("#### 📊 Quick Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    high_eff_count = len(result.high_efficiency_results)
    low_eff_count = len(result.low_efficiency_results)
    total_sheets = sum(len(r.placement_results) for r in result.processing_results)
    
    with col1:
        st.metric("Overall Efficiency", f"{result.overall_efficiency:.1%}")
    
    with col2:
        st.metric("High Efficiency Batches", high_eff_count)
    
    with col3:
        st.metric("Total Sheets Used", total_sheets)
    
    with col4:
        st.metric("Processing Time", f"{result.total_processing_time:.2f}s")
    
    # Individual batch results
    if result.high_efficiency_results:
        st.success(f"✅ {high_eff_count} material batch(es) achieved high efficiency")
    
    if result.low_efficiency_results:
        st.warning(f"⚠️ {low_eff_count} material batch(es) need residual processing")

def display_completed_results():
    """Display monitoring for completed optimization"""
    result = st.session_state.optimization_results
    
    if not result:
        st.info("No optimization results found.")
        return
    
    st.markdown("#### 📈 Completed Optimization Monitor")
    
    # Create monitoring dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # Efficiency chart
        fig_efficiency = create_efficiency_chart(result)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        # Material breakdown
        fig_material = create_material_breakdown_chart(result)
        st.plotly_chart(fig_material, use_container_width=True)
    
    # Processing details
    st.markdown("#### 🔍 Processing Details")
    
    for i, batch_result in enumerate(result.processing_results):
        with st.expander(f"{batch_result.batch.material_type} - {batch_result.efficiency:.1%} efficiency"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sheets Used", batch_result.total_sheets_used)
                st.metric("Algorithm", batch_result.algorithm_used)
            
            with col2:
                st.metric("Processing Time", f"{batch_result.processing_time:.2f}s")
                st.metric("Unplaced Panels", len(batch_result.unplaced_panels))
            
            with col3:
                if batch_result.placement_results:
                    first_sheet = batch_result.placement_results[0]
                    st.metric("Panels/Sheet", len(first_sheet.panels))
                    st.metric("Sheet Efficiency", f"{first_sheet.efficiency:.1%}")

def create_efficiency_chart(result: BatchOptimizationResult):
    """Create efficiency comparison chart"""
    materials = []
    efficiencies = []
    colors = []
    
    for batch_result in result.processing_results:
        materials.append(batch_result.batch.material_type)
        efficiencies.append(batch_result.efficiency * 100)
        
        # Color based on efficiency
        if batch_result.efficiency >= 0.7:
            colors.append('#4CAF50')  # Green
        elif batch_result.efficiency >= 0.5:
            colors.append('#FF9800')  # Orange
        else:
            colors.append('#F44336')  # Red
    
    fig = go.Figure(data=[
        go.Bar(x=materials, y=efficiencies, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Material Efficiency Comparison",
        xaxis_title="Material Type",
        yaxis_title="Efficiency (%)",
        height=300
    )
    
    # Add threshold line
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="70% Threshold")
    
    return fig

def create_material_breakdown_chart(result: BatchOptimizationResult):
    """Create material processing breakdown pie chart"""
    high_eff = len(result.high_efficiency_results)
    low_eff = len(result.low_efficiency_results)
    
    fig = go.Figure(data=[
        go.Pie(labels=['High Efficiency', 'Low Efficiency'],
               values=[high_eff, low_eff],
               marker_colors=['#4CAF50', '#FF9800'])
    ])
    
    fig.update_layout(
        title="Batch Efficiency Distribution",
        height=300
    )
    
    return fig

if __name__ == "__main__":
    main()