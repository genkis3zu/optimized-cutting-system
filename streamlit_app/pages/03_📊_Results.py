#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Display Page - Steel Cutting Optimizer
結果表示ページ - 鋼板切断最適化システム
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.core_models import PlacementResult, PlacedPanel
from models.batch_models import BatchOptimizationResult, ProcessingResult
from utils.session_manager import initialize_session_state
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Results - Steel Cutting Optimizer",
    page_icon="📊",
    layout="wide"
)

# Initialize session
initialize_session_state()

def main():
    """Main results page"""
    
    # Render sidebar
    render_sidebar()
    
    # Custom CSS for results styling
    st.markdown("""
    <style>
    .results-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    
    .sheet-visualization {
        border: 2px solid #2196F3;
        border-radius: 8px;
        background: #f8f9fa;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .panel-item {
        background: #e3f2fd;
        border: 1px solid #2196F3;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.25rem;
        font-size: 0.8rem;
    }
    
    .high-efficiency { border-left-color: #4CAF50; }
    .low-efficiency { border-left-color: #FF9800; }
    .failed { border-left-color: #F44336; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50, #388E3C); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">📊 Optimization Results</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Interactive cutting plan visualization and detailed analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have results
    if not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results found. Please run optimization first.")
        if st.button("➡️ Go to Optimization"):
            st.switch_page("pages/02_🎯_Optimization.py")
        return
    
    result = st.session_state.optimization_results
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📐 Cutting Plans", "📊 Analytics", "📄 Export"])
    
    with tab1:
        render_results_overview(result)
    
    with tab2:
        render_cutting_plans(result)
    
    with tab3:
        render_analytics(result)
    
    with tab4:
        render_export_options(result)

def render_results_overview(result: BatchOptimizationResult):
    """Render results overview"""
    st.markdown("### 📈 Optimization Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sheets = sum(len(r.placement_results) for r in result.processing_results)
    total_panels = sum(sum(p.quantity for p in r.batch.panels) for r in result.processing_results)
    high_eff_count = len(result.high_efficiency_results)
    low_eff_count = len(result.low_efficiency_results)
    
    with col1:
        st.metric("Overall Efficiency", f"{result.overall_efficiency:.1%}")
    
    with col2:
        st.metric("Total Sheets", total_sheets)
    
    with col3:
        st.metric("Panels Processed", total_panels)
    
    with col4:
        st.metric("High Efficiency", high_eff_count, help="Batches ≥70% efficiency")
    
    with col5:
        st.metric("Processing Time", f"{result.total_processing_time:.2f}s")
    
    # Performance summary
    st.markdown("#### 🎯 Performance Summary")
    
    if high_eff_count > 0:
        st.success(f"✅ **{high_eff_count}** material batch(es) achieved high efficiency (≥70%)")
    
    if low_eff_count > 0:
        st.warning(f"⚠️ **{low_eff_count}** material batch(es) have low efficiency and may benefit from residual processing")
    
    # Individual batch results
    st.markdown("#### 📦 Batch Results by Material")
    
    for batch_result in result.processing_results:
        efficiency_class = "high-efficiency" if batch_result.efficiency >= 0.7 else "low-efficiency"
        
        with st.container():
            st.markdown(f'<div class="results-card {efficiency_class}">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"### {batch_result.batch.material_type}")
                st.text(f"{batch_result.batch.thickness}mm thick")
            
            with col2:
                st.metric("Efficiency", f"{batch_result.efficiency:.1%}")
                st.metric("Algorithm", batch_result.algorithm_used)
            
            with col3:
                st.metric("Sheets Used", batch_result.total_sheets_used)
                st.metric("Panel Types", len(batch_result.batch.panels))
            
            with col4:
                total_cost = sum(r.cost for r in batch_result.placement_results)
                st.metric("Total Cost", f"¥{total_cost:,.0f}")
                st.metric("Unplaced", len(batch_result.unplaced_panels))
            
            # Show algorithm details for user's case
            if (batch_result.batch.material_type == "SGCC" and 
                batch_result.algorithm_used == "SimpleMath" and 
                len(batch_result.batch.panels) == 1):
                
                panel = batch_result.batch.panels[0]
                if panel.cutting_width == 968 and panel.cutting_height == 712:
                    st.info("🎯 **User's 968×712 case detected!** Successfully using SimpleMath optimization.")
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_cutting_plans(result: BatchOptimizationResult):
    """Render interactive cutting plan visualizations"""
    st.markdown("### 📐 Interactive Cutting Plans")
    
    # Material selector
    materials = [r.batch.material_type for r in result.processing_results]
    selected_material = st.selectbox(
        "Select Material to View",
        options=materials,
        help="Choose material to view detailed cutting plans"
    )
    
    # Get selected batch result
    batch_result = next(r for r in result.processing_results if r.batch.material_type == selected_material)
    
    if not batch_result.placement_results:
        st.info(f"No cutting plans available for {selected_material}")
        return
    
    # Sheet selector
    sheet_options = [f"Sheet {i+1}" for i in range(len(batch_result.placement_results))]
    selected_sheet_idx = st.selectbox(
        "Select Sheet to View",
        options=range(len(sheet_options)),
        format_func=lambda x: sheet_options[x],
        help="Choose specific sheet to visualize"
    )
    
    placement = batch_result.placement_results[selected_sheet_idx]
    
    # Sheet information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sheet Size", f"{placement.sheet.width}×{placement.sheet.height}mm")
    
    with col2:
        st.metric("Panels Placed", len(placement.panels))
    
    with col3:
        st.metric("Efficiency", f"{placement.efficiency:.1%}")
    
    with col4:
        st.metric("Waste Area", f"{placement.waste_area:,.0f}mm²")
    
    # Visualization
    st.markdown("#### 📊 2D Cutting Plan Visualization")
    
    fig = create_cutting_plan_visualization(placement)
    st.plotly_chart(fig, use_container_width=True, height=600)
    
    # Panel details table
    st.markdown("#### 📋 Panel Details")
    
    panel_data = []
    for i, placed_panel in enumerate(placement.panels):
        panel_data.append({
            'Panel': i + 1,
            'ID': placed_panel.panel.id,
            'Position': f"({placed_panel.x:.0f}, {placed_panel.y:.0f})",
            'Size': f"{placed_panel.actual_width:.0f}×{placed_panel.actual_height:.0f}mm",
            'Rotated': "Yes" if placed_panel.rotated else "No",
            'Area': f"{placed_panel.actual_width * placed_panel.actual_height:,.0f}mm²"
        })
    
    df = pd.DataFrame(panel_data)
    st.dataframe(df, use_container_width=True)

def create_cutting_plan_visualization(placement: PlacementResult):
    """Create interactive 2D cutting plan visualization"""
    
    fig = go.Figure()
    
    # Add sheet outline
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=placement.sheet.width, y1=placement.sheet.height,
        line=dict(color="black", width=3),
        fillcolor="lightgray",
        fillalpha=0.1
    )
    
    # Color palette for panels
    colors = px.colors.qualitative.Set3
    
    # Add panels
    for i, placed_panel in enumerate(placement.panels):
        color = colors[i % len(colors)]
        
        # Panel rectangle
        fig.add_shape(
            type="rect",
            x0=placed_panel.x,
            y0=placed_panel.y,
            x1=placed_panel.x + placed_panel.actual_width,
            y1=placed_panel.y + placed_panel.actual_height,
            line=dict(color=color, width=2),
            fillcolor=color,
            fillalpha=0.6
        )
        
        # Panel label
        center_x = placed_panel.x + placed_panel.actual_width / 2
        center_y = placed_panel.y + placed_panel.actual_height / 2
        
        label = f"{placed_panel.panel.id}"
        if placed_panel.rotated:
            label += " (R)"
        
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=label,
            showarrow=False,
            font=dict(color="black", size=10),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Sheet {placement.sheet_id} - {placement.sheet.material} ({placement.sheet.width}×{placement.sheet.height}mm)",
        xaxis_title="Width (mm)",
        yaxis_title="Height (mm)",
        xaxis=dict(range=[0, placement.sheet.width], constrain="domain"),
        yaxis=dict(range=[0, placement.sheet.height], scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=600,
        annotations=[
            dict(
                x=placement.sheet.width/2,
                y=-100,
                text=f"Efficiency: {placement.efficiency:.1%} | Algorithm: {placement.algorithm}",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    
    return fig

def render_analytics(result: BatchOptimizationResult):
    """Render detailed analytics"""
    st.markdown("### 📊 Detailed Analytics")
    
    # Efficiency comparison chart
    fig_efficiency = create_efficiency_comparison(result)
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Material cost breakdown
        fig_cost = create_cost_breakdown(result)
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        # Algorithm usage pie chart
        fig_algo = create_algorithm_usage_chart(result)
        st.plotly_chart(fig_algo, use_container_width=True)
    
    # Detailed statistics
    st.markdown("#### 📈 Performance Statistics")
    
    stats_data = []
    for batch_result in result.processing_results:
        total_area = sum(p.cutting_area * p.quantity for p in batch_result.batch.panels)
        used_sheets = len(batch_result.placement_results)
        cost_per_m2 = batch_result.total_cost / (total_area / 1_000_000) if total_area > 0 else 0
        
        stats_data.append({
            'Material': batch_result.batch.material_type,
            'Efficiency (%)': f"{batch_result.efficiency:.1%}",
            'Sheets Used': used_sheets,
            'Total Cost (¥)': f"¥{batch_result.total_cost:,.0f}",
            'Cost per m² (¥)': f"¥{cost_per_m2:,.0f}",
            'Algorithm': batch_result.algorithm_used,
            'Processing Time (s)': f"{batch_result.processing_time:.2f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

def create_efficiency_comparison(result: BatchOptimizationResult):
    """Create efficiency comparison bar chart"""
    materials = []
    efficiencies = []
    colors = []
    
    for batch_result in result.processing_results:
        materials.append(batch_result.batch.material_type)
        efficiencies.append(batch_result.efficiency * 100)
        colors.append('#4CAF50' if batch_result.efficiency >= 0.7 else '#FF9800')
    
    fig = go.Figure(data=[
        go.Bar(x=materials, y=efficiencies, marker_color=colors, text=[f"{e:.1f}%" for e in efficiencies], textposition='auto')
    ])
    
    fig.update_layout(
        title="Material Efficiency Comparison",
        xaxis_title="Material Type",
        yaxis_title="Efficiency (%)",
        height=400
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70% Target")
    
    return fig

def create_cost_breakdown(result: BatchOptimizationResult):
    """Create cost breakdown pie chart"""
    materials = []
    costs = []
    
    for batch_result in result.processing_results:
        materials.append(batch_result.batch.material_type)
        costs.append(batch_result.total_cost)
    
    fig = go.Figure(data=[
        go.Pie(labels=materials, values=costs, hole=0.4)
    ])
    
    fig.update_layout(
        title="Cost Distribution by Material",
        height=400
    )
    
    return fig

def create_algorithm_usage_chart(result: BatchOptimizationResult):
    """Create algorithm usage chart"""
    algorithms = {}
    
    for batch_result in result.processing_results:
        algo = batch_result.algorithm_used
        if algo not in algorithms:
            algorithms[algo] = 0
        algorithms[algo] += 1
    
    fig = go.Figure(data=[
        go.Pie(labels=list(algorithms.keys()), values=list(algorithms.values()), hole=0.4)
    ])
    
    fig.update_layout(
        title="Algorithm Usage Distribution",
        height=400
    )
    
    return fig

def render_export_options(result: BatchOptimizationResult):
    """Render export options"""
    st.markdown("### 📄 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Export Data")
        
        # JSON export
        if st.button("📁 Export Full Results (JSON)"):
            export_data = create_json_export(result)
            st.download_button(
                label="💾 Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"cutting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # CSV summary export
        if st.button("📈 Export Summary (CSV)"):
            csv_data = create_csv_summary(result)
            st.download_button(
                label="📁 Download CSV",
                data=csv_data,
                file_name=f"cutting_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("#### 🔧 Export Cutting Instructions")
        
        # Cutting instructions
        if st.button("📋 Generate Cutting Instructions"):
            instructions = create_cutting_instructions(result)
            st.download_button(
                label="📄 Download Instructions",
                data=instructions,
                file_name=f"cutting_instructions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Work order export
        if st.button("📝 Generate Work Order"):
            work_order = create_work_order(result)
            st.download_button(
                label="📋 Download Work Order",
                data=work_order,
                file_name=f"work_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def create_json_export(result: BatchOptimizationResult) -> Dict[str, Any]:
    """Create comprehensive JSON export"""
    return {
        "export_info": {
            "timestamp": datetime.now().isoformat(),
            "system_version": "2.0",
            "architecture": "Individual Material Processing"
        },
        "summary": {
            "overall_efficiency": result.overall_efficiency,
            "processing_time": result.total_processing_time,
            "high_efficiency_batches": len(result.high_efficiency_results),
            "low_efficiency_batches": len(result.low_efficiency_results)
        },
        "batch_results": [
            {
                "material": br.batch.material_type,
                "thickness": br.batch.thickness,
                "efficiency": br.efficiency,
                "algorithm": br.algorithm_used,
                "sheets_used": br.total_sheets_used,
                "total_cost": br.total_cost,
                "processing_time": br.processing_time
            }
            for br in result.processing_results
        ]
    }

def create_csv_summary(result: BatchOptimizationResult) -> str:
    """Create CSV summary export"""
    import io
    
    summary_data = []
    for br in result.processing_results:
        summary_data.append({
            'Material': br.batch.material_type,
            'Thickness_mm': br.batch.thickness,
            'Efficiency_%': f"{br.efficiency:.1%}",
            'Sheets_Used': br.total_sheets_used,
            'Total_Cost_JPY': br.total_cost,
            'Algorithm': br.algorithm_used,
            'Processing_Time_sec': br.processing_time
        })
    
    df = pd.DataFrame(summary_data)
    return df.to_csv(index=False)

def create_cutting_instructions(result: BatchOptimizationResult) -> str:
    """Create cutting instructions text"""
    instructions = []
    instructions.append("STEEL CUTTING OPTIMIZATION RESULTS")
    instructions.append("=" * 50)
    instructions.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    instructions.append(f"Overall Efficiency: {result.overall_efficiency:.1%}")
    instructions.append("")
    
    for batch_result in result.processing_results:
        instructions.append(f"MATERIAL: {batch_result.batch.material_type} ({batch_result.batch.thickness}mm)")
        instructions.append(f"Algorithm: {batch_result.algorithm_used}")
        instructions.append(f"Efficiency: {batch_result.efficiency:.1%}")
        instructions.append("-" * 30)
        
        for i, placement in enumerate(batch_result.placement_results):
            instructions.append(f"Sheet {i+1}:")
            instructions.append(f"  Size: {placement.sheet.width}x{placement.sheet.height}mm")
            instructions.append(f"  Panels: {len(placement.panels)}")
            
            for j, panel in enumerate(placement.panels):
                rotation = " (ROTATED)" if panel.rotated else ""
                instructions.append(f"    Panel {j+1}: {panel.panel.id} at ({panel.x:.0f},{panel.y:.0f}) "
                                 f"size {panel.actual_width:.0f}x{panel.actual_height:.0f}mm{rotation}")
        
        instructions.append("")
    
    return "\n".join(instructions)

def create_work_order(result: BatchOptimizationResult) -> str:
    """Create work order document"""
    work_order = []
    work_order.append("STEEL CUTTING WORK ORDER")
    work_order.append("=" * 40)
    work_order.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    work_order.append(f"Order ID: WO-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    work_order.append("")
    
    total_sheets = sum(len(br.placement_results) for br in result.processing_results)
    work_order.append(f"SUMMARY:")
    work_order.append(f"  Total Sheets: {total_sheets}")
    work_order.append(f"  Overall Efficiency: {result.overall_efficiency:.1%}")
    work_order.append("")
    
    work_order.append("MATERIALS REQUIRED:")
    for batch_result in result.processing_results:
        work_order.append(f"  - {batch_result.batch.material_type} {batch_result.batch.thickness}mm: {batch_result.total_sheets_used} sheets")
    
    work_order.append("")
    work_order.append("CUTTING SEQUENCE:")
    
    sheet_counter = 1
    for batch_result in result.processing_results:
        work_order.append(f"\n{batch_result.batch.material_type} Processing:")
        for placement in batch_result.placement_results:
            work_order.append(f"  Sheet {sheet_counter}: {len(placement.panels)} panels - {placement.efficiency:.1%} efficiency")
            sheet_counter += 1
    
    return "\n".join(work_order)

if __name__ == "__main__":
    main()