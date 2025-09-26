#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidebar Component for Steel Cutting Optimizer
鋼板切断最適化システムのサイドバーコンポーネント
"""

import streamlit as st
from utils.session_manager import (
    get_material_summary, 
    get_optimization_constraints,
    update_optimization_constraints,
    get_cached_material_options,
    get_cached_thickness_options
)

def render_sidebar():
    """
    Render the main sidebar with system settings and status
    システム設定とステータス表示のメインサイドバーをレンダリング
    """
    
    with st.sidebar:
        # Header
        st.markdown("### ⚙️ System Settings")
        st.markdown("---")
        
        # Current panel summary
        render_panel_summary()
        
        st.markdown("---")
        
        # Optimization settings
        render_optimization_settings()
        
        st.markdown("---")
        
        # Sheet templates
        render_sheet_templates()
        
        st.markdown("---")
        
        # Quick actions
        render_quick_actions()

def render_panel_summary():
    """Render current panels summary"""
    st.markdown("#### 📊 Current Panels")
    
    if not st.session_state.panels:
        st.info("No panels added yet")
        return
    
    summary = get_material_summary()
    
    # Total summary
    total_panels = len(st.session_state.panels)
    total_quantity = sum(p.quantity for p in st.session_state.panels)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Panel Types", total_panels)
    with col2:
        st.metric("Total Pieces", total_quantity)
    
    # Material breakdown
    st.markdown("**By Material:**")
    for material, data in summary.items():
        with st.expander(f"{material} ({data['panel_count']} types)"):
            st.metric("Pieces", data['total_quantity'])
            st.metric("Area", f"{data['total_area']:,.0f} mm²")
            st.metric("Thickness", f"{data['thickness']} mm")

def render_optimization_settings():
    """Render optimization settings"""
    st.markdown("#### 🎯 Optimization Settings")
    
    constraints = get_optimization_constraints()
    
    # Key settings
    kerf_width = st.slider(
        "Kerf Width (mm)",
        min_value=1.0,
        max_value=10.0,
        value=constraints.kerf_width,
        step=0.5,
        help="Width of cutting blade"
    )
    
    safety_margin = st.slider(
        "Safety Margin (mm)", 
        min_value=0.0,
        max_value=20.0,
        value=constraints.safety_margin,
        step=1.0,
        help="Additional margin for safe cutting"
    )
    
    min_efficiency = st.slider(
        "Min Efficiency Threshold",
        min_value=0.4,
        max_value=0.9,
        value=constraints.min_efficiency_threshold,
        step=0.05,
        format="%.0f%%",
        help="Minimum efficiency for high-quality results"
    )
    
    # Advanced settings toggle
    show_advanced = st.checkbox("Show Advanced Settings", 
                               value=st.session_state.show_advanced_settings)
    st.session_state.show_advanced_settings = show_advanced
    
    if show_advanced:
        max_time = st.number_input(
            "Max Processing Time (sec)",
            min_value=10,
            max_value=300,
            value=int(constraints.max_processing_time),
            help="Maximum time for optimization"
        )
        
        max_sheets = st.number_input(
            "Max Sheets per Material",
            min_value=1,
            max_value=50,
            value=constraints.max_sheets_per_material,
            help="Limit sheets per material type"
        )
    else:
        max_time = constraints.max_processing_time
        max_sheets = constraints.max_sheets_per_material
    
    # Update constraints if changed
    update_optimization_constraints(
        kerf_width=kerf_width,
        safety_margin=safety_margin,
        min_efficiency_threshold=min_efficiency,
        max_processing_time=max_time,
        max_sheets_per_material=max_sheets
    )

def render_sheet_templates():
    """Render sheet template settings"""
    st.markdown("#### 📐 Sheet Templates")
    
    # Show current templates
    templates = st.session_state.sheet_templates
    
    selected_material = st.selectbox(
        "Material to Configure",
        options=list(templates.keys()),
        help="Select material to view/edit sheet template"
    )
    
    if selected_material:
        template = templates[selected_material]
        
        with st.expander(f"{selected_material} Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_width = st.number_input(
                    "Width (mm)",
                    min_value=500,
                    max_value=3000,
                    value=int(template.width),
                    key=f"width_{selected_material}"
                )
                
                new_thickness = st.selectbox(
                    "Thickness (mm)",
                    options=get_cached_thickness_options(),
                    index=get_cached_thickness_options().index(template.thickness) if template.thickness in get_cached_thickness_options() else 5,
                    key=f"thickness_{selected_material}"
                )
            
            with col2:
                new_height = st.number_input(
                    "Height (mm)",
                    min_value=1000,
                    max_value=6000,
                    value=int(template.height),
                    key=f"height_{selected_material}"
                )
                
                new_cost = st.number_input(
                    "Cost per Sheet (¥)",
                    min_value=1000,
                    max_value=50000,
                    value=int(template.cost_per_sheet),
                    step=1000,
                    key=f"cost_{selected_material}"
                )
            
            # Update template if Apply button pressed
            if st.button(f"Apply Changes to {selected_material}", key=f"apply_{selected_material}"):
                from models.core_models import SteelSheet
                
                st.session_state.sheet_templates[selected_material] = SteelSheet(
                    width=new_width,
                    height=new_height,
                    thickness=new_thickness,
                    material=selected_material,
                    cost_per_sheet=new_cost
                )
                st.success(f"Updated {selected_material} template")
                st.experimental_rerun()

def render_quick_actions():
    """Render quick action buttons"""
    st.markdown("#### ⚡ Quick Actions")
    
    # Demo button
    if st.button("🚀 Run 968×712 Demo", help="Test the system with user's example case"):
        st.session_state.demo_running = True
        st.info("Demo started! Check main page for results.")
    
    # Clear all panels
    if st.button("🗑️ Clear All Panels", help="Remove all panels from workspace"):
        from utils.session_manager import clear_all_panels
        clear_all_panels()
        st.success("All panels cleared!")
        st.experimental_rerun()
    
    # Export data
    if st.button("💾 Export Session", help="Download current session data"):
        from utils.session_manager import export_session_data
        import json
        
        data = export_session_data()
        st.download_button(
            label="📁 Download JSON",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name="cutting_optimizer_session.json",
            mime="application/json"
        )
    
    # Processing history
    if st.session_state.processing_history:
        st.markdown("#### 📈 Recent Optimizations")
        
        for i, entry in enumerate(reversed(st.session_state.processing_history[-3:])):  # Show last 3
            with st.expander(f"Run {len(st.session_state.processing_history) - i}"):
                st.metric("Panels", entry['panel_count'])
                st.metric("Efficiency", f"{entry['efficiency']:.1%}")
                st.metric("Processing Time", f"{entry['processing_time']:.2f}s")
    
    # System status
    st.markdown("---")
    st.markdown("#### 🔧 System Status")
    
    # Algorithm status
    algorithm_status = "✅ SimpleMath + Complex"
    st.success(f"**Algorithms:** {algorithm_status}")
    
    # Material mixing status  
    mixing_status = "🚫 Disabled (Individual)"
    st.info(f"**Material Mixing:** {mixing_status}")
    
    # Architecture status
    architecture_status = "✅ Legacy-Free"
    st.success(f"**Architecture:** {architecture_status}")

def render_navigation_helper():
    """Render navigation helper in sidebar"""
    st.markdown("---")
    st.markdown("#### 🧭 Navigation")
    
    pages = [
        ("🏠 Home", "Overview and demo"),
        ("🔧 Panel Input", "Add and manage panels"),
        ("🎯 Optimization", "Run optimization process"),
        ("📊 Results", "View cutting plans")
    ]
    
    for page_name, description in pages:
        st.markdown(f"**{page_name}**")
        st.caption(description)