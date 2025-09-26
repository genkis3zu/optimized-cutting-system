#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steel Cutting Optimization System - Streamlit App
鋼板切断最適化システム - Streamlit アプリ

Modern Material-UI inspired interface with new architecture integration
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.session_manager import initialize_session_state
from components.sidebar import render_sidebar

# Page configuration with Material Design inspiration
st.set_page_config(
    page_title="Steel Cutting Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        ## Steel Cutting Optimization System
        
        **Version 2.0** - Complete Redesign
        
        ### Key Features:
        - ✅ Simple Mathematical Optimization (968x712 → 4 panels)
        - ✅ Individual Material Processing (No mixing)
        - ✅ Dynamic Efficiency Evaluation
        - ✅ Real-time Visual Results
        
        ### Architecture:
        - **Phase 1**: Individual Material Processing
        - **Phase 2**: Efficiency Evaluation  
        - **Phase 3**: Residual Optimization (Future)
        
        Built with modern Streamlit best practices and Material Design principles.
        """
    }
)

# Custom CSS for Material Design inspired styling
def load_custom_css():
    """Load custom CSS for Material Design styling"""
    st.markdown("""
    <style>
    /* Material Design Color Palette */
    :root {
        --primary-color: #2196F3;
        --primary-variant: #1976D2;
        --secondary-color: #FF9800;
        --background: #FAFAFA;
        --surface: #FFFFFF;
        --error: #F44336;
        --success: #4CAF50;
        --warning: #FF9800;
        --on-primary: #FFFFFF;
        --on-background: #212121;
        --on-surface: #212121;
    }
    
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-variant));
        color: var(--on-primary);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(33, 150, 243, 0.2);
    }
    
    /* Card-like containers */
    .optimization-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
    }
    
    /* Status indicators */
    .status-success { 
        color: var(--success); 
        font-weight: 600; 
    }
    .status-warning { 
        color: var(--warning); 
        font-weight: 600; 
    }
    .status-error { 
        color: var(--error); 
        font-weight: 600; 
    }
    
    /* Metrics styling */
    .metric-container {
        background: var(--surface);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Hide Streamlit default styling that conflicts */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom styling
    load_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 2.5rem;">🔧 Steel Cutting Optimizer</h1>
        <p style="margin:0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Intelligent material-based optimization with simple math priority
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message and system status
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### Welcome to the Redesigned System")
        st.markdown("""
        This version prioritizes **simple mathematical calculations** for straightforward cases 
        like the 968×712 panels (4 pieces per sheet), while using complex algorithms only when necessary.
        
        **Key Improvements:**
        - ✅ Individual material processing (no mixing)
        - ✅ Simple math for obvious cases  
        - ✅ Dynamic efficiency thresholds
        - ✅ Clean legacy-free architecture
        """)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🎯 Optimization Strategy", 
            value="Smart Selection",
            help="Automatically chooses between simple math and complex algorithms"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="🏭 Material Handling", 
            value="Individual",
            help="Each material type processed separately (no mixing)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation instructions
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        #### 1️⃣ Panel Input
        Navigate to **Panel Input** page to:
        - Add panels manually or import from CSV
        - Set material types and thicknesses
        - Configure cutting parameters
        """)
    
    with nav_col2:
        st.markdown("""
        #### 2️⃣ Optimization
        Go to **Optimization** page to:
        - Review material batches
        - Run optimization with strategy selection
        - Monitor processing in real-time
        """)
        
    with nav_col3:
        st.markdown("""
        #### 3️⃣ Results
        View **Results** page for:
        - Interactive cutting plan visualization
        - Efficiency analysis by material
        - Export cutting instructions
        """)
    
    # Demo section
    if st.button("🚀 Run 968×712 Demo", type="primary", help="Test the user's example case"):
        run_demo()

def run_demo():
    """Run the 968x712 demo case"""
    with st.spinner("Running 968×712 demo optimization..."):
        try:
            # Import and run the demo
            from models.core_models import Panel
            from processing.phase1_individual import IndividualMaterialProcessor
            
            # Create the user's test case
            demo_panel = Panel(
                id="demo_968x712",
                width=968,
                height=712,
                quantity=12,
                material="SGCC", 
                thickness=6.0
            )
            
            # Process using new system
            processor = IndividualMaterialProcessor()
            result = processor.process_all_materials([demo_panel])
            
            # Display results
            all_batches = result.high_efficiency_results + result.low_efficiency_results
            if all_batches:
                batch = all_batches[0]
                
                st.success("✅ Demo completed successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Algorithm", batch.algorithm_used)
                
                with col2:  
                    st.metric("Efficiency", f"{batch.efficiency:.1%}")
                
                with col3:
                    st.metric("Sheets Used", batch.total_sheets_used)
                
                with col4:
                    panels_per_sheet = len(batch.placement_results[0].panels) if batch.placement_results else 0
                    st.metric("Panels/Sheet", panels_per_sheet)
                
                if batch.placement_results:
                    st.markdown("**First Sheet Layout:**")
                    for i, panel in enumerate(batch.placement_results[0].panels[:4]):  # Show first 4
                        st.text(f"Panel {i+1}: position ({panel.x:.0f}, {panel.y:.0f}) size {panel.actual_width:.0f}×{panel.actual_height:.0f}mm")
                
                if panels_per_sheet == 4:
                    st.success("🎯 **Perfect!** Achieved expected 4 panels per sheet as user requested")
                else:
                    st.warning(f"⚠️ Got {panels_per_sheet} panels per sheet, expected 4")
                    
        except Exception as e:
            st.error(f"Demo failed: {str(e)}")
            st.info("Please ensure all required modules are available.")

if __name__ == "__main__":
    main()