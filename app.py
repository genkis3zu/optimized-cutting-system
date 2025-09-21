"""
Steel Cutting Optimization System - Main Application
鋼板切断最適化システム - メインアプリケーション

A Streamlit-based application for optimizing steel panel cutting operations
with guillotine cut constraints.
"""

import streamlit as st
import logging
import time
from typing import List, Optional

# Import core modules
from core.models import Panel, SteelSheet, PlacementResult
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import create_ffd_algorithm
from ui.components import (
    PanelInputComponent, 
    SteelSheetComponent, 
    OptimizationSettingsComponent
)


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="鋼板切断最適化システム - Steel Cutting Optimizer",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render application header"""
    st.title("🔧 鋼板切断最適化システム")
    st.subheader("Steel Cutting Optimization System with Guillotine Constraints")
    
    st.markdown("""
    **システム概要 / System Overview:**
    - ギロチンカット制約下での2Dビンパッキング最適化
    - 2D bin packing optimization with guillotine cut constraints
    - 材料効率向上と作業時間短縮を実現 
    - Achieve material efficiency improvement and work time reduction
    """)


def create_visualization_placeholder(result: PlacementResult) -> str:
    """Create simple text visualization of placement result"""
    if not result or not result.panels:
        return "No panels placed"
    
    viz = f"📊 Cutting Plan Visualization\n\n"
    viz += f"Sheet: {result.sheet.width:.0f} × {result.sheet.height:.0f} mm\n"
    viz += f"Material: {result.material_block}\n"
    viz += f"Efficiency: {result.efficiency:.1%}\n\n"
    
    viz += "Placed Panels:\n"
    for i, placed_panel in enumerate(result.panels, 1):
        panel = placed_panel.panel
        viz += f"{i:2d}. {panel.id}: "
        viz += f"{placed_panel.actual_width:.0f}×{placed_panel.actual_height:.0f}mm "
        viz += f"at ({placed_panel.x:.0f}, {placed_panel.y:.0f}) "
        viz += f"{'[ROTATED]' if placed_panel.rotated else ''}\n"
    
    return viz


def render_results(results: List[PlacementResult]):
    """Render optimization results"""
    if not results:
        st.warning("最適化結果がありません / No optimization results")
        return
    
    st.success(f"✅ 最適化完了 / Optimization completed: {len(results)} sheet(s)")
    
    # Summary metrics
    total_panels = sum(len(result.panels) for result in results)
    avg_efficiency = sum(result.efficiency for result in results) / len(results)
    total_cost = sum(result.cost for result in results)
    total_time = sum(result.processing_time for result in results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("配置パネル数 / Placed Panels", total_panels)
    with col2:
        st.metric("平均効率 / Average Efficiency", f"{avg_efficiency:.1%}")
    with col3:
        st.metric("総コスト / Total Cost", f"¥{total_cost:,.0f}")
    with col4:
        st.metric("処理時間 / Processing Time", f"{total_time:.2f}s")
    
    # Individual results
    for i, result in enumerate(results, 1):
        with st.expander(f"Sheet {i}: {result.material_block} - {result.efficiency:.1%} efficiency"):
            col_info, col_viz = st.columns([1, 2])
            
            with col_info:
                st.write("**詳細情報 / Details:**")
                st.write(f"- アルゴリズム / Algorithm: {result.algorithm}")
                st.write(f"- 配置パネル数 / Placed Panels: {len(result.panels)}")
                st.write(f"- 効率 / Efficiency: {result.efficiency:.1%}")
                st.write(f"- 無駄面積 / Waste Area: {result.waste_area:,.0f} mm²")
                st.write(f"- 切断長 / Cut Length: {result.cut_length:,.0f} mm")
                st.write(f"- 処理時間 / Time: {result.processing_time:.3f}s")
            
            with col_viz:
                st.write("**配置図 / Layout:**")
                visualization = create_visualization_placeholder(result)
                st.text(visualization)


def run_optimization(panels: List[Panel], sheet: SteelSheet, algorithm: str, constraints):
    """Run optimization with progress tracking"""
    if not panels:
        st.error("パネルが入力されていません / No panels provided")
        return []
    
    # Create and configure optimization engine
    engine = create_optimization_engine()
    
    # Register FFD algorithm
    ffd_algorithm = create_ffd_algorithm()
    engine.register_algorithm(ffd_algorithm)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("最適化を開始しています... / Starting optimization...")
        progress_bar.progress(10)
        
        # Run optimization
        start_time = time.time()
        
        algorithm_hint = None if algorithm == 'AUTO' else algorithm
        results = engine.optimize(
            panels=panels,
            constraints=constraints,
            algorithm_hint=algorithm_hint
        )
        
        processing_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text(f"最適化完了 / Optimization completed in {processing_time:.2f}s")
        
        return results
    
    except Exception as e:
        st.error(f"最適化エラー / Optimization error: {str(e)}")
        return []
    
    finally:
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()


def main():
    """Main application function"""
    setup_logging()
    setup_page_config()
    render_header()
    
    # Sidebar for input
    with st.sidebar:
        st.header("入力設定 / Input Settings")
        
        # Panel input component
        panel_component = PanelInputComponent()
        panels = panel_component.render()
        
        st.divider()
        
        # Steel sheet component
        sheet_component = SteelSheetComponent()
        sheet = sheet_component.render()
        
        st.divider()
        
        # Optimization settings
        settings_component = OptimizationSettingsComponent()
        algorithm, constraints = settings_component.render()
        
        st.divider()
        
        # Optimization button
        optimize_button = st.button(
            "🚀 最適化実行 / Run Optimization",
            type="primary",
            disabled=len(panels) == 0,
            use_container_width=True
        )
    
    # Main content area
    if optimize_button and panels:
        st.header("最適化結果 / Optimization Results")
        
        with st.spinner("最適化を実行中... / Running optimization..."):
            results = run_optimization(panels, sheet, algorithm, constraints)
        
        render_results(results)
        
        # Store results in session state for export
        if results:
            st.session_state.optimization_results = results
    
    elif not panels:
        # Show welcome message and instructions
        st.info("""
        ### 使用方法 / How to Use
        
        1. **パネル入力 / Panel Input**: 左側のサイドバーでパネル情報を入力
           - 手動入力、テキストデータ、またはファイルアップロード
           - Manual input, text data, or file upload
        
        2. **鋼板設定 / Steel Sheet Settings**: 母材の寸法と仕様を設定
           - Configure dimensions and specifications
        
        3. **最適化設定 / Optimization Settings**: アルゴリズムと制約条件を選択
           - Select algorithm and constraint conditions
        
        4. **実行 / Execute**: 最適化を実行して結果を確認
           - Run optimization and view results
        
        ### サンプルデータ / Sample Data
        
        以下のサンプルデータを試してみてください:
        """)
        
        sample_data = """panel1,300,200,2,SS400,6.0
panel2,400,300,1,SS400,6.0
panel3,250,150,3,SS400,6.0"""
        
        st.code(sample_data, language="csv")
        
        if st.button("サンプルデータを読み込み / Load Sample Data"):
            # Parse sample data and add to session
            from core.text_parser import parse_text_data
            result = parse_text_data(sample_data, 'csv')
            if result.panels:
                st.session_state.panels = result.panels
                st.success(f"サンプルデータを読み込みました / Loaded sample data: {len(result.panels)} panels")
                st.rerun()
    
    else:
        st.info("パネルを入力して最適化を実行してください / Please input panels and run optimization")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    鋼板切断最適化システム v1.0 | Steel Cutting Optimization System v1.0<br>
    Developed with Streamlit and Python | ギロチンカット制約対応
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()