"""
Steel Cutting Optimization System - Main Application
鋼板切断最適化システム - メインアプリケーション

A Streamlit-based application for optimizing steel panel cutting operations
with guillotine cut constraints.
"""

import streamlit as st
import logging
import time
import pandas as pd
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


def render_enhanced_results(results: List[PlacementResult]):
    """Render enhanced optimization results with visualization"""
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

    # Interactive visualization
    from ui.visualizer import render_cutting_visualization
    render_cutting_visualization(results)

    # Export options
    st.subheader("📤 エクスポート / Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 作業指示書生成 / Generate Work Instructions"):
            generate_work_instructions(results)

    with col2:
        if st.button("📊 レポート出力 / Export Report"):
            export_optimization_report(results)

    with col3:
        if st.button("💾 結果保存 / Save Results"):
            save_optimization_results(results)


def generate_work_instructions(results: List[PlacementResult]):
    """Generate work instructions for cutting"""
    try:
        from cutting.instruction import WorkInstructionGenerator
        from cutting.sequence import CuttingSequenceOptimizer

        generator = WorkInstructionGenerator()
        optimizer = CuttingSequenceOptimizer()

        with st.spinner("作業指示書を生成中... / Generating work instructions..."):
            for i, result in enumerate(results, 1):
                # Optimize cutting sequence
                optimized_sequence = optimizer.optimize_sequence(
                    result.panels,
                    result.sheet,
                    strategy="efficiency_first"
                )

                # Generate work instruction
                work_instruction = generator.generate_work_instruction(
                    sheet_id=f"SHEET_{i:03d}",
                    placed_panels=optimized_sequence,
                    sheet_specs=result.sheet,
                    constraints={
                        'kerf_width': 0.0,
                        'material_type': result.material_block
                    }
                )

                st.success(f"✅ Sheet {i} 作業指示書生成完了")

                # Display key information
                with st.expander(f"📋 Sheet {i} 作業指示書詳細"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**基本情報:**")
                        st.write(f"- シートID: {work_instruction.sheet_id}")
                        st.write(f"- 材質: {work_instruction.material_type}")
                        st.write(f"- ステップ数: {work_instruction.total_steps}")
                        st.write(f"- 予想時間: {work_instruction.estimated_total_time:.1f}分")

                    with col2:
                        st.write("**品質情報:**")
                        st.write(f"- 複雑度: {work_instruction.complexity_score:.2f}")
                        st.write(f"- 切断長: {work_instruction.total_cut_length:.0f}mm")
                        st.write(f"- 安全注意: {len(work_instruction.safety_notes)}項目")

    except Exception as e:
        st.error(f"作業指示書生成エラー: {str(e)}")


def export_optimization_report(results: List[PlacementResult]):
    """Export optimization report"""
    try:
        from cutting.export import DocumentExporter
        import tempfile
        import os

        exporter = DocumentExporter()

        with st.spinner("レポートを出力中... / Exporting report..."):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Export efficiency report
                success = exporter.export_efficiency_report_excel(
                    results=results,
                    file_path=tmp_file.name.replace('.pdf', '.xlsx'),
                    include_charts=True
                )

                if success:
                    st.success("✅ レポート出力完了")

                    # Provide download button
                    with open(tmp_file.name.replace('.pdf', '.xlsx'), 'rb') as f:
                        st.download_button(
                            "📁 レポートダウンロード / Download Report",
                            data=f.read(),
                            file_name=f"optimization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # Clean up
                    os.unlink(tmp_file.name.replace('.pdf', '.xlsx'))
                else:
                    st.error("レポート出力に失敗しました")

    except Exception as e:
        st.error(f"レポート出力エラー: {str(e)}")


def save_optimization_results(results: List[PlacementResult]):
    """Save optimization results to session storage"""
    try:
        import json
        from datetime import datetime

        # Prepare data for storage
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_sheets': len(results),
            'summary': {
                'total_panels': sum(len(r.panels) for r in results),
                'average_efficiency': sum(r.efficiency for r in results) / len(results),
                'total_cost': sum(r.cost for r in results),
                'total_time': sum(r.processing_time for r in results)
            },
            'sheets': []
        }

        for i, result in enumerate(results, 1):
            sheet_data = {
                'sheet_id': f"SHEET_{i:03d}",
                'material': result.material_block,
                'algorithm': result.algorithm,
                'efficiency': result.efficiency,
                'panels_count': len(result.panels),
                'waste_area': result.waste_area,
                'cut_length': result.cut_length,
                'processing_time': result.processing_time,
                'panels': [
                    {
                        'id': p.panel.id,
                        'x': p.x,
                        'y': p.y,
                        'width': p.actual_width,
                        'height': p.actual_height,
                        'rotated': p.rotated
                    }
                    for p in result.panels
                ]
            }
            results_data['sheets'].append(sheet_data)

        # Store in session state
        st.session_state.saved_results = results_data

        st.success("✅ 結果を保存しました / Results saved successfully")

        # Show summary
        with st.expander("保存された結果サマリー / Saved Results Summary"):
            st.json(results_data['summary'])

    except Exception as e:
        st.error(f"結果保存エラー: {str(e)}")


def render_results(results: List[PlacementResult]):
    """Legacy render results (kept for compatibility)"""
    render_enhanced_results(results)


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

    # Sidebar for navigation
    with st.sidebar:
        st.header("ナビゲーション / Navigation")

        page = st.radio(
            "ページ選択 / Select Page",
            ['optimization', 'material_management', 'data_management'],
            format_func=lambda x: {
                'optimization': '🔧 最適化 / Optimization',
                'material_management': '📦 材料管理 / Material Management',
                'data_management': '💾 データ管理 / Data Management'
            }[x]
        )

        st.divider()

    if page == 'optimization':
        render_optimization_page()
    elif page == 'material_management':
        render_material_management_page()
    elif page == 'data_management':
        render_data_management_page()


def render_optimization_page():
    """Render optimization page"""
    render_header()

    # Sidebar for input
    with st.sidebar:
        st.header("入力設定 / Input Settings")
        
        # Panel input component with material validation
        from core.persistence_adapter import get_persistence_adapter
        from core.material_manager import get_material_manager

        # Use persistence adapter for database-first approach
        persistence = get_persistence_adapter()
        material_manager = get_material_manager()  # Fallback for compatibility

        # Auto-load sample data if empty
        materials = persistence.get_materials()
        if len(materials) == 0:
            st.info("材料在庫が空です。サンプルデータを読み込み中...")
            sample_file = "sample_data/sizaidata.txt"
            if os.path.exists(sample_file):
                added_count = material_manager.load_from_sample_data(sample_file)
                if added_count > 0:
                    st.success(f"{added_count}個の材料を読み込みました")

        panel_component = PanelInputComponent()
        panels = panel_component.render()

        # Material validation for panels
        if panels:
            st.write("### 材料検証 / Material Validation")
            validation_issues = []
            for panel in panels:
                is_valid, message = material_manager.validate_panel_against_inventory(
                    panel.material, panel.thickness, panel.width, panel.height
                )
                if not is_valid:
                    validation_issues.append(f"⚠️ Panel {panel.id}: {message}")

            if validation_issues:
                with st.expander("⚠️ 材料検証エラー / Material Validation Issues"):
                    for issue in validation_issues:
                        st.warning(issue)
                    st.info("材料管理ページで在庫を確認・追加してください")
            else:
                st.success("✅ すべてのパネルで材料検証が通りました")

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

    # Main content area - show panel details if requested
    if hasattr(st.session_state, 'show_panel_details') and st.session_state.show_panel_details and panels:
        from ui.visualizer import render_panel_details
        render_panel_details(panels, show_validation=True)

        if st.button("パネル詳細を閉じる / Close Panel Details"):
            st.session_state.show_panel_details = False
            st.rerun()

        st.divider()

    # Optimization execution and results
    if optimize_button and panels:
        st.header("🚀 最適化結果 / Optimization Results")

        with st.spinner("最適化を実行中... / Running optimization..."):
            results = run_optimization(panels, sheet, algorithm, constraints)

        if results:
            # Enhanced results display
            render_enhanced_results(results)

            # Store results in session state for export
            st.session_state.optimization_results = results

            # Save optimization result to database
            try:
                from core.persistence_adapter import get_persistence_adapter
                persistence = get_persistence_adapter()

                # Calculate metrics
                total_panels = sum(panel.quantity for panel in panels)
                placed_panels = sum(len(result.panels) for result in results)
                placement_rate = (placed_panels / total_panels) * 100 if total_panels > 0 else 0
                avg_efficiency = sum(result.efficiency for result in results) / len(results) if results else 0

                metrics = {
                    'total_panels': total_panels,
                    'placed_panels': placed_panels,
                    'placement_rate': placement_rate,
                    'efficiency': avg_efficiency,
                    'sheets_used': len(results)
                }

                # Save to database
                optimization_id = persistence.save_optimization_result(
                    panels=panels,
                    constraints=constraints,
                    results=[{
                        'id': i,
                        'efficiency': result.efficiency,
                        'panels_count': len(result.panels),
                        'material_block': result.material_block
                    } for i, result in enumerate(results)],
                    algorithm_used=algorithm.name,
                    processing_time=getattr(results[0], 'processing_time', 0.0) if results else 0.0,
                    metrics=metrics
                )

                if optimization_id:
                    st.success(f"✅ 最適化結果をデータベースに保存しました (ID: {optimization_id})")

            except Exception as e:
                st.warning(f"⚠️ データベース保存でエラーが発生しました: {e}")
                # Continue without failing the optimization
        else:
            st.error("最適化に失敗しました / Optimization failed")

    elif not panels:
        # Show welcome message and instructions
        st.info("""
        ### 使用方法 / How to Use

        1. **材料管理 / Material Management**: まず材料在庫を設定してください
           - 材料管理ページでサンプルデータを読み込み
           - Setup material inventory first

        2. **パネル入力 / Panel Input**: パネル情報を入力
           - サンプルデータまたは手動入力
           - Sample data or manual input

        3. **最適化実行 / Run Optimization**: 最適化を実行して結果を確認
           - Execute optimization and view results

        ### 実際のデータ対応 / Real Data Support

        本システムは実際の製造データ形式に対応しています:
        - data0923.txt (切断データ)
        - sizaidata.txt (材料在庫データ)
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 材料管理ページへ / Go to Material Management"):
                st.session_state.page_redirect = 'material_management'
        with col2:
            if st.button("💾 データ管理ページへ / Go to Data Management"):
                st.session_state.page_redirect = 'data_management'
                st.rerun()

    else:
        st.info("パネルを入力して最適化を実行してください / Please input panels and run optimization")


def render_material_management_page():
    """Render material management page"""
    from ui.material_management_ui import render_material_management
    render_material_management()


def render_cutting_optimization():
    """Render cutting optimization page"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cutting_opt", "pages/1_🔧_Cutting_Optimization.py")
    cutting_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cutting_module)
    cutting_module.main()

def render_analysis_results():
    """Render analysis results page"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("analysis", "pages/4_📊_Analysis_Results.py")
    analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis_module)
    analysis_module.main()

def render_pi_management():
    """Render PI management page"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pi_mgmt", "pages/3_⚙️_PI_Management.py")
    pi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi_module)
    pi_module.main()

def render_data_management_page():
    """Render data management page"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_management", "pages/3_💾_Data_Management.py")
    data_management_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_management_module)
    data_management_module.render_data_management_page()


# Import os for file operations
import os


if __name__ == "__main__":
    main()