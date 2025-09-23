"""
Cutting Optimization Page
切断最適化ページ

Streamlit page for steel cutting optimization with integrated panel input
"""

import streamlit as st
import logging
import time
import pandas as pd
from typing import List, Optional
import os

# Import core modules
from core.models import Panel, SteelSheet, PlacementResult
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import create_ffd_algorithm
from ui.components import PanelInputComponent, OptimizationSettingsComponent
from core.material_manager import get_material_manager


def setup_page():
    """Setup page configuration and styling"""
    st.set_page_config(
        page_title="Cutting Optimization - Steel Cutting System",
        page_icon="🔧",
        layout="wide"
    )

    # Enhanced CSS for optimization page
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #ff7f0e, #d62728);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .optimization-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff7f0e;
        margin-bottom: 1.5rem;
    }
    .panel-input-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .results-section {
        background: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #87ceeb;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header():
    """Render enhanced page header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔧 鋼板切断最適化システム</h1>
        <h3>Steel Cutting Optimization System</h3>
        <p>ギロチンカット制約下での2Dビンパッキング最適化 | 2D bin packing optimization with guillotine constraints</p>
    </div>
    """, unsafe_allow_html=True)


def render_panel_input_section():
    """Render panel input section in main content area"""
    st.markdown("""
    <div class="panel-input-section">
        <h2>📋 パネル入力 / Panel Input</h2>
        <p>切断するパネルの情報を入力してください。複数の入力方法に対応しています。</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize panel input component
    panel_component = PanelInputComponent()
    panels = panel_component.render()

    # Material validation in main area
    if panels:
        material_manager = get_material_manager()

        # Enhanced validation display
        st.markdown("### 🔍 材料検証 / Material Validation")

        validation_issues = []
        validation_success = []

        for panel in panels:
            is_valid, message = material_manager.validate_panel_against_inventory(
                panel.material, panel.thickness, panel.width, panel.height
            )
            if not is_valid:
                validation_issues.append(f"⚠️ Panel {panel.id}: {message}")
            else:
                validation_success.append(f"✅ Panel {panel.id}: {message}")

        if validation_issues:
            st.markdown("""
            <div class="warning-message">
                <h4>⚠️ 材料検証エラー / Material Validation Issues</h4>
            </div>
            """, unsafe_allow_html=True)

            for issue in validation_issues:
                st.warning(issue)

            st.info("💡 材料管理ページで在庫を確認・追加してください / Please check inventory in Material Management page")

        if validation_success:
            with st.expander("✅ 検証成功 / Validation Success", expanded=False):
                for success in validation_success:
                    st.success(success)

        # Enhanced panel summary
        col1, col2, col3, col4 = st.columns(4)

        total_panels = len(panels)
        total_quantity = sum(p.quantity for p in panels)
        total_area = sum(p.area * p.quantity for p in panels)
        materials = set(p.material for p in panels)

        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <h4 style="color: #1f77b4; margin: 0;">パネル種類</h4>
                <h2 style="margin: 0;">{total_panels}</h2>
                <p style="margin: 0; color: #666;">Panel Types</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <h4 style="color: #2ca02c; margin: 0;">総数量</h4>
                <h2 style="margin: 0;">{total_quantity}</h2>
                <p style="margin: 0; color: #666;">Total Quantity</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <h4 style="color: #ff7f0e; margin: 0;">総面積</h4>
                <h2 style="margin: 0;">{total_area:,.0f}</h2>
                <p style="margin: 0; color: #666;">Total Area (mm²)</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-item">
                <h4 style="color: #d62728; margin: 0;">材質種類</h4>
                <h2 style="margin: 0;">{len(materials)}</h2>
                <p style="margin: 0; color: #666;">Material Types</p>
            </div>
            """, unsafe_allow_html=True)

        # Panel details table
        if st.checkbox("📊 パネル詳細を表示 / Show Panel Details", value=False):
            render_panel_details_table(panels)

    return panels


def render_panel_details_table(panels: List[Panel]):
    """Render detailed panel information table"""
    st.subheader("📋 パネル詳細情報 / Panel Details")

    # Create enhanced panel data
    panel_data = []
    for i, panel in enumerate(panels):
        panel_data.append({
            'No.': i + 1,
            'パネルID / Panel ID': panel.id,
            '幅 / Width (mm)': f"{panel.width:.0f}",
            '高さ / Height (mm)': f"{panel.height:.0f}",
            '数量 / Quantity': panel.quantity,
            '材質 / Material': panel.material,
            '板厚 / Thickness (mm)': f"{panel.thickness:.1f}",
            '面積 / Area (mm²)': f"{panel.area:,.0f}",
            '総面積 / Total Area (mm²)': f"{panel.area * panel.quantity:,.0f}",
            '回転許可 / Rotation': '○' if panel.allow_rotation else '×',
            '優先度 / Priority': panel.priority
        })

    df = pd.DataFrame(panel_data)

    # Enhanced dataframe display
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Material breakdown
    material_summary = {}
    for panel in panels:
        if panel.material not in material_summary:
            material_summary[panel.material] = {
                'count': 0,
                'quantity': 0,
                'total_area': 0
            }
        material_summary[panel.material]['count'] += 1
        material_summary[panel.material]['quantity'] += panel.quantity
        material_summary[panel.material]['total_area'] += panel.area * panel.quantity

    # Display material breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 材質別サマリー / Material Summary")
        summary_data = []
        for material, data in material_summary.items():
            summary_data.append({
                '材質 / Material': material,
                'パネル種類 / Types': data['count'],
                '総数量 / Total Qty': data['quantity'],
                '総面積 / Total Area (mm²)': f"{data['total_area']:,.0f}"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col2:
        # Material pie chart
        import plotly.express as px
        if material_summary:
            fig = px.pie(
                values=list(data['total_area'] for data in material_summary.values()),
                names=list(material_summary.keys()),
                title="材質別面積分布 / Area Distribution by Material"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)


def render_optimization_settings():
    """Render optimization settings section"""
    st.markdown("""
    <div class="optimization-card">
        <h3>⚙️ 最適化設定 / Optimization Settings</h3>
        <p>最適化アルゴリズムとパラメータを設定してください。</p>
    </div>
    """, unsafe_allow_html=True)

    # Optimization settings component
    settings_component = OptimizationSettingsComponent()
    algorithm, constraints = settings_component.render()

    return algorithm, constraints


def run_optimization(panels: List[Panel], algorithm: str, constraints):
    """Run optimization with enhanced progress tracking"""
    if not panels:
        st.error("パネルが入力されていません / No panels provided")
        return []

    # Create optimization engine exactly like test_all_placement.py (67.7% placement rate)
    from core.optimizer import OptimizationEngine
    from core.algorithms.ffd import FirstFitDecreasing
    from core.algorithms.bfd import BestFitDecreasing
    from core.algorithms.genetic import GeneticAlgorithm

    engine = OptimizationEngine()  # Empty engine like test_all_placement.py

    # Register algorithms including Genetic Algorithm option
    ffd_algorithm = FirstFitDecreasing()
    bfd_algorithm = BestFitDecreasing()
    genetic_algorithm = GeneticAlgorithm(
        population_size=50,    # Larger population for better exploration
        generations=999999,    # Effectively unlimited generations (will stop at 100% or convergence)
        mutation_rate=0.2,     # Higher mutation for better exploration
        crossover_rate=0.85    # High crossover for exploitation
    )

    engine.register_algorithm(ffd_algorithm)
    engine.register_algorithm(bfd_algorithm)
    engine.register_algorithm(genetic_algorithm)

    # Enhanced progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_display = st.empty()

    try:
        status_text.markdown("**🚀 最適化を開始しています... / Starting optimization...**")
        progress_bar.progress(10)

        # Run optimization
        start_time = time.time()

        # Extract algorithm from UI selection and allow user choice
        if 'FFD' in algorithm:
            algorithm_hint = 'FFD'
        elif 'BFD' in algorithm:
            algorithm_hint = 'BFD'  # Allow BFD but warn about potential issues
        elif 'GA' in algorithm or 'Genetic' in algorithm:
            algorithm_hint = 'GA'   # Allow Genetic Algorithm for advanced users
        else:
            algorithm_hint = 'FFD'  # Default to FFD for best proven results

        # Display algorithm info
        if algorithm_hint == 'FFD':
            st.info(f"💡 使用アルゴリズム: FFD (推奨 - テスト環境で67.7%配置率確認済み) / Using Algorithm: FFD (Recommended - 67.7% placement rate verified)")
        elif algorithm_hint == 'GA':
            st.warning(f"⏱️ 使用アルゴリズム: 遺伝的アルゴリズム (高品質だが時間がかかります) / Using Algorithm: Genetic Algorithm (Higher quality but slower)")
        else:
            st.warning(f"⚠️ 使用アルゴリズム: {algorithm_hint} (実験的 - FFD推奨) / Using Algorithm: {algorithm_hint} (Experimental - FFD recommended)")

        # Call optimize exactly like test_all_placement.py (67.7% placement rate)
        results = engine.optimize(panels, constraints, algorithm_hint)

        processing_time = time.time() - start_time

        progress_bar.progress(100)
        status_text.markdown(f"**✅ 最適化完了 / Optimization completed in {processing_time:.2f}s**")

        return results

    except Exception as e:
        st.error(f"最適化エラー / Optimization error: {str(e)}")
        return []

    finally:
        # Clean up progress indicators after delay
        time.sleep(2)
        progress_container.empty()


def render_enhanced_results(results: List[PlacementResult]):
    """Render enhanced optimization results with comprehensive analytics"""
    if not results:
        st.warning("最適化結果がありません / No optimization results")
        return

    st.markdown("""
    <div class="results-section">
        <h2>🎯 最適化結果 / Optimization Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced summary metrics
    total_panels = sum(len(result.panels) for result in results)
    avg_efficiency = sum(result.efficiency for result in results) / len(results)
    total_cost = sum(result.cost for result in results)
    total_time = sum(result.processing_time for result in results)
    total_waste_area = sum(result.waste_area for result in results)

    # Metrics grid
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <h4 style="color: #1f77b4; margin: 0;">使用シート数</h4>
            <h2 style="margin: 0;">{len(results)}</h2>
            <p style="margin: 0; color: #666;">Sheets Used</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <h4 style="color: #2ca02c; margin: 0;">配置パネル数</h4>
            <h2 style="margin: 0;">{total_panels}</h2>
            <p style="margin: 0; color: #666;">Placed Panels</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-item">
            <h4 style="color: #ff7f0e; margin: 0;">平均効率</h4>
            <h2 style="margin: 0;">{avg_efficiency:.1%}</h2>
            <p style="margin: 0; color: #666;">Average Efficiency</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-item">
            <h4 style="color: #d62728; margin: 0;">総コスト</h4>
            <h2 style="margin: 0;">¥{total_cost:,.0f}</h2>
            <p style="margin: 0; color: #666;">Total Cost</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-item">
            <h4 style="color: #9467bd; margin: 0;">廃棄面積</h4>
            <h2 style="margin: 0;">{total_waste_area:,.0f}</h2>
            <p style="margin: 0; color: #666;">Waste Area (mm²)</p>
        </div>
        """, unsafe_allow_html=True)

    # Navigation to detailed analysis
    render_analysis_navigation(results)

    # Interactive visualization with coordinate system update
    from ui.visualizer import render_cutting_visualization
    render_cutting_visualization(results)

    # Enhanced export options
    render_export_options(results)


def render_analysis_navigation(results: List[PlacementResult]):
    """Render navigation to detailed analysis page"""
    st.subheader("📊 詳細分析 / Detailed Analysis")

    # Quick summary
    total_sheets = len(results)
    total_panels = sum(len(result.panels) for result in results)
    avg_efficiency = sum(result.efficiency for result in results) / len(results) * 100 if results else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("使用シート数", f"{total_sheets:,}")
    with col2:
        st.metric("配置パネル数", f"{total_panels:,}")
    with col3:
        st.metric("平均効率", f"{avg_efficiency:.1f}%")
    with col4:
        if st.button("📊 詳細分析を見る", type="primary", use_container_width=True):
            st.switch_page("pages/4_📊_Analysis_Results.py")

    st.info("💡 詳細な材料使用分析、コスト内訳、パフォーマンス情報は専用の分析ページで確認できます")


def render_export_options(results: List[PlacementResult]):
    """Render enhanced export options"""
    st.subheader("📤 エクスポート / Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 作業指示書生成 / Generate Work Instructions", use_container_width=True):
            generate_work_instructions(results)

    with col2:
        if st.button("📊 レポート出力 / Export Report", use_container_width=True):
            export_optimization_report(results)

    with col3:
        if st.button("💾 結果保存 / Save Results", use_container_width=True):
            save_optimization_results(results)


def generate_work_instructions(results: List[PlacementResult]):
    """Generate enhanced work instructions"""
    try:
        from cutting.instruction import WorkInstructionGenerator
        from cutting.sequence import CuttingSequenceOptimizer

        generator = WorkInstructionGenerator()
        optimizer = CuttingSequenceOptimizer()

        with st.spinner("作業指示書を生成中... / Generating work instructions..."):
            instruction_data = []

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
                        'kerf_width': 3.5,
                        'material_type': result.material_block
                    }
                )

                instruction_data.append({
                    'Sheet': f"Sheet {i}",
                    'Material': result.material_block,
                    'Steps': work_instruction.total_steps,
                    'Est. Time (min)': f"{work_instruction.estimated_total_time:.1f}",
                    'Complexity': f"{work_instruction.complexity_score:.2f}",
                    'Cut Length (mm)': f"{work_instruction.total_cut_length:.0f}"
                })

        # Display summary table
        st.success("✅ 作業指示書生成完了 / Work instructions generated successfully")
        instructions_df = pd.DataFrame(instruction_data)
        st.dataframe(instructions_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"作業指示書生成エラー: {str(e)}")


def export_optimization_report(results: List[PlacementResult]):
    """Export comprehensive optimization report"""
    try:
        from cutting.export import DocumentExporter
        import tempfile
        import os

        exporter = DocumentExporter()

        with st.spinner("レポートを出力中... / Exporting report..."):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                success = exporter.export_efficiency_report_excel(
                    results=results,
                    file_path=tmp_file.name,
                    include_charts=True
                )

                if success:
                    st.success("✅ レポート出力完了 / Report export completed")

                    # Provide download button
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            "📁 レポートダウンロード / Download Report",
                            data=f.read(),
                            file_name=f"optimization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                    # Clean up
                    os.unlink(tmp_file.name)
                else:
                    st.error("レポート出力に失敗しました / Report export failed")

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
        with st.expander("📊 保存された結果サマリー / Saved Results Summary", expanded=False):
            st.json(results_data['summary'])

    except Exception as e:
        st.error(f"結果保存エラー: {str(e)}")


def render_sidebar_help():
    """Render usage help in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🚀 使用方法 / How to Use")

        with st.expander("📋 基本的な流れ / Basic Flow", expanded=False):
            st.markdown("""
            1. **パネル入力** - ファイルアップロードまたはテキストデータでパネル情報を入力
            2. **材料検証** - 入力されたパネルが材料在庫と照合されます
            3. **最適化実行** - アルゴリズムを選択して実行
            4. **結果確認** - 切断レイアウト、効率、コストを確認
            """)

        with st.expander("📁 ファイル形式 / File Formats", expanded=False):
            st.markdown("""
            **対応形式 / Supported Formats:**
            - TSV形式 (data0923.txt形式)
            - CSV形式
            - JSON形式

            **必須項目 / Required Fields:**
            - 製造番号, PI, W, H, 数量, 材質, 板厚
            """)

        with st.expander("💡 最適化のヒント / Optimization Tips", expanded=False):
            st.markdown("""
            - パネルの回転を許可すると効率が向上します
            - 材質別分離により品質が向上します
            - 薄板切断では切断代を0に設定済み
            - PIコードによる寸法展開が自動実行されます
            """)

        with st.expander("⚙️ 技術情報 / Technical Info", expanded=False):
            st.markdown("""
            **アルゴリズム / Algorithms:**
            - FFD: 高速、基本効率
            - BFD: 中速、高効率
            - GA: 低速、最高効率
            - HYBRID: バランス型

            **制約 / Constraints:**
            - ギロチンカット制約
            - 最小パネルサイズ: 50×50mm
            - 最大シートサイズ: 1500×3100mm
            """)


def main():
    """Main function for cutting optimization page"""
    setup_page()
    render_page_header()
    render_sidebar_help()

    # Check material inventory status
    material_manager = get_material_manager()
    if len(material_manager.inventory) == 0:
        st.markdown("""
        <div class="warning-message">
            <h4>⚠️ 材料在庫が空です / Material Inventory is Empty</h4>
            <p>最適化を実行する前に、材料管理ページで材料を追加してください。</p>
            <p>Please add materials via the Material Management page before running optimization.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📦 材料管理ページへ / Go to Material Management", type="primary"):
            st.switch_page("pages/2_📦_Material_Management.py")
        return

    # Main workflow
    st.markdown("---")

    # Panel input section (main content area)
    panels = render_panel_input_section()

    if panels:
        st.markdown("---")

        # Optimization settings
        algorithm, constraints = render_optimization_settings()

        # Optimization execution
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "🚀 最適化実行 / Run Optimization",
                type="primary",
                use_container_width=True,
                disabled=len(panels) == 0
            ):
                st.markdown("---")
                results = run_optimization(panels, algorithm, constraints)

                if results:
                    # Store results in session state
                    st.session_state.optimization_results = results
                    render_enhanced_results(results)
                else:
                    st.error("最適化に失敗しました / Optimization failed")

        with col2:
            if st.button("🔄 パネルクリア / Clear Panels", use_container_width=True):
                st.session_state.panels = []
                st.rerun()

        # Show previous results if available
        if hasattr(st.session_state, 'optimization_results') and st.session_state.optimization_results:
            if st.checkbox("📊 前回の結果を表示 / Show Previous Results", value=False):
                st.markdown("---")
                st.subheader("📈 前回の最適化結果 / Previous Optimization Results")
                render_enhanced_results(st.session_state.optimization_results)

    else:
        # Show brief welcome message when no panels are entered
        st.info("📋 パネル情報を入力して最適化を開始してください / Please enter panel information to start optimization")
        st.markdown("詳しい使用方法は左のサイドバーをご確認ください / Please check the sidebar for detailed usage instructions")


if __name__ == "__main__":
    main()