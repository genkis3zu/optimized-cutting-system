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
from ui.page_headers import render_unified_header, get_page_config
from ui.common_styles import get_common_css
from core.result_formatter import ResultFormatter


def setup_page():
    """Setup page configuration and styling"""
    st.set_page_config(
        page_title="Cutting Optimization - Steel Cutting System",
        page_icon="🔧",
        layout="wide"
    )

    # Apply unified styling
    st.markdown(get_common_css(), unsafe_allow_html=True)


def render_page_header():
    """Render unified page header"""
    config = get_page_config("cutting_optimization")
    render_unified_header(
        title_ja=config["title_ja"],
        title_en=config["title_en"],
        description=config["description"],
        icon=config["icon"]
    )


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

        # Compact material validation display
        with st.expander("🔍 材料検証 / Material Validation", expanded=False):
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
                st.warning("⚠️ 材料検証エラーがあります / Material validation issues found")
                for issue in validation_issues:
                    st.text(issue)
                st.info("💡 材料管理ページで在庫を確認・追加してください / Please check inventory in Material Management page")
            else:
                st.success("✅ すべての材料が検証されました / All materials validated")

            if validation_success and st.checkbox("詳細を表示 / Show details", key="validation_details"):
                for success in validation_success:
                    st.text(success)

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


def estimate_optimization_time(panel_count: int) -> dict:
    """Estimate optimization time based on panel count"""
    if panel_count <= 20:
        return {"estimated_seconds": 30, "complexity": "Simple", "description": "小規模問題"}
    elif panel_count <= 50:
        return {"estimated_seconds": 120, "complexity": "Medium", "description": "中規模問題"}
    elif panel_count <= 100:
        return {"estimated_seconds": 300, "complexity": "Large", "description": "大規模問題"}
    else:
        return {"estimated_seconds": 600, "complexity": "Very Large", "description": "超大規模問題"}


def run_optimization_with_progress(panels: List[Panel], algorithm: str, constraints, estimated_time: dict):
    """Run optimization with detailed progress tracking and cancellation support"""
    if not panels:
        st.error("パネルが入力されていません / No panels provided")
        return []

    # Create and configure optimization engine
    engine = create_optimization_engine()

    # Register FFD algorithm
    ffd_algorithm = create_ffd_algorithm()
    engine.register_algorithm(ffd_algorithm)

    # Enhanced progress tracking
    progress_container = st.container()

    with progress_container:
        st.markdown(f"### 🔄 最適化進行中 / Optimization in Progress")

        # Estimation display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("パネル数 / Panels", len(panels))
        with col2:
            st.metric("複雑度 / Complexity", estimated_time["description"])
        with col3:
            st.metric("予測時間 / Estimated", f"{estimated_time['estimated_seconds']}秒")

        progress_bar = st.progress(0)
        status_text = st.empty()
        time_display = st.empty()
        detail_text = st.empty()

    start_time = time.time()

    try:
        # Phase 1: Initialization
        status_text.markdown("**🚀 初期化中... / Initializing...**")
        progress_bar.progress(10)
        time.sleep(0.5)  # Allow UI update

        # Check for cancellation
        if st.session_state.get('optimization_cancelled', False):
            return []

        # Phase 2: Algorithm selection
        status_text.markdown("**🧠 アルゴリズム選択中... / Selecting algorithm...**")
        progress_bar.progress(20)
        time.sleep(0.5)

        if st.session_state.get('optimization_cancelled', False):
            return []

        # Phase 3: Optimization execution
        status_text.markdown("**⚙️ 最適化実行中... / Running optimization...**")
        progress_bar.progress(30)

        algorithm_hint = None if algorithm == 'AUTO' else algorithm

        # Simulate progress updates during optimization
        for i in range(30, 90, 10):
            if st.session_state.get('optimization_cancelled', False):
                status_text.markdown("**⏹️ 最適化を中止しています... / Cancelling optimization...**")
                return []

            progress_bar.progress(i)
            elapsed = time.time() - start_time
            remaining = max(0, estimated_time["estimated_seconds"] - elapsed)

            time_display.markdown(f"""
            **⏰ 時間情報 / Time Info:**
            - 経過時間 / Elapsed: {elapsed:.1f}秒
            - 残り時間 / Remaining: ~{remaining:.1f}秒
            """)

            detail_text.markdown(f"**📊 進行状況**: アルゴリズム実行中... ({i}% complete)")
            time.sleep(1)  # Simulate work and allow cancellation check

        # Show what we're optimizing
        st.info(f"""
        **🔍 最適化対象 / Optimization Target:**
        - 入力パネル数 / Input panels: {len(panels)}
        - 材質別 / By material: {dict((material, count) for material, count in [(p.material, sum(1 for q in panels if q.material == p.material)) for p in set(panels)])}
        """)

        # Final optimization call
        results = engine.optimize(
            panels=panels,
            constraints=constraints,
            algorithm_hint=algorithm_hint
        )

        processing_time = time.time() - start_time

        # Phase 4: Completion
        progress_bar.progress(100)
        status_text.markdown(f"**✅ 最適化完了 / Optimization completed in {processing_time:.2f}s**")

        # Show final statistics with debug info
        if results:
            total_panels = sum(len(sheet.panels) for sheet in results)
            total_expected = len(panels)
            efficiency = (sum(sheet.efficiency for sheet in results) / len(results)) if results else 0

            detail_text.markdown(f"""
            **📊 最終結果 / Final Results:**
            - 入力パネル数 / Input panels: {total_expected}
            - 配置パネル数 / Placed panels: {total_panels}
            - 配置率 / Placement rate: {(total_panels/total_expected*100):.1f}%
            - 平均効率 / Average efficiency: {efficiency:.1%}
            - 使用シート数 / Sheets used: {len(results)}
            """)

            # Warning if not all panels placed
            if total_panels < total_expected:
                st.warning(f"""
                ⚠️ 警告: {total_expected - total_panels} パネルが配置されていません！
                Warning: {total_expected - total_panels} panels were not placed!
                """)
        else:
            st.error("結果なし / No results returned from optimization")

        return results

    except Exception as e:
        st.error(f"最適化エラー / Optimization error: {str(e)}")
        return []

    finally:
        # Keep progress display for a moment
        time.sleep(3)
        progress_container.empty()


def run_optimization(panels: List[Panel], algorithm: str, constraints):
    """Legacy function for backward compatibility"""
    estimated_time = estimate_optimization_time(len(panels))
    return run_optimization_with_progress(panels, algorithm, constraints, estimated_time)


def render_enhanced_results(results: List[PlacementResult], panels: List[Panel] = None):
    """Render enhanced optimization results with comprehensive analytics"""
    if not results:
        st.warning("最適化結果がありません / No optimization results")
        return

    st.markdown("""
    <div class="results-section">
        <h2>🎯 最適化結果 / Optimization Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Create formatted result table like result.txt
    st.markdown("### 📊 切断割当表 / Cutting Assignment Table")

    if panels and hasattr(st.session_state, 'panel_data_df'):
        formatter = ResultFormatter()

        # Format results to match result.txt format
        result_df = formatter.format_results(st.session_state.panel_data_df, results)

        # Display the formatted table
        st.dataframe(
            result_df,
            use_container_width=True,
            height=400,
            column_config={
                "鋼板サイズ": st.column_config.TextColumn(
                    "鋼板サイズ",
                    help="使用する母材のサイズ",
                    width="large",
                ),
                "資材コード": st.column_config.TextColumn(
                    "資材コード",
                    help="母材の資材コード",
                ),
                "数量": st.column_config.NumberColumn(
                    "数量",
                    help="使用するシート数",
                    format="%d",
                ),
                "ｺﾒﾝﾄ": st.column_config.TextColumn(
                    "組合せ",
                    help="同じシートに配置される行番号",
                ),
                "歩留まり率": st.column_config.TextColumn(
                    "歩留まり",
                    help="材料使用効率",
                ),
            }
        )

        # Download button for the formatted results
        csv_data = result_df.to_csv(sep='\t', index=False)
        st.download_button(
            label="📥 結果をダウンロード (TSV)",
            data=csv_data,
            file_name="optimization_result.txt",
            mime="text/tab-separated-values",
        )

    st.markdown("---")

    # Debug information - Show detailed placement information
    st.markdown("### 📋 配置詳細 / Placement Details")

    # Count panels per material and sheet
    material_summary = {}
    for idx, result in enumerate(results, 1):
        material = result.sheet.material if hasattr(result, 'sheet') else 'Unknown'
        sheet_dims = f"{result.sheet.width}x{result.sheet.height}" if hasattr(result, 'sheet') else 'Unknown'

        if material not in material_summary:
            material_summary[material] = []

        panel_info = {
            'sheet_num': idx,
            'sheet_dims': sheet_dims,
            'panels': result.panels if hasattr(result, 'panels') else [],
            'efficiency': result.efficiency if hasattr(result, 'efficiency') else 0
        }
        material_summary[material].append(panel_info)

    # Display material-wise breakdown
    for material, sheets in material_summary.items():
        st.markdown(f"**材質 / Material: {material}**")
        total_panels_for_material = sum(len(sheet['panels']) for sheet in sheets)
        st.write(f"- シート数 / Sheets: {len(sheets)}")
        st.write(f"- 配置パネル数 / Placed panels: {total_panels_for_material}")

        # Show each sheet's details
        for sheet in sheets:
            with st.expander(f"シート #{sheet['sheet_num']} ({sheet['sheet_dims']}) - {len(sheet['panels'])} panels - {sheet['efficiency']:.1%} efficiency"):
                if sheet['panels']:
                    panel_list = []
                    for panel in sheet['panels']:
                        panel_str = f"{panel.id}: {panel.width}x{panel.height}"
                        if hasattr(panel, 'pi_code'):
                            panel_str += f" (PI: {panel.pi_code})"
                        panel_list.append(panel_str)
                    st.write("配置パネル / Placed panels:")
                    for p in panel_list:
                        st.write(f"  - {p}")
                else:
                    st.write("パネルなし / No panels")

    st.markdown("---")

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
                        'kerf_width': 0.0,
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
    """Render simplified usage help in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📖 クイックガイド")
        st.markdown("""
        **基本手順:**
        1. パネルデータ入力
        2. 最適化実行
        3. 結果確認

        **対応形式:** TSV, CSV, JSON
        **ヒント:** 回転許可で効率向上
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
            # Check if optimization is running
            optimization_running = st.session_state.get('optimization_running', False)

            if not optimization_running:
                if st.button(
                    "🚀 最適化実行 / Run Optimization",
                    type="primary",
                    use_container_width=True,
                    disabled=len(panels) == 0
                ):
                    st.session_state.optimization_running = True
                    st.session_state.optimization_cancelled = False
                    st.rerun()
            else:
                if st.button(
                    "⏹️ 最適化中止 / Cancel Optimization",
                    type="secondary",
                    use_container_width=True
                ):
                    st.session_state.optimization_cancelled = True
                    st.session_state.optimization_running = False
                    st.success("最適化が中止されました / Optimization cancelled")
                    st.rerun()

            # Run optimization if requested
            if optimization_running and not st.session_state.get('optimization_cancelled', False):
                st.markdown("---")

                # Estimate time based on panel count
                panel_count = len(panels)
                estimated_time = estimate_optimization_time(panel_count)

                results = run_optimization_with_progress(panels, algorithm, constraints, estimated_time)

                if results:
                    # Store results in session state
                    st.session_state.optimization_results = results
                    st.session_state.optimization_running = False
                    render_enhanced_results(results, panels)
                else:
                    st.session_state.optimization_running = False
                    if not st.session_state.get('optimization_cancelled', False):
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
                # Use current panels for the previous results display
                render_enhanced_results(st.session_state.optimization_results, panels if panels else None)

    else:
        # Show brief welcome message when no panels are entered
        st.info("📋 パネル情報を入力して最適化を開始してください / Please enter panel information to start optimization")
        st.markdown("詳しい使用方法は左のサイドバーをご確認ください / Please check the sidebar for detailed usage instructions")


if __name__ == "__main__":
    main()