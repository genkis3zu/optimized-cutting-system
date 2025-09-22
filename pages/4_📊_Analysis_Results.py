"""
Analysis Results Page
分析結果ページ

This page displays detailed analysis results including material usage analysis,
cost breakdown, and comprehensive optimization reports.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional
import plotly.express as px

from core.models import PlacementResult


def main():
    """Main analysis results page"""

    st.title("📊 分析結果 / Analysis Results")
    st.markdown("最適化結果の詳細分析とレポート / Detailed analysis and reports of optimization results")

    # Check if optimization results exist in session state
    if 'optimization_results' not in st.session_state or not st.session_state.optimization_results:
        st.warning("⚠️ 分析結果がありません。まず切断最適化ページで最適化を実行してください。")
        st.warning("⚠️ No analysis results available. Please run optimization from the Cutting Optimization page first.")

        if st.button("🔧 切断最適化ページへ移動 / Go to Cutting Optimization"):
            st.switch_page("pages/1_🔧_Cutting_Optimization.py")
        return

    results = st.session_state.optimization_results

    # Analysis overview
    st.markdown("---")
    render_analysis_overview(results)

    # Material usage analysis
    st.markdown("---")
    render_material_usage_analysis(results)

    # Additional analysis sections could be added here
    st.markdown("---")
    render_performance_analysis(results)


def render_analysis_overview(results: List[PlacementResult]):
    """Render analysis overview summary"""
    st.subheader("📈 分析概要 / Analysis Overview")

    # Calculate overall statistics
    total_sheets = len(results)
    total_panels = sum(len(result.panels) for result in results)
    total_cost = sum(result.cost for result in results)
    avg_efficiency = sum(result.efficiency for result in results) / len(results) * 100 if results else 0
    total_waste = sum(result.waste_area for result in results)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="使用シート数 / Sheets Used",
            value=f"{total_sheets:,}",
            delta=None
        )

    with col2:
        st.metric(
            label="配置パネル数 / Panels Placed",
            value=f"{total_panels:,}",
            delta=None
        )

    with col3:
        st.metric(
            label="総コスト / Total Cost",
            value=f"¥{total_cost:,.0f}",
            delta=None
        )

    with col4:
        st.metric(
            label="平均効率 / Average Efficiency",
            value=f"{avg_efficiency:.1f}%",
            delta=None
        )

    # Additional details
    with st.expander("📋 詳細統計 / Detailed Statistics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**シート情報 / Sheet Information:**")
            st.write(f"- 平均配置パネル数: {total_panels/total_sheets:.1f} パネル/シート")
            st.write(f"- 平均シートコスト: ¥{total_cost/total_sheets:,.0f}")

        with col2:
            st.write("**効率情報 / Efficiency Information:**")
            st.write(f"- 総廃棄面積: {total_waste:,.0f} mm²")
            st.write(f"- 平均廃棄面積: {total_waste/total_sheets:,.0f} mm²/シート")


def render_material_usage_analysis(results: List[PlacementResult]):
    """Render detailed material usage analysis"""
    st.subheader("📊 材料使用分析 / Material Usage Analysis")

    # Material usage summary table
    material_usage = {}
    panel_placement_details = []

    for i, result in enumerate(results, 1):
        sheet_id = f"Sheet_{i:03d}"
        material_type = result.material_block

        if material_type not in material_usage:
            material_usage[material_type] = {
                'sheets_used': 0,
                'total_panels': 0,
                'total_efficiency': 0,
                'total_cost': 0,
                'total_area': 0,
                'total_waste': 0
            }

        material_usage[material_type]['sheets_used'] += 1
        material_usage[material_type]['total_panels'] += len(result.panels)
        material_usage[material_type]['total_efficiency'] += result.efficiency
        material_usage[material_type]['total_cost'] += result.cost
        material_usage[material_type]['total_area'] += result.sheet.area
        material_usage[material_type]['total_waste'] += result.waste_area

        # Panel placement details
        for placed_panel in result.panels:
            panel_placement_details.append({
                'シートID / Sheet ID': sheet_id,
                '材質 / Material': material_type,
                'パネルID / Panel ID': placed_panel.panel.id,
                '配置位置 / Position': f"({placed_panel.x:.0f}, {placed_panel.y:.0f})",
                'サイズ / Size': f"{placed_panel.actual_width:.0f}×{placed_panel.actual_height:.0f}mm",
                '回転 / Rotated': '○' if placed_panel.rotated else '×',
                '面積 / Area (mm²)': f"{placed_panel.actual_width * placed_panel.actual_height:,.0f}"
            })

    # Material usage summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 材質別使用状況 / Material Usage Summary")
        usage_data = []
        for material, data in material_usage.items():
            avg_efficiency = (data['total_efficiency'] / data['sheets_used']) * 100
            usage_data.append({
                '材質 / Material': material,
                'シート数 / Sheets': data['sheets_used'],
                'パネル数 / Panels': data['total_panels'],
                '平均効率 / Avg Efficiency (%)': f"{avg_efficiency:.1f}",
                'コスト / Cost (¥)': f"{data['total_cost']:,.0f}",
                '廃棄面積 / Waste (mm²)': f"{data['total_waste']:,.0f}"
            })

        usage_df = pd.DataFrame(usage_data)
        st.dataframe(usage_df, use_container_width=True, hide_index=True)

    with col2:
        # Cost breakdown chart
        if material_usage:
            fig = px.bar(
                x=list(material_usage.keys()),
                y=[data['total_cost'] for data in material_usage.values()],
                title="材質別コスト分布 / Cost Distribution by Material",
                labels={'x': 'Material Type', 'y': 'Total Cost (JPY)'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Detailed panel placement table
    if st.checkbox("📋 詳細配置情報を表示 / Show Detailed Placement", value=False):
        st.markdown("#### 🎯 パネル配置詳細 / Panel Placement Details")
        placement_df = pd.DataFrame(panel_placement_details)

        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            selected_materials = st.multiselect(
                "材質フィルタ / Material Filter",
                options=list(material_usage.keys()),
                default=list(material_usage.keys())
            )
        with col2:
            selected_sheets = st.multiselect(
                "シートフィルタ / Sheet Filter",
                options=placement_df['シートID / Sheet ID'].unique(),
                default=placement_df['シートID / Sheet ID'].unique()
            )

        # Apply filters
        filtered_df = placement_df[
            (placement_df['材質 / Material'].isin(selected_materials)) &
            (placement_df['シートID / Sheet ID'].isin(selected_sheets))
        ]

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Summary of filtered results
        st.info(f"📊 表示中: {len(filtered_df)} / {len(placement_df)} パネル配置")


def render_performance_analysis(results: List[PlacementResult]):
    """Render performance analysis"""
    st.subheader("⚡ パフォーマンス分析 / Performance Analysis")

    # Efficiency distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 効率分布 / Efficiency Distribution")
        efficiency_data = [result.efficiency * 100 for result in results]

        if efficiency_data:
            fig = px.histogram(
                x=efficiency_data,
                nbins=20,
                title="シート効率分布 / Sheet Efficiency Distribution",
                labels={'x': 'Efficiency (%)', 'y': 'Number of Sheets'},
                color_discrete_sequence=['#2E8B57']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ⏱️ アルゴリズム情報 / Algorithm Information")
        if results:
            # Show algorithm and processing time information
            algorithms_used = list(set(result.algorithm for result in results))
            total_processing_time = sum(result.processing_time for result in results)

            st.write(f"**使用アルゴリズム / Algorithm Used:** {', '.join(algorithms_used)}")
            st.write(f"**総処理時間 / Total Processing Time:** {total_processing_time:.3f}秒")
            st.write(f"**平均処理時間 / Average Processing Time:** {total_processing_time/len(results):.3f}秒/シート")

            # Processing time by sheet
            processing_times = [result.processing_time * 1000 for result in results]  # Convert to ms
            if processing_times:
                fig = px.line(
                    x=list(range(1, len(processing_times) + 1)),
                    y=processing_times,
                    title="シート別処理時間 / Processing Time per Sheet",
                    labels={'x': 'Sheet Number', 'y': 'Processing Time (ms)'},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()