"""
Visual Cutting Layout Display
視覚的切断レイアウト表示

Interactive visualization of cutting plans and layouts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional, Tuple
import colorsys
import random

from core.models import PlacementResult, PlacedPanel


class CuttingLayoutVisualizer:
    """
    Interactive cutting layout visualization
    インタラクティブ切断レイアウト可視化
    """

    def __init__(self):
        # Define color palette for materials
        self.material_colors = {}
        self.panel_colors = {}

    def generate_color_palette(self, items: List[str]) -> Dict[str, str]:
        """Generate distinct colors for different items"""
        colors = {}
        for i, item in enumerate(items):
            # Generate HSV colors for better distribution
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
            value = 0.8 + (i % 2) * 0.15  # Vary brightness

            # Convert to RGB
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation, value)
            colors[item] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

        return colors

    def create_cutting_layout(self, result: PlacementResult,
                            show_cut_lines: bool = True,
                            show_measurements: bool = True) -> go.Figure:
        """Create interactive cutting layout visualization"""

        # Create figure
        fig = go.Figure()

        # Sheet outline
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=result.sheet.width, y1=result.sheet.height,
            line=dict(color="black", width=3),
            fillcolor="lightgray",
            opacity=0.3
        )

        # Generate colors for panels
        panel_ids = list(set(placed.panel.id for placed in result.panels))
        if not self.panel_colors:
            self.panel_colors = self.generate_color_palette(panel_ids)

        # Add placed panels
        for i, placed_panel in enumerate(result.panels):
            panel = placed_panel.panel
            color = self.panel_colors.get(panel.id, "lightblue")

            # Panel rectangle
            fig.add_shape(
                type="rect",
                x0=placed_panel.x,
                y0=placed_panel.y,
                x1=placed_panel.x + placed_panel.actual_width,
                y1=placed_panel.y + placed_panel.actual_height,
                line=dict(color="black", width=2),
                fillcolor=color,
                opacity=0.7
            )

            # Panel label
            center_x = placed_panel.x + placed_panel.actual_width / 2
            center_y = placed_panel.y + placed_panel.actual_height / 2

            rotation_text = " (R)" if placed_panel.rotated else ""
            label_text = f"{panel.id}<br>{placed_panel.actual_width:.0f}×{placed_panel.actual_height:.0f}{rotation_text}"

            fig.add_annotation(
                x=center_x,
                y=center_y,
                text=label_text,
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                opacity=0.8
            )

            # Show measurements if requested
            if show_measurements:
                # Width measurement
                fig.add_annotation(
                    x=center_x,
                    y=placed_panel.y - 20,
                    text=f"{placed_panel.actual_width:.0f}mm",
                    showarrow=False,
                    font=dict(size=8, color="blue")
                )

                # Height measurement
                fig.add_annotation(
                    x=placed_panel.x - 30,
                    y=center_y,
                    text=f"{placed_panel.actual_height:.0f}mm",
                    showarrow=False,
                    font=dict(size=8, color="blue"),
                    textangle=90
                )

        # Add cut lines if requested
        if show_cut_lines:
            self._add_cut_lines(fig, result)

        # Update layout
        fig.update_layout(
            title=f"切断レイアウト / Cutting Layout - Sheet {result.sheet_id}<br>" +
                  f"材質: {result.material_block} | 効率: {result.efficiency:.1%} | " +
                  f"パネル数: {len(result.panels)}",
            xaxis=dict(
                title="幅 / Width (mm)",
                range=[0, result.sheet.width + 100],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title="高さ / Height (mm)",
                range=[0, result.sheet.height + 100]
            ),
            showlegend=False,
            width=800,
            height=600,
            plot_bgcolor="white"
        )

        return fig

    def _add_cut_lines(self, fig: go.Figure, result: PlacementResult):
        """Add cutting lines to visualization"""
        # Collect all horizontal and vertical cut lines
        h_lines = set()
        v_lines = set()

        for placed_panel in result.panels:
            # Add panel boundaries as potential cut lines
            h_lines.add(placed_panel.y)  # Bottom edge
            h_lines.add(placed_panel.y + placed_panel.actual_height)  # Top edge
            v_lines.add(placed_panel.x)  # Left edge
            v_lines.add(placed_panel.x + placed_panel.actual_width)  # Right edge

        # Add sheet boundaries
        h_lines.add(0)
        h_lines.add(result.sheet.height)
        v_lines.add(0)
        v_lines.add(result.sheet.width)

        # Draw horizontal cut lines
        for y in h_lines:
            if 0 <= y <= result.sheet.height:
                fig.add_shape(
                    type="line",
                    x0=0, y0=y,
                    x1=result.sheet.width, y1=y,
                    line=dict(color="red", width=1, dash="dash"),
                    opacity=0.5
                )

        # Draw vertical cut lines
        for x in v_lines:
            if 0 <= x <= result.sheet.width:
                fig.add_shape(
                    type="line",
                    x0=x, y0=0,
                    x1=x, y1=result.sheet.height,
                    line=dict(color="red", width=1, dash="dash"),
                    opacity=0.5
                )

    def create_efficiency_chart(self, results: List[PlacementResult]) -> go.Figure:
        """Create efficiency comparison chart"""
        if not results:
            return go.Figure()

        # Prepare data
        sheet_data = []
        for i, result in enumerate(results, 1):
            sheet_data.append({
                'Sheet': f"Sheet {i}",
                'Material': result.material_block,
                'Efficiency': result.efficiency * 100,
                'Panels': len(result.panels),
                'Waste_Area': result.waste_area,
                'Used_Area': result.sheet.width * result.sheet.height - result.waste_area
            })

        df = pd.DataFrame(sheet_data)

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('効率とパネル数 / Efficiency & Panel Count',
                          '面積利用状況 / Area Utilization'),
            specs=[[{"secondary_y": True}, {"type": "domain"}]]
        )

        # Efficiency bars
        fig.add_trace(
            go.Bar(
                x=df['Sheet'],
                y=df['Efficiency'],
                name='効率 / Efficiency (%)',
                marker_color='lightblue',
                yaxis='y'
            ),
            row=1, col=1
        )

        # Panel count line
        fig.add_trace(
            go.Scatter(
                x=df['Sheet'],
                y=df['Panels'],
                mode='lines+markers',
                name='パネル数 / Panel Count',
                line=dict(color='red', width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )

        # Area utilization for first sheet as pie chart
        if len(df) > 0:
            first_row = df.iloc[0]
            fig.add_trace(
                go.Pie(
                    labels=['使用面積 / Used', '廃棄面積 / Waste'],
                    values=[first_row['Used_Area'], first_row['Waste_Area']],
                    hole=0.3,
                    marker_colors=['lightgreen', 'lightcoral'],
                    showlegend=True
                ),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title="最適化結果分析 / Optimization Results Analysis",
            height=400
        )

        # Update y-axes
        fig.update_yaxes(title_text="効率 / Efficiency (%)", row=1, col=1)
        fig.update_yaxes(title_text="パネル数 / Panel Count", secondary_y=True, row=1, col=1)

        return fig

    def create_material_distribution_chart(self, results: List[PlacementResult]) -> go.Figure:
        """Create material distribution chart"""
        if not results:
            return go.Figure()

        # Collect material data
        material_data = {}
        for result in results:
            material = result.material_block
            if material not in material_data:
                material_data[material] = {
                    'sheets': 0,
                    'panels': 0,
                    'total_area': 0,
                    'waste_area': 0,
                    'efficiency': []
                }

            material_data[material]['sheets'] += 1
            material_data[material]['panels'] += len(result.panels)
            material_data[material]['total_area'] += result.sheet.width * result.sheet.height
            material_data[material]['waste_area'] += result.waste_area
            material_data[material]['efficiency'].append(result.efficiency)

        # Create comparison chart
        materials = list(material_data.keys())
        sheets = [material_data[m]['sheets'] for m in materials]
        panels = [material_data[m]['panels'] for m in materials]
        avg_efficiency = [sum(material_data[m]['efficiency']) / len(material_data[m]['efficiency']) * 100
                         for m in materials]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '材質別シート数 / Sheets by Material',
                '材質別パネル数 / Panels by Material',
                '材質別平均効率 / Average Efficiency by Material',
                '材質別面積利用 / Area Utilization by Material'
            ),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )

        # Generate colors for materials
        if not self.material_colors:
            self.material_colors = self.generate_color_palette(materials)
        colors = [self.material_colors.get(m, 'lightblue') for m in materials]

        # Sheets by material
        fig.add_trace(
            go.Bar(x=materials, y=sheets, marker_color=colors, showlegend=False),
            row=1, col=1
        )

        # Panels by material
        fig.add_trace(
            go.Bar(x=materials, y=panels, marker_color=colors, showlegend=False),
            row=1, col=2
        )

        # Average efficiency by material
        fig.add_trace(
            go.Bar(x=materials, y=avg_efficiency, marker_color=colors, showlegend=False),
            row=2, col=1
        )

        # Area utilization
        used_areas = [material_data[m]['total_area'] - material_data[m]['waste_area'] for m in materials]
        fig.add_trace(
            go.Bar(x=materials, y=used_areas, name='使用面積 / Used Area', marker_color='lightgreen'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=materials, y=[material_data[m]['waste_area'] for m in materials],
                  name='廃棄面積 / Waste Area', marker_color='lightcoral'),
            row=2, col=2
        )

        fig.update_layout(
            title="材質別分析 / Analysis by Material",
            height=600,
            showlegend=False
        )

        return fig


def render_panel_details(panels: List, show_validation: bool = True):
    """Render detailed panel information on main page"""
    if not panels:
        return

    st.subheader("📋 パネル詳細情報 / Panel Details")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_panels = len(panels)
    total_quantity = sum(p.quantity for p in panels)
    total_area = sum(p.area * p.quantity for p in panels)
    materials = set(p.material for p in panels)

    with col1:
        st.metric("パネル種類 / Panel Types", total_panels)
    with col2:
        st.metric("総数量 / Total Quantity", total_quantity)
    with col3:
        st.metric("総面積 / Total Area (mm²)", f"{total_area:,.0f}")
    with col4:
        st.metric("材質種類 / Material Types", len(materials))

    # Panel data table
    panel_data = []
    for i, panel in enumerate(panels):
        panel_data.append({
            'No.': i + 1,
            'ID': panel.id,
            '幅/Width (mm)': panel.width,
            '高さ/Height (mm)': panel.height,
            '数量/Qty': panel.quantity,
            '材質/Material': panel.material,
            '板厚/Thickness (mm)': panel.thickness,
            '面積/Area (mm²)': f"{panel.area:,.0f}",
            '回転/Rotation': '○' if panel.allow_rotation else '×',
            '優先度/Priority': panel.priority
        })

    df = pd.DataFrame(panel_data)

    # Material validation if requested
    if show_validation:
        from core.material_manager import get_material_manager
        manager = get_material_manager()

        validation_results = []
        for panel in panels:
            is_valid, message = manager.validate_panel_against_inventory(
                panel.material, panel.thickness, panel.width, panel.height
            )
            validation_results.append("✅" if is_valid else "⚠️")

        df['検証/Validation'] = validation_results

    # Display with filtering options
    col1, col2 = st.columns([2, 1])

    with col1:
        # Material filter
        material_filter = st.selectbox(
            "材質フィルタ / Material Filter",
            ['すべて / All'] + sorted(list(materials))
        )

    with col2:
        # Search
        search_term = st.text_input("検索 / Search", placeholder="Panel ID")

    # Apply filters
    filtered_df = df.copy()

    if material_filter != 'すべて / All':
        filtered_df = filtered_df[filtered_df['材質/Material'] == material_filter]

    if search_term:
        filtered_df = filtered_df[filtered_df['ID'].str.contains(search_term, case=False, na=False)]

    # Display table
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Material breakdown chart
    if len(materials) > 1:
        st.subheader("材質別分布 / Material Distribution")

        material_summary = []
        for material in materials:
            material_panels = [p for p in panels if p.material == material]
            material_quantity = sum(p.quantity for p in material_panels)
            material_area = sum(p.area * p.quantity for p in material_panels)

            material_summary.append({
                'Material': material,
                'Panels': len(material_panels),
                'Quantity': material_quantity,
                'Area': material_area
            })

        df_summary = pd.DataFrame(material_summary)

        # Create pie chart
        fig = px.pie(
            df_summary,
            values='Quantity',
            names='Material',
            title="材質別数量分布 / Quantity Distribution by Material"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_cutting_visualization(results: List[PlacementResult]):
    """Render cutting visualization interface"""
    if not results:
        st.info("最適化結果がありません / No optimization results to display")
        return

    st.subheader("🎨 切断レイアウト可視化 / Cutting Layout Visualization")

    visualizer = CuttingLayoutVisualizer()

    # Visualization options
    col1, col2, col3 = st.columns(3)

    with col1:
        show_cut_lines = st.checkbox("切断線表示 / Show Cut Lines", value=True)
    with col2:
        show_measurements = st.checkbox("寸法表示 / Show Measurements", value=True)
    with col3:
        sheet_index = st.selectbox(
            "シート選択 / Select Sheet",
            range(len(results)),
            format_func=lambda x: f"Sheet {x+1} ({results[x].material_block})"
        )

    # Individual sheet layout
    st.write("### 個別シートレイアウト / Individual Sheet Layout")

    selected_result = results[sheet_index]
    layout_fig = visualizer.create_cutting_layout(
        selected_result,
        show_cut_lines=show_cut_lines,
        show_measurements=show_measurements
    )
    st.plotly_chart(layout_fig, use_container_width=True)

    # Sheet details
    with st.expander("シート詳細情報 / Sheet Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**基本情報 / Basic Info:**")
            st.write(f"- シートID / Sheet ID: {selected_result.sheet_id}")
            st.write(f"- 材質 / Material: {selected_result.material_block}")
            st.write(f"- シートサイズ / Sheet Size: {selected_result.sheet.width}×{selected_result.sheet.height}mm")
            st.write(f"- 配置パネル数 / Placed Panels: {len(selected_result.panels)}")

        with col2:
            st.write("**効率情報 / Efficiency Info:**")
            st.write(f"- 効率 / Efficiency: {selected_result.efficiency:.1%}")
            st.write(f"- 使用面積 / Used Area: {(selected_result.sheet.width * selected_result.sheet.height - selected_result.waste_area):,.0f} mm²")
            st.write(f"- 廃棄面積 / Waste Area: {selected_result.waste_area:,.0f} mm²")
            st.write(f"- 切断長 / Cut Length: {selected_result.cut_length:,.0f} mm")

    # Analysis charts if multiple sheets
    if len(results) > 1:
        st.write("### 分析チャート / Analysis Charts")

        tab1, tab2 = st.tabs(["効率分析 / Efficiency Analysis", "材質分析 / Material Analysis"])

        with tab1:
            efficiency_fig = visualizer.create_efficiency_chart(results)
            st.plotly_chart(efficiency_fig, use_container_width=True)

        with tab2:
            material_fig = visualizer.create_material_distribution_chart(results)
            st.plotly_chart(material_fig, use_container_width=True)