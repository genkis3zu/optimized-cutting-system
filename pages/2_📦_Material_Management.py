"""
Material Management Page
材料管理ページ

Streamlit page for managing material inventory with enhanced UI
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import json
import os

from core.material_manager import MaterialInventoryManager, MaterialSheet, get_material_manager


def setup_page():
    """Setup page configuration and styling"""
    st.set_page_config(
        page_title="Material Management - Steel Cutting System",
        page_icon="📦",
        layout="wide"
    )

    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


class MaterialManagementUI:
    """
    Enhanced Material inventory management UI
    強化された材料在庫管理UI
    """

    def __init__(self):
        self.manager = get_material_manager()

        self.ui_text = {
            'title': '材料在庫管理 / Material Inventory Management',
            'add_material': '材料追加 / Add Material',
            'edit_material': '材料編集 / Edit Material',
            'delete_material': '材料削除 / Delete Material',
            'inventory_summary': '在庫サマリー / Inventory Summary',
            'material_list': '材料一覧 / Material List',
            'material_code': '材料コード / Material Code',
            'material_type': '材質 / Material Type',
            'thickness': '板厚 (mm) / Thickness (mm)',
            'width': '幅 (mm) / Width (mm)',
            'height': '高さ (mm) / Height (mm)',
            'cost': '単価 (円) / Cost (JPY)',
            'availability': '在庫数 / Stock Count',
            'supplier': 'サプライヤー / Supplier',
            'save': '保存 / Save',
            'cancel': 'キャンセル / Cancel',
            'confirm_delete': '削除確認 / Confirm Delete'
        }

    def render(self):
        """Render enhanced material management interface"""
        # Page header
        st.markdown("""
        <div class="main-header">
            <h1>📦 材料在庫管理システム</h1>
            <p>Material Inventory Management System</p>
        </div>
        """, unsafe_allow_html=True)

        # Navigation tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 在庫概要 / Overview",
            "📋 材料一覧 / Material List",
            "⚙️ 材料管理 / Manage Materials"
        ])

        with tab1:
            self._render_inventory_overview()

        with tab2:
            self._render_material_list()

        with tab3:
            self._render_material_management()

    def _render_inventory_overview(self):
        """Render enhanced inventory overview"""
        st.subheader("📊 " + self.ui_text['inventory_summary'])

        summary = self.manager.get_inventory_summary()

        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0;">総材料数</h3>
                <h2 style="margin: 0;">{}</h2>
                <p style="margin: 0; color: #666;">Total Sheets</p>
            </div>
            """.format(summary['total_sheets']), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #2ca02c; margin: 0;">材質種類</h3>
                <h2 style="margin: 0;">{}</h2>
                <p style="margin: 0; color: #666;">Material Types</p>
            </div>
            """.format(summary['material_types']), unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #ff7f0e; margin: 0;">総面積</h3>
                <h2 style="margin: 0;">{:,.0f}</h2>
                <p style="margin: 0; color: #666;">Total Area (mm²)</p>
            </div>
            """.format(summary['total_area']), unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #d62728; margin: 0;">総価値</h3>
                <h2 style="margin: 0;">¥{:,.0f}</h2>
                <p style="margin: 0; color: #666;">Total Value (JPY)</p>
            </div>
            """.format(summary['total_value']), unsafe_allow_html=True)

        # Material type breakdown with enhanced visualization
        if summary['by_material_type']:
            st.subheader("📈 材質別内訳 / Breakdown by Material Type")

            breakdown_data = []
            for material_type, data in summary['by_material_type'].items():
                breakdown_data.append({
                    '材質 / Material Type': material_type,
                    '数量 / Count': data['count'],
                    '総面積 / Total Area (mm²)': f"{data['total_area']:,.0f}",
                    '総価値 / Total Value (¥)': f"{data['total_value']:,.0f}",
                    '平均面積 / Avg Area (mm²)': f"{data['total_area']/data['count']:,.0f}",
                    '単価レンジ / Price Range (¥)': f"{data['total_value']/data['count']:,.0f}"
                })

            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown, use_container_width=True)

            # Enhanced visualization
            import plotly.express as px
            if len(breakdown_data) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart for material distribution
                    fig_pie = px.pie(
                        values=[data['count'] for data in summary['by_material_type'].values()],
                        names=list(summary['by_material_type'].keys()),
                        title="材質別分布 / Distribution by Material Type",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Bar chart for value distribution
                    fig_bar = px.bar(
                        x=list(summary['by_material_type'].keys()),
                        y=[data['total_value'] for data in summary['by_material_type'].values()],
                        title="材質別価値 / Value by Material Type",
                        labels={'x': 'Material Type', 'y': 'Total Value (JPY)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

    def _render_material_list(self):
        """Render enhanced material list with filtering"""
        st.subheader("📋 " + self.ui_text['material_list'])

        if not self.manager.inventory:
            st.info("材料在庫がありません。手動で材料を追加してください。")
            st.info("No materials in inventory. Please add materials manually.")
            return

        # Enhanced filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            material_types = ['すべて / All'] + self.manager.get_all_material_types()
            selected_type = st.selectbox(
                "材質フィルタ / Material Type Filter",
                material_types
            )

        with col2:
            thicknesses = ['すべて / All'] + sorted(list(set(f"{s.thickness}mm" for s in self.manager.inventory)))
            selected_thickness = st.selectbox(
                "板厚フィルタ / Thickness Filter",
                thicknesses
            )

        with col3:
            search_term = st.text_input("検索 / Search", placeholder="材料コードまたは材質名")

        # Apply filters
        filtered_materials = self.manager.inventory.copy()

        if selected_type != 'すべて / All':
            filtered_materials = [m for m in filtered_materials if m.material_type == selected_type]

        if selected_thickness != 'すべて / All':
            thickness_value = float(selected_thickness.replace('mm', ''))
            filtered_materials = [m for m in filtered_materials if abs(m.thickness - thickness_value) < 0.01]


        if search_term:
            search_term = search_term.lower()
            filtered_materials = [
                m for m in filtered_materials
                if search_term in m.material_code.lower() or search_term in m.material_type.lower()
            ]

        # Display enhanced material table
        if filtered_materials:
            material_data = []
            for material in filtered_materials:
                material_data.append({
                    '材料コード / Code': material.material_code,
                    '材質 / Type': material.material_type,
                    '板厚 / Thickness': f"{material.thickness}mm",
                    '幅 / Width (mm)': f"{material.width:.0f}",
                    '高さ / Height (mm)': f"{material.height:.0f}",
                    '面積 / Area': f"{material.area:,.0f}mm²",
                    '在庫 / Stock': material.availability,
                    'サプライヤー / Supplier': material.supplier or '-',
                    '最終更新 / Updated': material.last_updated[:10] if material.last_updated else '-'
                })

            df = pd.DataFrame(material_data)

            # Enhanced table display with selection
            event = st.dataframe(
                df,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            # Summary of filtered results
            st.info(f"📋 表示中: {len(filtered_materials)} / {len(self.manager.inventory)} 材料")

        else:
            st.warning("フィルタ条件に一致する材料がありません / No materials match the filter criteria")

    def _render_material_management(self):
        """Render enhanced material add/edit/delete interface"""
        st.subheader("⚙️ 材料管理 / Material Management")

        # Action selection with enhanced UI
        action = st.radio(
            "アクション / Action",
            ['add', 'edit', 'delete'],
            format_func=lambda x: {
                'add': '➕ 新規追加 / Add New',
                'edit': '✏️ 編集 / Edit',
                'delete': '🗑️ 削除 / Delete'
            }[x],
            horizontal=True
        )

        if action == 'add':
            self._render_add_material()
        elif action == 'edit':
            self._render_edit_material()
        elif action == 'delete':
            self._render_delete_material()

    def _render_add_material(self):
        """Render enhanced add material form"""
        st.write("### ➕ 新規材料追加 / Add New Material")

        with st.form("add_material_form", clear_on_submit=False):
            col1, col2 = st.columns(2)

            with col1:
                material_code = st.text_input(
                    self.ui_text['material_code'] + " *",
                    placeholder="例: E-201P",
                    help="一意の材料識別コード"
                )

                # Enhanced material type selection
                existing_types = self.manager.get_all_material_types()
                if existing_types:
                    material_type = st.selectbox(
                        self.ui_text['material_type'] + " *",
                        ['新しい材質 / New Type'] + existing_types
                    )
                    if material_type == '新しい材質 / New Type':
                        material_type = st.text_input(
                            "新しい材質名 / New Material Type *",
                            placeholder="例: SECC"
                        )
                else:
                    material_type = st.text_input(
                        self.ui_text['material_type'] + " *",
                        placeholder="例: SECC"
                    )

                thickness = st.number_input(
                    self.ui_text['thickness'] + " *",
                    min_value=0.1,
                    max_value=50.0,
                    value=0.6,
                    step=0.1,
                    help="材料の板厚"
                )

                width = st.number_input(
                    self.ui_text['width'] + " *",
                    min_value=100.0,
                    max_value=2000.0,
                    value=1200.0,
                    step=10.0,
                    help="材料の幅"
                )

            with col2:
                height = st.number_input(
                    self.ui_text['height'] + " *",
                    min_value=100.0,
                    max_value=4000.0,
                    value=2400.0,
                    step=10.0,
                    help="材料の高さ"
                )


                availability = st.number_input(
                    self.ui_text['availability'],
                    min_value=0,
                    max_value=1000,
                    value=100,
                    step=1,
                    help="現在の在庫数"
                )

                supplier = st.text_input(
                    self.ui_text['supplier'],
                    placeholder="例: サプライヤーA",
                    help="材料の供給業者"
                )

            # Real-time calculations
            area = width * height

            st.markdown(f"""
            **計算値 / Calculated Values:**
            - 面積 / Area: **{area:,.0f} mm²**
            """)

            # Enhanced submit section
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button(
                    "💾 " + self.ui_text['save'],
                    type="primary",
                    use_container_width=True
                )
            with col2:
                if st.form_submit_button("🔄 Reset", use_container_width=True):
                    st.rerun()

            if submitted:
                if material_code and material_type:
                    new_material = MaterialSheet(
                        material_code=material_code,
                        material_type=material_type,
                        thickness=thickness,
                        width=width,
                        height=height,
                        area=area,
                        cost_per_sheet=0.0,
                        availability=availability,
                        supplier=supplier
                    )

                    if self.manager.add_material_sheet(new_material):
                        st.markdown(f"""
                        <div class="success-message">
                            ✅ <strong>成功！</strong> 材料 {material_code} を追加しました<br>
                            <strong>Success!</strong> Added material {material_code}
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error(f"❌ 材料 {material_code} は既に存在します / Material {material_code} already exists")
                else:
                    st.error("⚠️ 必須項目を入力してください / Please fill required fields")

    def _render_edit_material(self):
        """Render enhanced edit material form"""
        st.write("### ✏️ 材料編集 / Edit Material")

        if not self.manager.inventory:
            st.info("編集する材料がありません / No materials to edit")
            return

        # Enhanced material selection
        material_codes = [m.material_code for m in self.manager.inventory]
        selected_code = st.selectbox(
            "編集する材料を選択 / Select Material to Edit",
            material_codes,
            help="編集したい材料を選択してください"
        )

        if selected_code:
            material = self.manager.get_material_sheet(selected_code)
            if material:
                # Display current material info
                with st.expander("📋 現在の材料情報 / Current Material Info", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**材料コード:** {material.material_code}")
                        st.write(f"**材質:** {material.material_type}")
                        st.write(f"**板厚:** {material.thickness}mm")
                    with col2:
                        st.write(f"**サイズ:** {material.width:.0f}×{material.height:.0f}mm")
                        st.write(f"**面積:** {material.area:,.0f}mm²")
                        st.write(f"**在庫数:** {material.availability}")
                    with col3:
                        st.write(f"**サプライヤー:** {material.supplier or 'N/A'}")
                        st.write(f"**更新日:** {material.last_updated[:10] if material.last_updated else 'N/A'}")

                with st.form("edit_material_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        new_material_type = st.text_input(
                            self.ui_text['material_type'],
                            value=material.material_type
                        )
                        new_thickness = st.number_input(
                            self.ui_text['thickness'],
                            min_value=0.1,
                            max_value=50.0,
                            value=material.thickness,
                            step=0.1
                        )
                        new_width = st.number_input(
                            self.ui_text['width'],
                            min_value=100.0,
                            max_value=2000.0,
                            value=material.width,
                            step=10.0
                        )

                    with col2:
                        new_height = st.number_input(
                            self.ui_text['height'],
                            min_value=100.0,
                            max_value=4000.0,
                            value=material.height,
                            step=10.0
                        )
                        new_availability = st.number_input(
                            self.ui_text['availability'],
                            min_value=0,
                            max_value=1000,
                            value=material.availability,
                            step=1
                        )
                        new_supplier = st.text_input(
                            self.ui_text['supplier'],
                            value=material.supplier or ""
                        )

                    new_area = new_width * new_height
                    st.info(f"📐 新しい計算面積 / New Calculated Area: {new_area:,.0f} mm²")

                    submitted = st.form_submit_button("💾 " + self.ui_text['save'], type="primary")

                    if submitted:
                        updates = {
                            'material_type': new_material_type,
                            'thickness': new_thickness,
                            'width': new_width,
                            'height': new_height,
                            'area': new_area,
                            'cost_per_sheet': 0.0,
                            'availability': new_availability,
                            'supplier': new_supplier
                        }

                        if self.manager.update_material_sheet(selected_code, updates):
                            st.success(f"✅ 材料 {selected_code} を更新しました / Updated material {selected_code}")
                            st.rerun()
                        else:
                            st.error("❌ 更新に失敗しました / Failed to update")

    def _render_delete_material(self):
        """Render enhanced delete material interface"""
        st.write("### 🗑️ 材料削除 / Delete Material")

        if not self.manager.inventory:
            st.info("削除する材料がありません / No materials to delete")
            return

        # Enhanced material selection for deletion
        material_codes = [m.material_code for m in self.manager.inventory]
        selected_code = st.selectbox(
            "削除する材料を選択 / Select Material to Delete",
            material_codes,
            help="⚠️ この操作は元に戻せません"
        )

        if selected_code:
            material = self.manager.get_material_sheet(selected_code)
            if material:
                # Enhanced warning display
                st.markdown(f"""
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                    <h4 style="color: #856404; margin: 0;">⚠️ 削除確認 / Confirm Deletion</h4>
                    <div style="margin-top: 1rem;">
                        <strong>材料コード:</strong> {material.material_code}<br>
                        <strong>材質:</strong> {material.material_type}<br>
                        <strong>サイズ:</strong> {material.width:.0f}×{material.height:.0f}mm<br>
                        <strong>板厚:</strong> {material.thickness}mm<br>
                        <strong>在庫数:</strong> {material.availability}<br>
                    </div>
                    <p style="color: #856404; margin-top: 1rem; margin-bottom: 0;">
                        <strong>警告:</strong> この操作は取り消せません / This action cannot be undone.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "🗑️ 削除実行 / Confirm Delete",
                        type="primary",
                        use_container_width=True
                    ):
                        if self.manager.remove_material_sheet(selected_code):
                            st.success(f"✅ 材料 {selected_code} を削除しました / Deleted material {selected_code}")
                            st.rerun()
                        else:
                            st.error("❌ 削除に失敗しました / Failed to delete")

                with col2:
                    if st.button(
                        "❌ キャンセル / Cancel",
                        use_container_width=True
                    ):
                        st.info("✅ 削除をキャンセルしました / Deletion cancelled")


def main():
    """Main function for material management page"""
    setup_page()

    # Initialize and render UI
    ui = MaterialManagementUI()
    ui.render()


if __name__ == "__main__":
    main()