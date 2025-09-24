"""
Material Management UI Components
材料管理UI コンポーネント

UI components for managing material inventory
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import json
import os

from core.material_manager import MaterialInventoryManager, MaterialSheet, get_material_manager
from core.persistence_adapter import get_persistence_adapter


class MaterialManagementUI:
    """
    Material inventory management UI
    材料在庫管理UI
    """

    def __init__(self):
        # Use persistence adapter for database-first approach with JSON fallback
        self.persistence = get_persistence_adapter()
        # Keep legacy manager for compatibility during transition
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

    def get_materials(self, material_type: Optional[str] = None) -> List[MaterialSheet]:
        """Get materials using persistence adapter"""
        return self.persistence.get_materials(material_type)

    def get_inventory_summary(self) -> Dict:
        """Get inventory summary using persistence adapter"""
        materials = self.get_materials()

        if not materials:
            return {
                'total_sheets': 0,
                'material_types': 0,
                'total_area': 0.0,
                'total_value': 0.0,
                'by_material_type': {}
            }

        # Calculate summary statistics
        total_sheets = sum(m.availability for m in materials)
        material_types = len(set(m.material_type for m in materials))
        total_area = sum(m.area * m.availability for m in materials)
        total_value = sum(m.cost_per_sheet * m.availability for m in materials)

        # Group by material type
        by_material_type = {}
        for material in materials:
            if material.material_type not in by_material_type:
                by_material_type[material.material_type] = {
                    'count': 0,
                    'area': 0.0,
                    'value': 0.0
                }

            by_material_type[material.material_type]['count'] += material.availability
            by_material_type[material.material_type]['area'] += material.area * material.availability
            by_material_type[material.material_type]['value'] += material.cost_per_sheet * material.availability

        return {
            'total_sheets': total_sheets,
            'material_types': material_types,
            'total_area': total_area,
            'total_value': total_value,
            'by_material_type': by_material_type
        }

    def get_all_material_types(self) -> List[str]:
        """Get all unique material types"""
        materials = self.get_materials()
        return sorted(list(set(m.material_type for m in materials)))

    def update_material(self, material_code: str, updates: Dict) -> bool:
        """Update material using persistence adapter"""
        # Get current material
        materials = self.get_materials()
        material = next((m for m in materials if m.material_code == material_code), None)
        if not material:
            return False

        # Create updated material object
        updated_material = MaterialSheet(
            material_code=material_code,
            material_type=updates.get('material_type', material.material_type),
            thickness=updates.get('thickness', material.thickness),
            width=updates.get('width', material.width),
            height=updates.get('height', material.height),
            area=updates.get('area', material.area),
            cost_per_sheet=updates.get('cost_per_sheet', material.cost_per_sheet),
            availability=updates.get('availability', material.availability),
            supplier=updates.get('supplier', material.supplier)
        )

        # For now, fall back to manager until persistence adapter implements update
        return self.manager.update_material_sheet(material_code, updates)

    def render(self):
        """Render material management interface"""
        st.title(self.ui_text['title'])

        # Create tabs for different functions
        tab1, tab2, tab3 = st.tabs([
            "📊 在庫概要 / Overview",
            "📋 材料一覧 / Material List",
            "➕ 材料管理 / Manage Materials"
        ])

        with tab1:
            self._render_inventory_overview()

        with tab2:
            self._render_material_list()

        with tab3:
            self._render_material_management()

    def _render_inventory_overview(self):
        """Render inventory overview"""
        st.subheader(self.ui_text['inventory_summary'])

        summary = self.get_inventory_summary()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総材料数 / Total Sheets", summary['total_sheets'])
        with col2:
            st.metric("材質種類 / Material Types", summary['material_types'])
        with col3:
            st.metric("総面積 / Total Area (mm²)", f"{summary['total_area']:,.0f}")
        with col4:
            st.metric("総価値 / Total Value (¥)", f"{summary['total_value']:,.0f}")

        # Material type breakdown
        if summary['by_material_type']:
            st.subheader("材質別内訳 / Breakdown by Material Type")

            breakdown_data = []
            for material_type, data in summary['by_material_type'].items():
                breakdown_data.append({
                    '材質 / Material Type': material_type,
                    '数量 / Count': data['count'],
                    '総面積 / Total Area (mm²)': f"{data['area']:,.0f}",
                    '総価値 / Total Value (¥)': f"{data['value']:,.0f}"
                })

            df_breakdown = pd.DataFrame(breakdown_data)
            st.dataframe(df_breakdown, use_container_width=True)

            # Chart
            import plotly.express as px
            if len(breakdown_data) > 0:
                fig = px.pie(
                    values=[data['count'] for data in summary['by_material_type'].values()],
                    names=list(summary['by_material_type'].keys()),
                    title="材質別分布 / Distribution by Material Type"
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_material_list(self):
        """Render material list with filtering"""
        st.subheader(self.ui_text['material_list'])

        materials = self.get_materials()
        if not materials:
            st.info("材料在庫がありません。手動で材料を追加してください。")
            st.info("No materials in inventory. Please add materials manually.")
            return

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            material_types = ['すべて / All'] + self.get_all_material_types()
            selected_type = st.selectbox(
                "材質フィルタ / Material Type Filter",
                material_types
            )

        with col2:
            thicknesses = ['すべて / All'] + sorted(list(set(f"{s.thickness}mm" for s in materials)))
            selected_thickness = st.selectbox(
                "板厚フィルタ / Thickness Filter",
                thicknesses
            )

        with col3:
            search_term = st.text_input("検索 / Search", placeholder="材料コードまたは材質名")

        # Filter materials
        filtered_materials = materials.copy()

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

        # Display materials
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
                    '単価 / Cost': f"¥{material.cost_per_sheet:,.0f}",
                    'サプライヤー / Supplier': material.supplier or '-'
                })

            df = pd.DataFrame(material_data)
            st.dataframe(df, use_container_width=True)

            st.info(f"表示中: {len(filtered_materials)} / {len(materials)} 材料")

        else:
            st.warning("フィルタ条件に一致する材料がありません / No materials match the filter criteria")

    def _render_material_management(self):
        """Render material add/edit/delete interface"""
        st.subheader("材料管理 / Material Management")

        # Action selection
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
        """Render add material form"""
        st.write("### 新規材料追加 / Add New Material")

        with st.form("add_material_form"):
            col1, col2 = st.columns(2)

            with col1:
                material_code = st.text_input(
                    self.ui_text['material_code'] + " *",
                    placeholder="例: E-201P"
                )

                # Suggest material types from existing inventory
                existing_types = self.get_all_material_types()
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
                    step=0.1
                )

                width = st.number_input(
                    self.ui_text['width'] + " *",
                    min_value=100.0,
                    max_value=2000.0,
                    value=1200.0,
                    step=10.0
                )

            with col2:
                height = st.number_input(
                    self.ui_text['height'] + " *",
                    min_value=100.0,
                    max_value=4000.0,
                    value=2400.0,
                    step=10.0
                )

                cost = st.number_input(
                    self.ui_text['cost'],
                    min_value=1000.0,
                    max_value=100000.0,
                    value=15000.0,
                    step=1000.0
                )

                availability = st.number_input(
                    self.ui_text['availability'],
                    min_value=0,
                    max_value=1000,
                    value=100,
                    step=1
                )

                supplier = st.text_input(
                    self.ui_text['supplier'],
                    placeholder="例: サプライヤーA"
                )

            # Calculated area
            area = width * height
            st.info(f"計算面積 / Calculated Area: {area:,.0f} mm²")

            submitted = st.form_submit_button(self.ui_text['save'], type="primary")

            if submitted:
                if material_code and material_type:
                    new_material = MaterialSheet(
                        material_code=material_code,
                        material_type=material_type,
                        thickness=thickness,
                        width=width,
                        height=height,
                        area=area,
                        cost_per_sheet=cost,
                        availability=availability,
                        supplier=supplier
                    )

                    if self.persistence.add_material(new_material):
                        st.success(f"材料 {material_code} を追加しました / Added material {material_code}")
                        st.rerun()
                    else:
                        st.error(f"材料 {material_code} は既に存在します / Material {material_code} already exists")
                else:
                    st.error("必須項目を入力してください / Please fill required fields")

    def _render_edit_material(self):
        """Render edit material form"""
        st.write("### 材料編集 / Edit Material")

        materials = self.get_materials()
        if not materials:
            st.info("編集する材料がありません / No materials to edit")
            return

        # Select material to edit
        material_codes = [m.material_code for m in materials]
        selected_code = st.selectbox(
            "編集する材料を選択 / Select Material to Edit",
            material_codes
        )

        if selected_code:
            # Find the selected material
            material = next((m for m in materials if m.material_code == selected_code), None)
            if material:
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
                        new_cost = st.number_input(
                            self.ui_text['cost'],
                            min_value=1000.0,
                            max_value=100000.0,
                            value=material.cost_per_sheet,
                            step=1000.0
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
                            value=material.supplier
                        )

                    new_area = new_width * new_height
                    st.info(f"計算面積 / Calculated Area: {new_area:,.0f} mm²")

                    submitted = st.form_submit_button(self.ui_text['save'], type="primary")

                    if submitted:
                        updates = {
                            'material_type': new_material_type,
                            'thickness': new_thickness,
                            'width': new_width,
                            'height': new_height,
                            'area': new_area,
                            'cost_per_sheet': new_cost,
                            'availability': new_availability,
                            'supplier': new_supplier
                        }

                        if self.update_material(selected_code, updates):
                            st.success(f"材料 {selected_code} を更新しました / Updated material {selected_code}")
                            st.rerun()
                        else:
                            st.error("更新に失敗しました / Failed to update")

    def _render_delete_material(self):
        """Render delete material interface"""
        st.write("### 材料削除 / Delete Material")

        materials = self.get_materials()
        if not materials:
            st.info("削除する材料がありません / No materials to delete")
            return

        # Select material to delete
        material_codes = [m.material_code for m in materials]
        selected_code = st.selectbox(
            "削除する材料を選択 / Select Material to Delete",
            material_codes
        )

        if selected_code:
            material = next((m for m in materials if m.material_code == selected_code), None)
            if material:
                st.warning("""
                **削除確認 / Confirm Deletion**

                材料コード / Material Code: {material.material_code}
                材質 / Material Type: {material.material_type}
                サイズ / Size: {material.width:.0f}×{material.height:.0f}mm
                板厚 / Thickness: {material.thickness}mm

                この操作は取り消せません / This action cannot be undone.
                """)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ 削除実行 / Delete", type="primary"):
                        # For now, use manager until persistence adapter implements delete
                        if self.manager.remove_material_sheet(selected_code):
                            st.success(f"材料 {selected_code} を削除しました / Deleted material {selected_code}")
                            st.rerun()
                        else:
                            st.error("削除に失敗しました / Failed to delete")

                with col2:
                    if st.button("キャンセル / Cancel"):
                        st.info("削除をキャンセルしました / Deletion cancelled")


def render_material_management():
    """Render material management interface"""
    ui = MaterialManagementUI()
    ui.render()
