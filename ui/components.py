"""
UI Components for Steel Cutting Optimization System
鋼板切断最適化システム用UIコンポーネント
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple

from core.models import Panel, SteelSheet, OptimizationConstraints
# Removed complex text parser - now using simple grid input


class PanelInputComponent:
    """
    Panel input component with Japanese support
    日本語対応パネル入力コンポーネント
    """

    def __init__(self):

        # UI text in Japanese and English
        self.ui_text = {
            'panel_input': 'パネル入力 / Panel Input',
            'input_method': '入力方法 / Input Method',
            'manual_input': '手動入力 / Manual Input',
            'text_input': 'テキストデータ / Text Data',
            'file_upload': 'ファイルアップロード / File Upload',
            'panel_id': 'パネルID / Panel ID',
            'width': '幅 (mm) / Width (mm)',
            'height': '高さ (mm) / Height (mm)',
            'quantity': '数量 / Quantity',
            'material': '材質 / Material',
            'thickness': '板厚 (mm) / Thickness (mm)',
            'priority': '優先度 / Priority',
            'allow_rotation': '回転許可 / Allow Rotation',
            'add_panel': 'パネル追加 / Add Panel',
            'clear_all': 'すべてクリア / Clear All',
            'parse_text': 'テキスト解析 / Parse Text',
            'sample_formats': 'サンプル形式 / Sample Formats'
        }

    def render(self) -> List[Panel]:
        """
        Render panel input component and return list of panels
        パネル入力コンポーネントを描画してパネルリストを返す
        """
        st.subheader(self.ui_text['panel_input'])

        # Initialize session state for panels
        if 'panels' not in st.session_state:
            st.session_state.panels = []

        # Simple data grid for Excel copy-paste
        self._render_data_grid()

        # Show panel count in sidebar (details moved to main page)
        if st.session_state.panels:
            total_panels = len(st.session_state.panels)
            total_quantity = sum(p.quantity for p in st.session_state.panels)
            st.info(f"📋 パネル: {total_panels}種類, {total_quantity}個")

            if st.button("詳細をメインページで確認 / View Details on Main Page"):
                st.session_state.show_panel_details = True

        return st.session_state.panels

    def _render_manual_input(self):
        """Render manual panel input form"""
        st.write("### 手動パネル入力 / Manual Panel Input")

        col1, col2, col3 = st.columns(3)

        with col1:
            panel_id = st.text_input(self.ui_text['panel_id'], key="manual_id")
            width = st.number_input(
                self.ui_text['width'],
                min_value=50.0,
                max_value=1500.0,
                value=300.0,
                step=10.0,
                key="manual_width"
            )
            height = st.number_input(
                self.ui_text['height'],
                min_value=50.0,
                max_value=3100.0,
                value=200.0,
                step=10.0,
                key="manual_height"
            )

        with col2:
            quantity = st.number_input(
                self.ui_text['quantity'],
                min_value=1,
                max_value=100,
                value=1,
                key="manual_quantity"
            )

            # Material selection with Japanese options
            material_options = ['SGCC', 'SUS304', 'SUS316', 'AL6061', 'その他 / Other']
            material = st.selectbox(
                self.ui_text['material'],
                material_options,
                key="manual_material"
            )

            if material == 'その他 / Other':
                material = st.text_input(
                    "材質名を入力 / Enter material name",
                    key="manual_material_custom"
                )

            thickness = st.number_input(
                self.ui_text['thickness'],
                min_value=0.1,
                max_value=50.0,
                value=6.0,
                step=0.1,
                key="manual_thickness"
            )

        with col3:
            priority = st.slider(
                self.ui_text['priority'],
                min_value=1,
                max_value=10,
                value=5,
                key="manual_priority"
            )

            allow_rotation = st.checkbox(
                self.ui_text['allow_rotation'],
                value=True,
                key="manual_rotation"
            )

        # Add panel button
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button(self.ui_text['add_panel'], type="primary"):
                if panel_id and width and height:
                    try:
                        panel = Panel(
                            id=panel_id,
                            width=width,
                            height=height,
                            quantity=quantity,
                            material=material or 'SGCC',
                            thickness=thickness,
                            priority=priority,
                            allow_rotation=allow_rotation
                        )
                        st.session_state.panels.append(panel)
                        st.success(f"パネル {panel_id} を追加しました / Added panel {panel_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"パネル追加エラー / Panel add error: {str(e)}")
                else:
                    st.warning("すべての必須項目を入力してください / Please fill all required fields")

        with col_btn2:
            if st.button(self.ui_text['clear_all']):
                st.session_state.panels = []
                st.success("すべてのパネルをクリアしました / Cleared all panels")
                st.rerun()

    def _render_data_grid(self):
        """Render data grid for Excel copy-paste input"""
        st.write("### 📋 データ入力グリッド / Data Input Grid")
        st.write("**Excelからコピー貼り付けが可能です / You can copy-paste from Excel**")

        # Define column configuration for the specified headers
        column_config = {
            "製造番号": st.column_config.TextColumn("製造番号", help="製造番号を入力", max_chars=20),
            "PI": st.column_config.TextColumn("PI", help="PIコードを入力", max_chars=20),
            "部材名": st.column_config.TextColumn("部材名", help="部材名を入力", max_chars=50),
            "W": st.column_config.NumberColumn("W", help="幅（mm）", min_value=50, max_value=1500, format="%.1f"),
            "H": st.column_config.NumberColumn("H", help="高さ（mm）", min_value=50, max_value=3100, format="%.1f"),
            "寸法3": st.column_config.NumberColumn("寸法3", help="寸法3", format="%.1f"),
            "数量": st.column_config.NumberColumn("数量", help="数量", min_value=1, format="%d"),
            "識別番号": st.column_config.NumberColumn("識別番号", help="識別番号", format="%d"),
            "品名": st.column_config.TextColumn("品名", help="品名を入力", max_chars=100),
            "色": st.column_config.TextColumn("色", help="材質・色を入力", max_chars=20),
        }

        # Initialize empty dataframe if not exists
        if 'grid_data' not in st.session_state:
            # Create empty dataframe with specified columns
            st.session_state.grid_data = pd.DataFrame({
                "製造番号": [""] * 10,
                "PI": [""] * 10,
                "部材名": [""] * 10,
                "W": [0.0] * 10,
                "H": [0.0] * 10,
                "寸法3": [0.0] * 10,
                "数量": [1] * 10,
                "識別番号": [0] * 10,
                "品名": [""] * 10,
                "色": [""] * 10,
            })

        # Data editor with copy-paste capability
        edited_data = st.data_editor(
            st.session_state.grid_data,
            column_config=column_config,
            num_rows="dynamic",
            use_container_width=True,
            height=400,
            key="data_grid"
        )

        # Update session state
        st.session_state.grid_data = edited_data

        # Process data button
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 データを追加 / Add Data", type="primary"):
                self._process_grid_data(edited_data)

        with col2:
            if st.button("🗑️ グリッドをクリア / Clear Grid"):
                st.session_state.grid_data = pd.DataFrame({
                    "製造番号": [""] * 10,
                    "PI": [""] * 10,
                    "部材名": [""] * 10,
                    "W": [0.0] * 10,
                    "H": [0.0] * 10,
                    "寸法3": [0.0] * 10,
                    "数量": [1] * 10,
                    "識別番号": [0] * 10,
                    "品名": [""] * 10,
                    "色": [""] * 10,
                })
                st.rerun()

        with col3:
            if st.button("🔄 行を追加 / Add Rows"):
                # Add 10 more empty rows
                new_rows = pd.DataFrame({
                    "製造番号": [""] * 10,
                    "PI": [""] * 10,
                    "部材名": [""] * 10,
                    "W": [0.0] * 10,
                    "H": [0.0] * 10,
                    "寸法3": [0.0] * 10,
                    "数量": [1] * 10,
                    "識別番号": [0] * 10,
                    "品名": [""] * 10,
                    "色": [""] * 10,
                })
                st.session_state.grid_data = pd.concat([st.session_state.grid_data, new_rows], ignore_index=True)
                st.rerun()

    def _process_grid_data(self, data):
        """Process data from the grid and convert to Panel objects"""
        try:
            # Filter out empty rows (where W and H are 0 or empty)
            valid_rows = []
            for _, row in data.iterrows():
                if (row['W'] > 0 and row['H'] > 0 and
                    str(row['製造番号']).strip() and
                    str(row['PI']).strip()):
                    valid_rows.append(row)

            if not valid_rows:
                st.warning("有効なデータがありません / No valid data found")
                return

            # Convert rows to Panel objects
            new_panels = []
            for row in valid_rows:
                try:
                    # Create panel with the grid data
                    panel = Panel(
                        id=str(row['製造番号']).strip(),
                        width=float(row['W']),
                        height=float(row['H']),
                        quantity=int(row['数量']),
                        material=str(row['色']).strip() if str(row['色']).strip() else 'SGCC',
                        thickness=0.5,  # Default thickness
                        pi_code=str(row['PI']).strip()
                    )
                    new_panels.append(panel)
                except Exception as e:
                    st.error(f"行データ変換エラー / Row conversion error: {str(e)}")
                    continue

            if new_panels:
                # Add to session state panels
                if 'panels' not in st.session_state:
                    st.session_state.panels = []
                st.session_state.panels.extend(new_panels)
                st.success(f"{len(new_panels)}個のパネルを追加しました / Added {len(new_panels)} panels")
                st.rerun()
            else:
                st.error("パネルデータの変換に失敗しました / Failed to convert panel data")

        except Exception as e:
            st.error(f"データ処理エラー / Data processing error: {str(e)}")

    def _show_loaded_data_table(self):
        """Show loaded panel data in a table for verification"""
        try:
            # Convert panels to DataFrame for display
            data = []
            for panel in st.session_state.panels:
                data.append({
                    'パネルID / Panel ID': panel.id,
                    '幅 / Width (mm)': panel.width,
                    '高さ / Height (mm)': panel.height,
                    '数量 / Quantity': panel.quantity,
                    '材質 / Material': panel.material,
                    '板厚 / Thickness (mm)': panel.thickness,
                    '回転許可 / Rotation': '○' if panel.allow_rotation else '×',
                    '面積 / Area (mm²)': f"{panel.area:,.0f}"
                })

            df = pd.DataFrame(data)

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("パネル種類 / Panel Types", len(st.session_state.panels))
            with col2:
                total_qty = sum(p.quantity for p in st.session_state.panels)
                st.metric("総数量 / Total Qty", total_qty)
            with col3:
                total_area = sum(p.area * p.quantity for p in st.session_state.panels)
                st.metric("総面積 / Total Area", f"{total_area:,.0f} mm²")
            with col4:
                materials = set(p.material for p in st.session_state.panels)
                st.metric("材質種類 / Material Types", len(materials))

            # Display the data table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(400, len(df) * 35 + 50)  # Limit height but allow scrolling
            )

            # Clear data button
            if st.button("データをクリア / Clear Data", type="secondary"):
                st.session_state.panels = []
                st.rerun()

        except Exception as e:
            st.error(f"データ表示エラー / Data display error: {str(e)}")


    def _render_panel_list(self):
        """Render current panel list"""
        if st.session_state.panels:
            st.write("### 現在のパネル / Current Panels")

            # Convert panels to DataFrame for display
            panel_data = []
            for i, panel in enumerate(st.session_state.panels):
                panel_data.append({
                    'No.': i + 1,
                    'ID': panel.id,
                    '幅/Width (mm)': panel.width,
                    '高さ/Height (mm)': panel.height,
                    '数量/Qty': panel.quantity,
                    '材質/Material': panel.material,
                    '板厚/Thickness (mm)': panel.thickness,
                    '面積/Area (mm²)': f"{panel.area:,.0f}",
                    '回転/Rotation': '○' if panel.allow_rotation else '×'
                })

            df = pd.DataFrame(panel_data)
            st.dataframe(df, use_container_width=True)

            # Summary
            total_panels = len(st.session_state.panels)
            total_quantity = sum(p.quantity for p in st.session_state.panels)
            total_area = sum(p.area * p.quantity for p in st.session_state.panels)
            materials = set(p.material for p in st.session_state.panels)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("パネル種類 / Panel Types", total_panels)
            with col2:
                st.metric("総数量 / Total Quantity", total_quantity)
            with col3:
                st.metric("総面積 / Total Area (mm²)", f"{total_area:,.0f}")
            with col4:
                st.metric("材質種類 / Material Types", len(materials))

            # Remove panel functionality
            if st.button("最後のパネルを削除 / Remove Last Panel"):
                if st.session_state.panels:
                    removed = st.session_state.panels.pop()
                    st.success(f"パネル {removed.id} を削除しました / Removed panel {removed.id}")
                    st.rerun()


class SteelSheetComponent:
    """Steel sheet configuration component"""

    def __init__(self):
        self.ui_text = {
            'sheet_config': '鋼板設定 / Steel Sheet Configuration',
            'width': '幅 (mm) / Width (mm)',
            'height': '高さ (mm) / Height (mm)',
            'thickness': '板厚 (mm) / Thickness (mm)',
            'material': '材質 / Material',
            'cost': '単価 (円) / Cost (JPY)',
            'availability': '在庫数 / Stock Count'
        }

    def render(self) -> SteelSheet:
        """Render steel sheet configuration"""
        st.subheader(self.ui_text['sheet_config'])

        col1, col2 = st.columns(2)

        with col1:
            width = st.number_input(
                self.ui_text['width'],
                min_value=100.0,
                max_value=2000.0,
                value=1500.0,
                step=50.0,
                key="sheet_width"
            )

            height = st.number_input(
                self.ui_text['height'],
                min_value=100.0,
                max_value=4000.0,
                value=3100.0,
                step=50.0,
                key="sheet_height"
            )

            thickness = st.number_input(
                self.ui_text['thickness'],
                min_value=0.1,
                max_value=50.0,
                value=6.0,
                step=0.1,
                key="sheet_thickness"
            )

        with col2:
            material = st.selectbox(
                self.ui_text['material'],
                ['SGCC', 'SUS304', 'SUS316', 'AL6061'],
                key="sheet_material"
            )

            cost = st.number_input(
                self.ui_text['cost'],
                min_value=1000.0,
                max_value=100000.0,
                value=15000.0,
                step=1000.0,
                key="sheet_cost"
            )

            availability = st.number_input(
                self.ui_text['availability'],
                min_value=1,
                max_value=1000,
                value=100,
                key="sheet_availability"
            )

        return SteelSheet(
            width=width,
            height=height,
            thickness=thickness,
            material=material,
            cost_per_sheet=cost,
            availability=availability
        )


class OptimizationSettingsComponent:
    """Optimization settings configuration component"""

    def __init__(self):
        self.ui_text = {
            'optimization_settings': '最適化設定 / Optimization Settings',
            'algorithm': 'アルゴリズム / Algorithm',
            'kerf_width': '切断代 (mm) / Kerf Width (mm) - 薄板切断用',
            'allow_rotation': '回転許可 / Allow Rotation',
            'material_separation': '材質別分離 / Material Separation'
        }

    def render(self) -> Tuple[str, OptimizationConstraints]:
        """Render optimization settings"""
        st.subheader(self.ui_text['optimization_settings'])

        col1, col2 = st.columns(2)

        with col1:
            algorithm = st.selectbox(
                self.ui_text['algorithm'],
                ['AUTO', 'FFD', 'BFD', 'GENETIC', 'HYBRID'],
                help="AUTO: 自動選択 / Automatic selection",
                key="opt_algorithm"
            )

        with col2:
            kerf_width = st.number_input(
                self.ui_text['kerf_width'],
                min_value=0.0,
                max_value=10.0,
                value=0.0,  # Changed to 0.0 for thin sheet cutting
                step=0.1,
                key="opt_kerf_width",
                help="薄板切断のため0に設定 / Set to 0 for thin sheet cutting"
            )

            allow_rotation = st.checkbox(
                self.ui_text['allow_rotation'],
                value=True,
                key="opt_allow_rotation"
            )

            material_separation = st.checkbox(
                self.ui_text['material_separation'],
                value=True,
                help="材質ごとに分けて最適化 / Optimize separately by material",
                key="opt_material_separation"
            )

            # 100% placement guarantee setting
            placement_guarantee = st.checkbox(
                "100%配置保証 / 100% Placement Guarantee",
                value=True,
                help="全パネルを必ず配置 (シート数増加) / Place all panels (may increase sheet count)",
                key="opt_placement_guarantee"
            )

            # GPU acceleration setting
            gpu_acceleration = st.checkbox(
                "GPU加速 / GPU Acceleration",
                value=True,
                help="Intel Iris Xe グラフィックスによる高速化 / Accelerate with Intel Iris Xe Graphics",
                key="opt_gpu_acceleration"
            )

        constraints = OptimizationConstraints(
            max_sheets=1000 if placement_guarantee else 20,  # 100% placement guarantee vs efficiency focus
            kerf_width=kerf_width,
            allow_rotation=allow_rotation,
            material_separation=material_separation,
            time_budget=86400.0,  # 24 hours - effectively unlimited for optimization
            target_efficiency=0.1,  # Very low target to avoid efficiency warnings
            enable_gpu=gpu_acceleration,
            gpu_memory_limit=2048
        )

        return algorithm, constraints
