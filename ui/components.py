"""
UI Components for Steel Cutting Optimization System
鋼板切断最適化システム用UIコンポーネント
"""

import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple, Any
import io

from core.models import Panel, SteelSheet, OptimizationConstraints
from core.text_parser import RobustTextParser, ParseResult


class PanelInputComponent:
    """
    Panel input component with Japanese support
    日本語対応パネル入力コンポーネント
    """
    
    def __init__(self):
        self.parser = RobustTextParser()
        
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
        
        # Input method selection - prioritize file upload, remove manual input
        input_method = st.radio(
            self.ui_text['input_method'],
            ['file_upload', 'text_input'],
            format_func=lambda x: self.ui_text.get(x, x.replace('_', ' ').title()),
            horizontal=True,
            help="ファイルアップロードを推奨します / File upload is recommended"
        )

        if input_method == 'file_upload':
            self._render_file_upload()
        elif input_method == 'text_input':
            self._render_text_input()
        
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
            material_options = ['SS400', 'SUS304', 'SUS316', 'AL6061', 'その他 / Other']
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
                            material=material or 'SS400',
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
    
    def _render_text_input(self):
        """Render text data input with parsing"""
        st.write("### テキストデータ入力 / Text Data Input")
        
        # Show sample formats
        with st.expander(self.ui_text['sample_formats']):
            st.write("**CSV形式 / CSV Format:**")
            st.code("""panel1,300,200,2,SS400,6.0,5,true
panel2,400,300,1,SUS304,3.0,3,false""")
            
            st.write("**TSV形式 / TSV Format:**")
            st.code("""panel1	300	200	2	SS400	6.0	5	true
panel2	400	300	1	SUS304	3.0	3	false""")
            
            st.write("**JSON形式 / JSON Format:**")
            st.code("""{
  "panels": [
    {"id": "panel1", "width": 300, "height": 200, "quantity": 2, "material": "SS400", "thickness": 6.0},
    {"id": "panel2", "width": 400, "height": 300, "quantity": 1, "material": "SUS304", "thickness": 3.0}
  ]
}""")
        
        # Text input area
        text_data = st.text_area(
            "テキストデータを入力 / Enter text data",
            height=200,
            placeholder="CSV, TSV, またはJSON形式でパネルデータを入力してください\nEnter panel data in CSV, TSV, or JSON format",
            key="text_input_data"
        )
        
        # Format hint
        format_hint = st.selectbox(
            "データ形式 / Data Format",
            ['auto', 'csv', 'tsv', 'json'],
            help="自動検出または手動選択 / Auto-detect or manual selection"
        )
        
        if st.button(self.ui_text['parse_text'], type="primary"):
            if text_data.strip():
                self._parse_and_add_panels(text_data, format_hint if format_hint != 'auto' else None)
            else:
                st.warning("テキストデータを入力してください / Please enter text data")
    
    def _render_file_upload(self):
        """Render file upload component"""
        st.write("### ファイルアップロード / File Upload")
        
        uploaded_file = st.file_uploader(
            "パネルデータファイルを選択 / Select panel data file",
            type=['csv', 'txt', 'json', 'xlsx'],
            help="CSV, TXT, JSON, またはExcelファイルをアップロード / Upload CSV, TXT, JSON, or Excel file"
        )
        
        if uploaded_file is not None:
            try:
                # Read file content
                if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    # Excel file
                    df = pd.read_excel(uploaded_file)
                    text_data = df.to_csv(index=False)
                    format_hint = 'csv'
                else:
                    # Text file
                    content = uploaded_file.read()
                    if isinstance(content, bytes):
                        text_data = content.decode('utf-8')
                    else:
                        text_data = content
                    format_hint = None
                
                st.write("**ファイル内容プレビュー / File Content Preview:**")
                st.text(text_data[:500] + "..." if len(text_data) > 500 else text_data)
                
                if st.button("ファイルを解析 / Parse File", type="primary"):
                    self._parse_and_add_panels(text_data, format_hint)
                    
            except Exception as e:
                st.error(f"ファイル読み込みエラー / File read error: {str(e)}")


    def _parse_and_add_panels(self, text_data: str, format_hint: Optional[str] = None):
        """Parse text data and add panels to session state"""
        try:
            result = self.parser.parse_to_panels(text_data, format_hint)
            
            if result.is_successful:
                # Add successfully parsed panels
                st.session_state.panels.extend(result.panels)
                
                st.success(
                    f"✅ {len(result.panels)}個のパネルを追加しました / "
                    f"Added {len(result.panels)} panels\n"
                    f"成功率 / Success rate: {result.success_rate:.1%}"
                )
                
                # Show warnings if any
                if result.warnings:
                    with st.expander("⚠️ 警告 / Warnings"):
                        for warning in result.warnings:
                            st.warning(warning)
                
                # Show errors if any
                if result.errors:
                    with st.expander("❌ エラー / Errors"):
                        for error in result.errors:
                            st.error(f"Line {error.line_number}: {error.error_message}")
                            if error.suggested_fix:
                                st.info(f"推奨修正 / Suggested fix: {error.suggested_fix}")
                
                st.rerun()
            else:
                st.error("パネルデータの解析に失敗しました / Failed to parse panel data")
                if result.errors:
                    for error in result.errors:
                        st.error(f"Line {error.line_number}: {error.error_message}")
        
        except Exception as e:
            st.error(f"解析エラー / Parse error: {str(e)}")
    
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
                ['SS400', 'SUS304', 'SUS316', 'AL6061'],
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
            'time_budget': '時間制限 (秒) / Time Budget (seconds)',
            'target_efficiency': '目標効率 / Target Efficiency (%)',
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
                ['FFD (推奨 - 67.7%配置率/Recommended)', 'GA (実験的 - 開発中/Experimental)', 'BFD (非推奨/Not Recommended)', 'AUTO'],
                index=0,  # Default to FFD
                help="FFD推奨 - 67.7%配置率で高速・安定 / FFD Recommended - 67.7% placement rate, fast and stable",
                key="opt_algorithm"
            )
            
            time_budget = st.slider(
                "⚠️ 時間制限 (無効化済み) / Time Budget (Disabled)",
                min_value=1.0,
                max_value=300.0,
                value=30.0,
                step=1.0,
                key="opt_time_budget",
                help="実際は時間制限なしで実行されます (全パネル配置まで継続) / Actually runs without time limit (continues until all panels placed)",
                disabled=True  # Make it clear this setting is overridden
            )
            
            target_efficiency = st.slider(
                self.ui_text['target_efficiency'],
                min_value=50,
                max_value=95,
                value=75,
                step=5,
                key="opt_target_efficiency"
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
                "⚠️ 材質別分離 (無効化済み) / Material Separation (Disabled)",
                value=False,
                help="マルチシート最適化が強制適用されます (テスト環境設定) / Multi-sheet optimization is forced (test environment setting)",
                key="opt_material_separation",
                disabled=True  # Make it clear this setting is overridden
            )
        
        # Override UI settings with optimized values for best panel placement
        # NO TIME LIMIT - run until 100% placement is achieved

        constraints = OptimizationConstraints(
            kerf_width=0.0,  # Force thin sheet cutting (test environment setting)
            allow_rotation=allow_rotation,
            material_separation=False,  # Force multi-sheet optimization (test environment setting)
            time_budget=0.0,  # NO TIME LIMIT - run until 100% placement
            target_efficiency=1.0,  # Target 100% placement
            max_sheets=1000  # Allow unlimited sheets for complete placement (test environment setting)
        )
        
        return algorithm, constraints