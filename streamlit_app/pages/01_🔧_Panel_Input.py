#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel Input Page - Steel Cutting Optimizer
パネル入力ページ - 鋼板切断最適化システム
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.core_models import Panel
from models.database_models import MaterialDatabase
from processing.integrated_processor import IntegratedProcessor, DimensionInput
from utils.session_manager import (
    initialize_session_state,
    add_panel,
    remove_panel,
    get_material_summary,
    get_cached_material_options,
    get_cached_thickness_options
)
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="Panel Input - Steel Cutting Optimizer",
    page_icon="🔧",
    layout="wide"
)

# Initialize session
initialize_session_state()

@st.cache_resource
def get_database():
    """Get database instance with caching"""
    try:
        return MaterialDatabase()
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return None

def get_pi_codes_list():
    """Get PI codes list for selection"""
    db = get_database()
    if db:
        try:
            pi_codes = db.get_pi_codes_list()
            return [""] + pi_codes  # Add empty option first
        except Exception:
            return [""]
    return [""]

def render_pi_code_selector(key_prefix: str, current_value: str = ""):
    """Render PI code selector with search"""
    pi_codes = get_pi_codes_list()

    # Find current index
    try:
        current_index = pi_codes.index(current_value) if current_value in pi_codes else 0
    except ValueError:
        current_index = 0

    selected_pi = st.selectbox(
        "PIコード",
        options=pi_codes,
        index=current_index,
        key=f"{key_prefix}_pi_code",
        help="PIコードを選択してください（空白で寸法展開なし）"
    )

    # Show expansion info if PI code is selected
    if selected_pi and selected_pi != "":
        db = get_database()
        if db:
            pi_data = db.get_pi_code(selected_pi)
            if pi_data:
                st.caption(f"展開: W+{pi_data.width_expansion:.0f}mm, H+{pi_data.height_expansion:.0f}mm")

    return selected_pi

def main():
    """Main panel input page"""
    
    # Render sidebar
    render_sidebar()
    
    # Custom CSS for Material Design styling
    st.markdown("""
    <style>
    .panel-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2196F3;
    }
    
    .input-section {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-panel { border-left-color: #4CAF50; }
    .warning-panel { border-left-color: #FF9800; }
    .error-panel { border-left-color: #F44336; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2196F3, #1976D2); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🔧 Panel Input</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Add panels manually or import from CSV/Excel files
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method selection
    tab1, tab2, tab3 = st.tabs(["➕ Manual Input", "📁 File Import", "📋 Current Panels"])
    
    with tab1:
        render_manual_input()
    
    with tab2:
        render_file_import()
    
    with tab3:
        render_current_panels()

def render_manual_input():
    """Render grid-based manual panel input"""
    st.markdown("### Grid Input - Add Multiple Panels")

    # Initialize grid data in session state
    if 'panel_grid' not in st.session_state:
        # Pre-fill with user's example and some common cases
        st.session_state.panel_grid = [
            {
                'id': 'Panel_968x712',
                'width': 968,
                'height': 712,
                'quantity': 12,
                'material': 'SGCC',
                'thickness': 6.0,
                'priority': 5,
                'rotation': True,
                'pi_code': ''
            },
            {
                'id': 'Panel_750x800',
                'width': 750,
                'height': 800,
                'quantity': 6,
                'material': 'SGCC',
                'thickness': 6.0,
                'priority': 5,
                'rotation': True,
                'pi_code': ''
            },
            {
                'id': 'Panel_500x600',
                'width': 500,
                'height': 600,
                'quantity': 8,
                'material': 'SPCC',
                'thickness': 4.5,
                'priority': 3,
                'rotation': True,
                'pi_code': ''
            }
        ]

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ Add Row", help="Add new panel row"):
            st.session_state.panel_grid.append({
                'id': f'Panel_{len(st.session_state.panel_grid) + 1}',
                'width': 968,
                'height': 712,
                'quantity': 4,
                'material': 'SGCC',
                'thickness': 6.0,
                'priority': 5,
                'rotation': True,
                'pi_code': ''
            })
            st.experimental_rerun()

    with col2:
        if st.button("🗑️ Remove Last", help="Remove last row"):
            if len(st.session_state.panel_grid) > 0:
                st.session_state.panel_grid.pop()
                st.experimental_rerun()

    with col3:
        if st.button("🔄 Reset Grid", help="Reset to default examples"):
            st.session_state.panel_grid = [
                {
                    'id': 'Panel_968x712',
                    'width': 968,
                    'height': 712,
                    'quantity': 12,
                    'material': 'SGCC',
                    'thickness': 6.0,
                    'priority': 5,
                    'rotation': True,
                    'pi_code': ''
                }
            ]
            st.experimental_rerun()

    with col4:
        if st.button("📋 Load Template", help="Load common panel templates"):
            render_template_selector()

    # File input section
    st.markdown("---")
    st.markdown("#### 📁 File Input")
    render_file_input()

    # Grid input interface
    st.markdown("#### 📊 Panel Input Grid")
    st.markdown("Edit the table below to define multiple panels at once:")

    # Create editable grid using columns
    if st.session_state.panel_grid:
        # Headers
        header_cols = st.columns([2, 1.2, 1.2, 1, 1.5, 1, 1, 1, 1.5])
        headers = ['Panel ID', 'Width(mm)', 'Height(mm)', 'Qty', 'Material', 'Thick(mm)', 'Priority', 'Rotate', 'PI Code']

        for i, header in enumerate(headers):
            header_cols[i].markdown(f"**{header}**")

        # Editable rows
        for idx, panel_data in enumerate(st.session_state.panel_grid):
            cols = st.columns([2, 1.2, 1.2, 1, 1.5, 1, 1, 1, 1.5])

            with cols[0]:
                panel_data['id'] = st.text_input(
                    "ID",
                    value=panel_data['id'],
                    key=f"id_{idx}",
                    label_visibility="collapsed"
                )

            with cols[1]:
                panel_data['width'] = st.number_input(
                    "Width",
                    min_value=50,
                    max_value=1500,
                    value=panel_data['width'],
                    key=f"width_{idx}",
                    label_visibility="collapsed"
                )

            with cols[2]:
                panel_data['height'] = st.number_input(
                    "Height",
                    min_value=50,
                    max_value=3100,
                    value=panel_data['height'],
                    key=f"height_{idx}",
                    label_visibility="collapsed"
                )

            with cols[3]:
                panel_data['quantity'] = st.number_input(
                    "Qty",
                    min_value=1,
                    max_value=1000,
                    value=panel_data['quantity'],
                    key=f"qty_{idx}",
                    label_visibility="collapsed"
                )

            with cols[4]:
                material_options = get_cached_material_options()
                try:
                    current_idx = material_options.index(panel_data['material'])
                except ValueError:
                    current_idx = 0

                panel_data['material'] = st.selectbox(
                    "Material",
                    options=material_options,
                    index=current_idx,
                    key=f"material_{idx}",
                    label_visibility="collapsed"
                )

            with cols[5]:
                thickness_options = get_cached_thickness_options()
                try:
                    current_idx = thickness_options.index(panel_data['thickness'])
                except ValueError:
                    current_idx = 5  # Default to 6.0mm

                panel_data['thickness'] = st.selectbox(
                    "Thickness",
                    options=thickness_options,
                    index=current_idx,
                    key=f"thickness_{idx}",
                    label_visibility="collapsed"
                )

            with cols[6]:
                panel_data['priority'] = st.slider(
                    "Priority",
                    min_value=1,
                    max_value=10,
                    value=panel_data['priority'],
                    key=f"priority_{idx}",
                    label_visibility="collapsed"
                )

            with cols[7]:
                panel_data['rotation'] = st.checkbox(
                    "Rotate",
                    value=panel_data['rotation'],
                    key=f"rotation_{idx}",
                    label_visibility="collapsed"
                )

            with cols[8]:
                # Get PI codes list
                pi_codes_options = get_pi_codes_list()
                current_pi_index = 0
                if panel_data['pi_code'] in pi_codes_options:
                    current_pi_index = pi_codes_options.index(panel_data['pi_code'])

                panel_data['pi_code'] = st.selectbox(
                    "PI Code",
                    options=pi_codes_options,
                    index=current_pi_index,
                    key=f"pi_code_{idx}",
                    label_visibility="collapsed",
                    help="寸法展開用PIコード"
                )

    # Summary and action buttons
    st.markdown("---")

    # Summary statistics
    if st.session_state.panel_grid:
        total_panels = len(st.session_state.panel_grid)
        total_quantity = sum(row['quantity'] for row in st.session_state.panel_grid)
        total_area = sum(row['width'] * row['height'] * row['quantity'] for row in st.session_state.panel_grid) / 1_000_000

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Panel Types", total_panels)
        with col2:
            st.metric("Total Pieces", total_quantity)
        with col3:
            st.metric("Total Area", f"{total_area:.1f} m²")
        with col4:
            materials = set(row['material'] for row in st.session_state.panel_grid)
            st.metric("Materials", len(materials))

    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.session_state.panel_grid:
            # Show preview of first few panels
            preview_text = "Preview: "
            for i, row in enumerate(st.session_state.panel_grid[:3]):
                preview_text += f"{row['id']}({row['width']}×{row['height']}×{row['quantity']})"
                if i < min(2, len(st.session_state.panel_grid) - 1):
                    preview_text += ", "

            if len(st.session_state.panel_grid) > 3:
                preview_text += f" ... and {len(st.session_state.panel_grid) - 3} more"

            st.info(preview_text)

    with col2:
        if st.button("🚀 Add All Panels", type="primary", help="Add all panels in grid to workspace"):
            add_panels_from_grid()

    with col3:
        if st.button("💾 Save Template", help="Save current grid as template"):
            save_grid_as_template()

def render_template_selector():
    """Render template selection popup"""
    with st.expander("📋 Panel Templates", expanded=True):

        templates = {
            "User's 968×712 Case": [
                {'id': 'Panel_968x712', 'width': 968, 'height': 712, 'quantity': 12, 'material': 'SGCC', 'thickness': 6.0, 'priority': 5, 'rotation': True, 'pi_code': ''}
            ],
            "Mixed Size Batch": [
                {'id': 'Large_Panel', 'width': 1200, 'height': 800, 'quantity': 4, 'material': 'SGCC', 'thickness': 6.0, 'priority': 8, 'rotation': True, 'pi_code': ''},
                {'id': 'Medium_Panel', 'width': 800, 'height': 600, 'quantity': 8, 'material': 'SGCC', 'thickness': 6.0, 'priority': 5, 'rotation': True, 'pi_code': ''},
                {'id': 'Small_Panel', 'width': 400, 'height': 300, 'quantity': 16, 'material': 'SGCC', 'thickness': 6.0, 'priority': 3, 'rotation': True, 'pi_code': ''}
            ],
            "Multi-Material": [
                {'id': 'SGCC_Panel', 'width': 750, 'height': 750, 'quantity': 6, 'material': 'SGCC', 'thickness': 6.0, 'priority': 5, 'rotation': True, 'pi_code': ''},
                {'id': 'SPCC_Panel', 'width': 600, 'height': 900, 'quantity': 4, 'material': 'SPCC', 'thickness': 4.5, 'priority': 7, 'rotation': True, 'pi_code': ''},
                {'id': 'SS400_Panel', 'width': 500, 'height': 1000, 'quantity': 8, 'material': 'SS400', 'thickness': 8.0, 'priority': 6, 'rotation': True, 'pi_code': ''}
            ]
        }

        selected_template = st.selectbox(
            "Choose Template",
            options=list(templates.keys()),
            help="Select a predefined panel template"
        )

        if st.button("📥 Load Template", key="load_template_btn"):
            st.session_state.panel_grid = templates[selected_template].copy()
            st.success(f"✅ Loaded template: {selected_template}")
            st.experimental_rerun()

def add_panels_from_grid():
    """Add all panels from grid to main panel list"""
    if not st.session_state.panel_grid:
        st.warning("No panels in grid to add")
        return

    added_count = 0
    error_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, panel_data in enumerate(st.session_state.panel_grid):
        try:
            new_panel = Panel(
                id=panel_data['id'],
                width=panel_data['width'],
                height=panel_data['height'],
                quantity=panel_data['quantity'],
                material=panel_data['material'],
                thickness=panel_data['thickness'],
                priority=panel_data['priority'],
                allow_rotation=panel_data['rotation'],
                pi_code=panel_data['pi_code'] if panel_data['pi_code'] else ""
            )

            if add_panel(new_panel):
                added_count += 1
            else:
                error_count += 1

        except Exception as e:
            error_count += 1
            st.error(f"Panel {panel_data['id']}: {str(e)}")

        # Update progress
        progress_bar.progress((i + 1) / len(st.session_state.panel_grid))
        status_text.text(f"Processing panel {i + 1} of {len(st.session_state.panel_grid)}")

    # Results
    if added_count > 0:
        st.success(f"✅ Successfully added {added_count} panels to workspace")

        # Clear the grid after successful addition
        st.session_state.panel_grid = []

    if error_count > 0:
        st.warning(f"⚠️ {error_count} panels failed to add")

    st.experimental_rerun()

def render_file_input():
    """Render file input interface for text/CSV files"""
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "テキストファイル/CSVファイルをアップロード",
            type=['txt', 'csv', 'tsv'],
            help="パネルデータを含むテキストファイルまたはCSVファイルを選択してください"
        )

    with col2:
        file_format = st.selectbox(
            "ファイル形式",
            options=['Auto-detect', 'CSV', 'TSV (Tab)', 'Custom Delimiter'],
            help="ファイルの区切り文字形式を選択"
        )

    if uploaded_file is not None:
        try:
            # Read file content
            content = uploaded_file.read()

            # Try to decode with different encodings
            text_content = None
            for encoding in ['utf-8', 'shift-jis', 'cp932', 'utf-8-sig']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text_content is None:
                st.error("ファイルの文字エンコードを認識できませんでした")
                return

            # Show file preview
            with st.expander("📋 ファイルプレビュー", expanded=False):
                lines = text_content.split('\n')[:10]  # Show first 10 lines
                st.text('\n'.join(lines))
                if len(text_content.split('\n')) > 10:
                    st.caption(f"... and {len(text_content.split('\n')) - 10} more lines")

            # Parse based on format
            parsed_data = parse_text_file(text_content, file_format)

            if parsed_data:
                st.success(f"✅ {len(parsed_data)}行のデータを解析しました")

                # Show parsed data preview
                if len(parsed_data) > 0:
                    df_preview = pd.DataFrame(parsed_data[:5])  # Show first 5 rows
                    st.dataframe(df_preview, use_container_width=True)

                    if len(parsed_data) > 5:
                        st.caption(f"... and {len(parsed_data) - 5} more rows")

                # Load into grid button
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 グリッドに読み込み", key="load_file_to_grid"):
                        load_file_data_to_grid(parsed_data)
                        st.success("✅ データをグリッドに読み込みました")
                        st.experimental_rerun()

                with col2:
                    if st.button("🚀 直接最適化実行", key="direct_optimize_file"):
                        execute_direct_optimization(parsed_data)
            else:
                st.error("ファイルの解析に失敗しました")

        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")

def parse_text_file(content: str, format_type: str) -> List[Dict[str, Any]]:
    """Parse text file content into panel data"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    if not lines:
        return []

    # Determine delimiter
    delimiter = ','
    if format_type == 'TSV (Tab)':
        delimiter = '\t'
    elif format_type == 'Custom Delimiter':
        # Try to auto-detect common delimiters
        sample_line = lines[0] if lines else ""
        if '\t' in sample_line and sample_line.count('\t') >= 3:
            delimiter = '\t'
        elif ';' in sample_line and sample_line.count(';') >= 3:
            delimiter = ';'
        elif '|' in sample_line and sample_line.count('|') >= 3:
            delimiter = '|'
    elif format_type == 'Auto-detect':
        # Auto-detect delimiter
        sample_line = lines[0] if lines else ""
        if '\t' in sample_line and sample_line.count('\t') >= 3:
            delimiter = '\t'
        elif ';' in sample_line and sample_line.count(';') >= 3:
            delimiter = ';'
        elif '|' in sample_line and sample_line.count('|') >= 3:
            delimiter = '|'
        # Default to comma

    parsed_data = []
    header_processed = False

    for line_num, line in enumerate(lines):
        if not line:
            continue

        parts = [part.strip() for part in line.split(delimiter)]

        # Skip header line if it contains text headers
        if not header_processed:
            if any(part.lower() in ['id', 'width', 'height', 'panel', 'quantity', 'material'] for part in parts):
                header_processed = True
                continue
            header_processed = True

        # Need at least 3 parts: width, height, quantity (minimum)
        if len(parts) < 3:
            continue

        try:
            # Parse panel data - flexible column mapping
            panel_data = {
                'id': parts[0] if len(parts) > 0 and not parts[0].replace('.', '').isdigit() else f"Panel_{line_num}",
                'width': 0,
                'height': 0,
                'quantity': 1,
                'material': 'SGCC',
                'thickness': 6.0,
                'priority': 5,
                'rotation': True,
                'pi_code': ''
            }

            # Try to identify numeric columns
            numeric_parts = []
            for part in parts:
                try:
                    numeric_parts.append(float(part))
                except ValueError:
                    numeric_parts.append(None)

            # Map based on number of numeric columns
            valid_numbers = [n for n in numeric_parts if n is not None]

            if len(valid_numbers) >= 2:
                # First two numbers are width and height
                panel_data['width'] = int(valid_numbers[0])
                panel_data['height'] = int(valid_numbers[1])

                if len(valid_numbers) >= 3:
                    panel_data['quantity'] = int(valid_numbers[2])

                if len(valid_numbers) >= 4:
                    panel_data['thickness'] = valid_numbers[3]

            # Try to find material in text parts
            for part in parts:
                if isinstance(part, str) and len(part) > 1 and not part.isdigit():
                    if any(mat in part.upper() for mat in ['SGCC', 'SPCC', 'SS400', 'SUS', 'SECC', 'SE/E']):
                        panel_data['material'] = part
                        break
                    elif part not in [panel_data['id']] and len(part) < 20:  # Likely material
                        panel_data['material'] = part
                        break

            # Try to find PI code (usually numeric code > 8 digits)
            for part in parts:
                if isinstance(part, str) and part.isdigit() and len(part) >= 8:
                    panel_data['pi_code'] = part
                    break

            # Validate minimum requirements
            if panel_data['width'] >= 50 and panel_data['height'] >= 50:
                parsed_data.append(panel_data)

        except (ValueError, IndexError) as e:
            # Skip lines that can't be parsed
            continue

    return parsed_data

def load_file_data_to_grid(parsed_data: List[Dict[str, Any]]):
    """Load parsed file data into the panel grid"""
    st.session_state.panel_grid = parsed_data.copy()

def execute_direct_optimization(parsed_data: List[Dict[str, Any]]):
    """Execute optimization directly from file data"""
    try:
        processor = IntegratedProcessor()

        # Convert to DimensionInput format
        input_data = []
        for i, panel_data in enumerate(parsed_data):
            input_item = DimensionInput(
                panel_id=panel_data.get('id', f'panel_{i+1}'),
                pi_code=panel_data.get('pi_code', ''),
                finished_width=float(panel_data['width']),
                finished_height=float(panel_data['height']),
                quantity=int(panel_data.get('quantity', 1)),
                material_type=panel_data.get('material', 'SGCC'),
                thickness=float(panel_data.get('thickness', 6.0)),
                priority=int(panel_data.get('priority', 5)),
                block_order=1
            )
            input_data.append(input_item)

        # Execute processing
        with st.spinner("最適化処理を実行中..."):
            result = processor.process_complete_workflow(input_data)

        # Show results
        report = processor.get_processing_report(result)

        st.success("✅ 最適化処理が完了しました")

        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総合効率", report['summary']['overall_efficiency'])
        with col2:
            st.metric("使用シート数", report['summary']['total_sheets_used'])
        with col3:
            st.metric("処理時間", report['summary']['processing_time'])
        with col4:
            if report['summary']['waste_reduction'] != "N/A":
                st.metric("歩留まり向上", report['summary']['waste_reduction'])

        # Store results in session state for further analysis
        st.session_state['optimization_result'] = result
        st.session_state['optimization_report'] = report

        st.info("📊 詳細な結果は「最適化結果」ページでご確認いただけます")

    except Exception as e:
        st.error(f"最適化処理エラー: {e}")

def save_grid_as_template():
    """Save current grid as a custom template"""
    if not st.session_state.panel_grid:
        st.warning("No panels in grid to save")
        return

    # This could be enhanced to save to a file or database
    # For now, just show the JSON that could be saved
    import json

    template_data = {
        'name': f'Custom_Template_{len(st.session_state.panel_grid)}_panels',
        'created': str(pd.Timestamp.now()),
        'panels': st.session_state.panel_grid
    }

    st.download_button(
        label="💾 Download Template JSON",
        data=json.dumps(template_data, indent=2, ensure_ascii=False),
        file_name=f"panel_template_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def render_file_import():
    """Render file import interface"""
    st.markdown("### Import from File")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload panel data in CSV or Excel format"
    )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
            
            # Show preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping
            st.markdown("#### Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            available_columns = df.columns.tolist()
            
            with col1:
                id_col = st.selectbox("Panel ID Column", options=available_columns, index=0)
                width_col = st.selectbox("Width Column", options=available_columns, index=1 if len(available_columns) > 1 else 0)
                height_col = st.selectbox("Height Column", options=available_columns, index=2 if len(available_columns) > 2 else 0)
            
            with col2:
                quantity_col = st.selectbox("Quantity Column", options=available_columns, index=3 if len(available_columns) > 3 else 0)
                material_col = st.selectbox("Material Column", options=available_columns, index=4 if len(available_columns) > 4 else 0)
                thickness_col = st.selectbox("Thickness Column", options=available_columns, index=5 if len(available_columns) > 5 else 0)
            
            with col3:
                default_material = st.selectbox("Default Material", options=get_cached_material_options(), index=0)
                default_thickness = st.selectbox("Default Thickness", options=get_cached_thickness_options(), index=5)
                allow_rotation_default = st.checkbox("Default Allow Rotation", value=True)
            
            # Import button
            if st.button("📥 Import Panels", type="primary"):
                imported_count = 0
                error_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for index, row in df.iterrows():
                    try:
                        # Extract values with defaults
                        panel_data = {
                            'id': str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else f"Imported_{index+1}",
                            'width': float(row[width_col]) if width_col in row and pd.notna(row[width_col]) else 0,
                            'height': float(row[height_col]) if height_col in row and pd.notna(row[height_col]) else 0,
                            'quantity': int(row[quantity_col]) if quantity_col in row and pd.notna(row[quantity_col]) else 1,
                            'material': str(row[material_col]) if material_col in row and pd.notna(row[material_col]) else default_material,
                            'thickness': float(row[thickness_col]) if thickness_col in row and pd.notna(row[thickness_col]) else default_thickness,
                            'allow_rotation': allow_rotation_default
                        }
                        
                        # Create and add panel
                        new_panel = Panel(**panel_data)
                        if add_panel(new_panel):
                            imported_count += 1
                        else:
                            error_count += 1
                    
                    except Exception as e:
                        error_count += 1
                        st.error(f"Row {index+1}: {str(e)}")
                    
                    # Update progress
                    progress_bar.progress((index + 1) / len(df))
                    status_text.text(f"Processing row {index + 1} of {len(df)}")
                
                # Results
                if imported_count > 0:
                    st.success(f"✅ Successfully imported {imported_count} panels")
                
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} panels failed to import")
                
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Failed to read file: {str(e)}")
    
    # Template download
    st.markdown("#### Template Files")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV template
        csv_template = pd.DataFrame({
            'id': ['Panel_1', 'Panel_2', 'Panel_3'],
            'width': [968, 750, 500],
            'height': [712, 800, 600],
            'quantity': [4, 6, 8],
            'material': ['SGCC', 'SGCC', 'SPCC'],
            'thickness': [6.0, 6.0, 4.5]
        })
        
        csv_buffer = io.StringIO()
        csv_template.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📁 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="panel_template.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel template
        excel_buffer = io.BytesIO()
        csv_template.to_excel(excel_buffer, index=False)
        
        st.download_button(
            "📊 Download Excel Template", 
            data=excel_buffer.getvalue(),
            file_name="panel_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def render_current_panels():
    """Render current panels management"""
    st.markdown("### Current Panels")
    
    if not st.session_state.panels:
        st.info("No panels added yet. Use the Manual Input or File Import tabs to add panels.")
        return
    
    # Summary statistics
    summary = get_material_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Panel Types", len(st.session_state.panels))
    
    with col2:
        total_quantity = sum(p.quantity for p in st.session_state.panels)
        st.metric("Total Pieces", total_quantity)
    
    with col3:
        st.metric("Materials", len(summary))
    
    with col4:
        total_area = sum(p.cutting_area * p.quantity for p in st.session_state.panels) / 1_000_000
        st.metric("Total Area", f"{total_area:.1f} m²")
    
    # Material breakdown
    st.markdown("#### By Material Type")
    
    for material, data in summary.items():
        with st.expander(f"📐 {material} - {data['panel_count']} types, {data['total_quantity']} pieces"):
            material_panels = [p for p in st.session_state.panels if p.material == material]
            
            # Create DataFrame for display
            panel_data = []
            for panel in material_panels:
                panel_data.append({
                    'ID': panel.id,
                    'Size (mm)': f"{panel.width}×{panel.height}",
                    'Quantity': panel.quantity,
                    'Area (m²)': f"{panel.cutting_area * panel.quantity / 1_000_000:.2f}",
                    'Thickness': f"{panel.thickness}mm",
                    'Priority': panel.priority,
                    'Rotation': "Yes" if panel.allow_rotation else "No"
                })
            
            df = pd.DataFrame(panel_data)
            st.dataframe(df, use_container_width=True)
            
            # Actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🗑️ Clear {material} Panels", key=f"clear_{material}"):
                    # Remove all panels of this material
                    st.session_state.panels = [p for p in st.session_state.panels if p.material != material]
                    st.success(f"Cleared all {material} panels")
                    st.experimental_rerun()
            
            with col2:
                # Export material panels
                csv_data = df.to_csv(index=False)
                st.download_button(
                    f"📁 Export {material}",
                    data=csv_data,
                    file_name=f"{material}_panels.csv",
                    mime="text/csv",
                    key=f"export_{material}"
                )

if __name__ == "__main__":
    main()