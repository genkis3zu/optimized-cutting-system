"""
Unified Metric Cards for Steel Cutting Optimization System
鋼板切断最適化システム統一メトリックカード

Consistent metric display components across all pages
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from ui.common_styles import get_metric_colors

def render_metric_row(metrics: List[Dict[str, Any]], columns: int = 3):
    """
    メトリックを行で表示

    Args:
        metrics: メトリック情報のリスト
        columns: カラム数（デフォルト3）

    metrics形式:
    [
        {
            'title': 'タイトル',
            'value': '値',
            'subtitle': 'サブタイトル',
            'color': 'カラー名' (primary/success/warning/error/neutral)
        }
    ]
    """
    colors = get_metric_colors()
    cols = st.columns(columns)

    for i, metric in enumerate(metrics):
        if i >= columns:
            break

        with cols[i]:
            color = colors.get(metric.get('color', 'primary'), colors['primary'])

            st.markdown("""
            <div class="metric-card" style="border-left-color: {color};">
                <h3>{metric['title']}</h3>
                <h2 style="color: {color};">{metric['value']}</h2>
                <p>{metric.get('subtitle', '')}</p>
            </div>
            """, unsafe_allow_html=True)

def render_summary_metrics(total_items: int, active_items: int, total_value: float,
                         currency: str = "¥", item_name: str = "アイテム"):
    """
    汎用サマリーメトリックを表示

    Args:
        total_items: 総アイテム数
        active_items: アクティブアイテム数
        total_value: 総価値
        currency: 通貨記号
        item_name: アイテム名
    """
    metrics = [
        {
            'title': f'総{item_name}数',
            'value': f'{total_items:,}',
            'subtitle': f'Total {item_name}s',
            'color': 'primary'
        },
        {
            'title': f'有効{item_name}数',
            'value': f'{active_items:,}',
            'subtitle': f'Active {item_name}s',
            'color': 'success'
        },
        {
            'title': '総価値',
            'value': f'{currency}{total_value:,.0f}',
            'subtitle': 'Total Value',
            'color': 'warning'
        }
    ]

    render_metric_row(metrics, 3)

def render_optimization_metrics(sheets_used: int, panels_placed: int,
                               efficiency: float, cost: float):
    """
    最適化結果メトリックを表示

    Args:
        sheets_used: 使用シート数
        panels_placed: 配置パネル数
        efficiency: 効率（0-1）
        cost: コスト
    """
    metrics = [
        {
            'title': '使用シート数',
            'value': f'{sheets_used:,}',
            'subtitle': 'Sheets Used',
            'color': 'primary'
        },
        {
            'title': '配置パネル数',
            'value': f'{panels_placed:,}',
            'subtitle': 'Panels Placed',
            'color': 'success'
        },
        {
            'title': '平均効率',
            'value': f'{efficiency:.1%}',
            'subtitle': 'Average Efficiency',
            'color': 'warning'
        },
        {
            'title': '総コスト',
            'value': f'¥{cost:,.0f}',
            'subtitle': 'Total Cost',
            'color': 'error'
        }
    ]

    render_metric_row(metrics, 4)

def render_material_metrics(materials: List[Any]):
    """
    材料在庫メトリックを表示

    Args:
        materials: 材料リスト
    """
    if not materials:
        st.info("材料データがありません")
        return

    total_sheets = sum(m.availability for m in materials)
    material_types = len(set(m.material_type for m in materials))
    total_area = sum(m.area * m.availability for m in materials)

    metrics = [
        {
            'title': '総材料数',
            'value': f'{total_sheets:,}',
            'subtitle': 'Total Sheets',
            'color': 'primary'
        },
        {
            'title': '材質種類',
            'value': f'{material_types}',
            'subtitle': 'Material Types',
            'color': 'success'
        },
        {
            'title': '総面積',
            'value': f'{total_area:,.0f}',
            'subtitle': 'Total Area (mm²)',
            'color': 'warning'
        }
    ]

    render_metric_row(metrics, 3)

def render_status_card(title: str, message: str, status: str = "info",
                      action_button: Optional[Dict] = None):
    """
    ステータスカードを表示

    Args:
        title: タイトル
        message: メッセージ
        status: ステータス（success/warning/error/info）
        action_button: ボタン情報 {'text': 'ボタンテキスト', 'key': 'ボタンキー'}
    """
    status_class = f"status-{status}" if status != "info" else "action-card"

    st.markdown("""
    <div class="{status_class}">
        <h4 style="margin: 0 0 0.5rem 0;">{title}</h4>
        <p style="margin: 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    if action_button:
        if st.button(action_button['text'], key=action_button.get('key', 'action_btn')):
            return True
    return False

def render_progress_metrics(current: int, total: int, label: str = "進捗"):
    """
    進捗メトリックを表示

    Args:
        current: 現在の値
        total: 総値
        label: ラベル
    """
    percentage = (current / total * 100) if total > 0 else 0

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**{label}**: {current:,} / {total:,}")
        st.progress(percentage / 100)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>完了率</h3>
            <h2 style="color: {'#2ca02c' if percentage >= 80 else '#ff7f0e' if percentage >= 50 else '#d62728'};">
                {percentage:.1f}%
            </h2>
            <p>Completion Rate</p>
        </div>
        """, unsafe_allow_html=True)
