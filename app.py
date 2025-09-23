"""
Steel Cutting Optimization System - Dashboard Homepage
鋼板切断最適化システム - ダッシュボードホームページ

Main dashboard for the steel cutting optimization system with system overview
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import core modules
from core.material_manager import get_material_manager


def setup_page_config():
    """Configure Streamlit page settings for dashboard"""
    st.set_page_config(
        page_title="Steel Cutting System Dashboard",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Enhanced CSS for dashboard
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .metric-item {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .quick-action-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 2rem;
        border-radius: 25px;
        border: none;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s;
    }
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .recent-activity {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .navigation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .navigation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)


def render_dashboard_header():
    """Render enhanced dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1>🏠 鋼板切断最適化システム ダッシュボード</h1>
        <h2>Steel Cutting Optimization System Dashboard</h2>
        <p>ギロチンカット制約下での2Dビンパッキング最適化システム</p>
        <p>2D bin packing optimization with guillotine cut constraints</p>
    </div>
    """, unsafe_allow_html=True)


def render_system_overview():
    """Render system overview and metrics"""
    material_manager = get_material_manager()
    summary = material_manager.get_inventory_summary()

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📊 システム概要 / System Overview")

    # System metrics - simplified without cost/value data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <h3 style="color: #667eea; margin: 0;">材料在庫</h3>
            <h2 style="margin: 0;">{summary['total_sheets']}</h2>
            <p style="margin: 0; color: #666;">Total Materials</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <h3 style="color: #28a745; margin: 0;">材質種類</h3>
            <h2 style="margin: 0;">{summary['material_types']}</h2>
            <p style="margin: 0; color: #666;">Material Types</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        recent_optimizations = len(st.session_state.get('optimization_results', []))
        st.markdown(f"""
        <div class="metric-item">
            <h3 style="color: #dc3545; margin: 0;">最近の最適化</h3>
            <h2 style="margin: 0;">{recent_optimizations}</h2>
            <p style="margin: 0; color: #666;">Recent Optimizations</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_material_overview():
    """Render material inventory overview"""
    material_manager = get_material_manager()

    if not material_manager.inventory:
        st.markdown("""
        <div class="dashboard-card">
            <h3>⚠️ 材料在庫が空です / Material Inventory is Empty</h3>
            <p>システムを使用するには、まず材料管理ページで材料を追加してください。</p>
            <p>Please add materials via the Material Management page to start using the system.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    summary = material_manager.get_inventory_summary()

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📦 材料在庫概要 / Material Inventory Overview")

    # Material breakdown visualization
    if summary['by_material_type']:
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
            # Bar chart for area distribution
            fig_bar = px.bar(
                x=list(summary['by_material_type'].keys()),
                y=[data['total_area'] for data in summary['by_material_type'].values()],
                title="材質別面積 / Area by Material Type",
                labels={'x': 'Material Type', 'y': 'Total Area (mm²)'},
                color_discrete_sequence=['#667eea']
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Material summary table
        st.subheader("📋 材質別サマリー / Material Summary")
        breakdown_data = []
        for material_type, data in summary['by_material_type'].items():
            breakdown_data.append({
                '材質 / Material Type': material_type,
                '数量 / Count': data['count'],
                '総面積 / Total Area (mm²)': f"{data['total_area']:,.0f}",
                '平均面積 / Avg Area (mm²)': f"{data['total_area']/data['count']:,.0f}"
            })

        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons"""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🚀 クイックアクション / Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔧 切断最適化を開始 / Start Cutting Optimization",
                    use_container_width=True, type="primary"):
            st.switch_page("pages/1_🔧_Cutting_Optimization.py")

    with col2:
        if st.button("📦 材料管理 / Manage Materials",
                    use_container_width=True):
            st.switch_page("pages/2_📦_Material_Management.py")

    st.markdown('</div>', unsafe_allow_html=True)


def render_recent_activity():
    """Render recent activity section"""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("📈 最近のアクティビティ / Recent Activity")

    # Check for recent optimization results
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        results = st.session_state.optimization_results
        total_panels = sum(len(result.panels) for result in results)
        avg_efficiency = sum(result.efficiency for result in results) / len(results)
        total_cost = sum(result.cost for result in results)

        st.markdown(f"""
        <div class="recent-activity">
            <h4>✅ 最新の最適化結果 / Latest Optimization Result</h4>
            <p><strong>シート数:</strong> {len(results)} | <strong>パネル数:</strong> {total_panels} |
            <strong>平均効率:</strong> {avg_efficiency:.1%} | <strong>コスト:</strong> ¥{total_cost:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("まだ最適化を実行していません。切断最適化ページで開始してください。")
        st.info("No optimizations run yet. Start with the Cutting Optimization page.")

    # System status
    material_manager = get_material_manager()
    inventory_status = "正常" if len(material_manager.inventory) > 0 else "要設定"
    status_color = "#28a745" if len(material_manager.inventory) > 0 else "#dc3545"

    st.markdown(f"""
    <div class="recent-activity">
        <h4>🔧 システム状態 / System Status</h4>
        <p><strong>材料在庫:</strong> <span style="color: {status_color}">{inventory_status}</span> |
        <strong>最終更新:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_features_overview():
    """Render system features overview"""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🎯 システム機能 / System Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 切断最適化</h3>
            <p>ギロチンカット制約下での高効率2Dビンパッキング</p>
            <small>High-efficiency 2D bin packing with guillotine constraints</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📦 材料管理</h3>
            <p>包括的な材料在庫管理とコスト追跡</p>
            <small>Comprehensive material inventory management and cost tracking</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 可視化</h3>
            <p>インタラクティブな切断レイアウト表示</p>
            <small>Interactive cutting layout visualization</small>
        </div>
        """, unsafe_allow_html=True)

    # Key benefits
    st.markdown("#### 🌟 主な利点 / Key Benefits")
    benefits = [
        "✅ 材料効率向上 (10-30% waste reduction)",
        "✅ 作業時間短縮 (Optimized cutting sequences)",
        "✅ コスト削減 (Material cost optimization)",
        "✅ 品質向上 (Precision cutting plans)"
    ]

    col1, col2 = st.columns(2)
    for i, benefit in enumerate(benefits):
        if i % 2 == 0:
            col1.markdown(benefit)
        else:
            col2.markdown(benefit)

    st.markdown('</div>', unsafe_allow_html=True)


def render_system_info():
    """Render system information"""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ システム情報 / System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📋 対応データ形式 / Supported Data Formats:**
        - TSV (Tab-separated values)
        - CSV (Comma-separated values)
        - 手動入力 / Manual input
        - ファイルアップロード / File upload
        """)

    with col2:
        st.markdown("""
        **⚙️ 最適化アルゴリズム / Optimization Algorithms:**
        - FFD (First Fit Decreasing)
        - BFD (Best Fit Decreasing)
        - ハイブリッド最適化 / Hybrid optimization
        """)

    st.markdown("""
    **🎯 対象製造業 / Target Industries:**
    鋼板加工、金属加工、建材製造、自動車部品、家電製造
    Steel processing, Metal fabrication, Construction materials, Automotive parts, Appliance manufacturing
    """)

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main dashboard function"""
    setup_page_config()
    render_dashboard_header()

    # Quick navigation
    render_quick_actions()

    # Main dashboard content
    col1, col2 = st.columns([2, 1])

    with col1:
        render_system_overview()
        render_material_overview()

    with col2:
        render_recent_activity()

    # Full width sections
    render_features_overview()

    # Footer
    st.markdown("---")
    render_system_info()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>© 2024 Steel Cutting Optimization System | Built with Streamlit</p>
        <p>鋼板切断最適化システム | Streamlitで構築</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()