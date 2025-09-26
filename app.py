#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steel Cutting Optimization System - Main Application
鋼板切断最適化システム - メインアプリケーション

A Streamlit-based application for optimizing steel panel cutting operations
with guillotine cut constraints.
"""

# Standard library imports
import traceback

# Third-party imports
import streamlit as st

# Local imports
from ui.common_styles import get_common_css
from ui.page_headers import get_page_config, render_unified_header

# Configure page
st.set_page_config(
    page_title="Steel Cutting Optimization System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply unified styling
st.markdown(get_common_css(), unsafe_allow_html=True)

config = get_page_config("cutting_optimization")
render_unified_header(
    title_ja="🔧 鋼板切断最適化システム",
    title_en="Steel Cutting Optimization System",
    description="Streamlit マルチページアプリケーション - 左サイドバーからページを選択してください",
    icon="🏠"
)

# Main content
st.markdown("---")

# Quick navigation
st.markdown("## 📍 ページナビゲーション")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🔧 切断最適化", use_container_width=True):
        st.switch_page("pages/1_🔧_Cutting_Optimization.py")

with col2:
    if st.button("📦 材料管理", use_container_width=True):
        st.switch_page("pages/2_📦_Material_Management.py")

with col3:
    if st.button("⚙️ PIコード管理", use_container_width=True):
        st.switch_page("pages/3_⚙️_PI_Management.py")

with col4:
    if st.button("📊 分析結果", use_container_width=True):
        st.switch_page("pages/4_📊_Analysis_Results.py")

with col5:
    if st.button("💾 データ管理", use_container_width=True):
        st.switch_page("pages/3_💾_Data_Management.py")

# System status
st.markdown("---")
st.markdown("## 🔧 システム状態")

# Quick status check
try:
    from core.persistence_adapter import get_persistence_adapter
    from core.material_manager import get_material_manager

    persistence = get_persistence_adapter()
    status = persistence.get_system_status()
    material_manager = get_material_manager()

    col1, col2, col3 = st.columns(3)

    with col1:
        db_status = "🟢 接続中" if status.get('database_available') else "🔴 接続エラー"
        st.markdown(f"**データベース**: {db_status}")

    with col2:
        material_count = len(material_manager.inventory)
        st.markdown(f"**材料在庫**: {material_count}種類")

    with col3:
        history_count = len(persistence.get_optimization_history()) if status.get('database_available') else 0
        st.markdown(f"**最適化履歴**: {history_count}件")

except ImportError as e:
    st.error(f"モジュール読み込みエラー / Module import error: {e}")
    st.info("必要な依存関係がインストールされていない可能性があります")
except ConnectionError as e:
    st.warning(f"データベース接続エラー / Database connection error: {e}")
    st.info("データベースサーバーの状態を確認してください")
except (AttributeError, KeyError) as e:
    st.warning(f"設定エラー / Configuration error: {e}")
    st.info("システム設定を確認してください")
except Exception as e:
    st.error(f"システム状態取得エラー / System status error: {e}")
    with st.expander("詳細ログ / Error Details"):
        st.code(str(e))
        st.code(traceback.format_exc())

# Usage instructions
st.markdown("---")
st.markdown("## 📖 使用方法")

with st.expander("システム概要", expanded=False):
    st.markdown("""
    このシステムは鋼板切断の最適化を行うStreamlitアプリケーションです。

    **主な機能:**
    - ギロチンカット制約下での2Dビンパッキング最適化
    - SQLiteデータベースによる永続化
    - 統一されたUIデザインシステム
    - 複数アルゴリズム対応（FFD, BFD, GENETIC, HYBRID）

    **使用手順:**
    1. 左サイドバーから「切断最適化」ページを選択
    2. パネルデータを入力または材料管理で事前設定
    3. 最適化を実行して結果を確認
    """)