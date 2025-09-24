"""
Unified Page Headers for Steel Cutting Optimization System
鋼板切断最適化システム統一ページヘッダー

Consistent header components across all pages
"""

import streamlit as st
from ui.common_styles import get_common_css

def render_unified_header(title_ja: str, title_en: str, description: str = "", icon: str = "⚙️"):
    """
    統一されたページヘッダーをレンダリング

    Args:
        title_ja: 日本語タイトル
        title_en: 英語タイトル
        description: 説明文
        icon: アイコン
    """
    st.markdown(get_common_css(), unsafe_allow_html=True)

    st.markdown("""
    <div class="unified-header">
        <h1>{icon} {title_ja}</h1>
        <h3>{title_en}</h3>
        {f'<p>{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def render_section_header(title: str, subtitle: str = "", color: str = "#1f77b4"):
    """
    セクションヘッダーをレンダリング

    Args:
        title: セクションタイトル
        subtitle: サブタイトル
        color: アクセントカラー
    """
    subtitle_html = f"<p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>{subtitle}</p>" if subtitle else ""

    st.markdown("""
    <div class="content-section">
        <h2 style="color: {color}; border-bottom-color: {color};">{title}</h2>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def render_page_navigation(current_page: str, pages: dict):
    """
    ページナビゲーションをレンダリング

    Args:
        current_page: 現在のページ
        pages: ページ辞書 {'key': 'display_name'}
    """
    st.markdown("### 📍 ナビゲーション")

    cols = st.columns(len(pages))

    for i, (page_key, display_name) in enumerate(pages.items()):
        with cols[i]:
            if page_key == current_page:
                st.markdown(f"**{display_name}** (現在)")
            else:
                if st.button(display_name, key=f"nav_{page_key}", use_container_width=True):
                    st.switch_page(f"pages/{page_key}")

# ページ定義
PAGE_CONFIGS = {
    "cutting_optimization": {
        "title_ja": "切断最適化",
        "title_en": "Cutting Optimization",
        "description": "ギロチンカット制約下での2Dビンパッキング最適化",
        "icon": "🔧"
    },
    "material_management": {
        "title_ja": "材料在庫管理",
        "title_en": "Material Inventory Management",
        "description": "材料在庫の追加・編集・削除と在庫状況の確認",
        "icon": "📦"
    },
    "pi_management": {
        "title_ja": "PIコード管理",
        "title_en": "PI Code Management",
        "description": "展開寸法計算用PIコードの登録・管理",
        "icon": "⚙️"
    },
    "analysis_results": {
        "title_ja": "分析結果",
        "title_en": "Analysis Results",
        "description": "最適化結果の詳細分析とパフォーマンス評価",
        "icon": "📊"
    },
    "data_management": {
        "title_ja": "データ管理",
        "title_en": "Data Management",
        "description": "プロジェクトデータと最適化履歴の管理",
        "icon": "💾"
    }
}

def get_page_config(page_key: str) -> dict:
    """ページ設定を取得"""
    return PAGE_CONFIGS.get(page_key, {
        "title_ja": "システム",
        "title_en": "System",
        "description": "",
        "icon": "⚙️"
    })
