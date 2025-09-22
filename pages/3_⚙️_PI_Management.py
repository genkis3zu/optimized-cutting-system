"""
PI Code Management Page
PI コード管理ページ

Manages PI codes for dimension expansion calculations
展開寸法計算用PIコード管理
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

from core.pi_manager import get_pi_manager, PICode


def setup_page_config():
    """ページ設定を構成"""
    st.set_page_config(
        page_title="PI管理 - Steel Cutting System",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # PI管理用CSS
    st.markdown("""
    <style>
    .pi-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .pi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .pi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .pi-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        text-align: center;
    }
    .expansion-example {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #856404;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_header():
    """ページヘッダーを描画"""
    st.markdown("""
    <div class="pi-header">
        <h1>🔧 PIコード管理 / PI Code Management</h1>
        <p>展開寸法計算用PIコード登録・管理システム</p>
        <p>PI Code Registration & Management for Dimension Expansion</p>
    </div>
    """, unsafe_allow_html=True)


def render_pi_summary():
    """PIコード概要を表示"""
    pi_manager = get_pi_manager()
    summary = pi_manager.get_pi_summary()

    st.markdown('<div class="pi-card">', unsafe_allow_html=True)
    st.subheader("📊 PIコード概要 / PI Code Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="pi-metric">
            <h3 style="color: #667eea; margin: 0;">総PIコード数</h3>
            <h2 style="margin: 0;">{summary['total_codes']}</h2>
            <p style="margin: 0; color: #666;">Total PI Codes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="pi-metric">
            <h3 style="color: #28a745; margin: 0;">裏板付き</h3>
            <h2 style="margin: 0;">{summary['codes_with_backing']}</h2>
            <p style="margin: 0; color: #666;">With Backing</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="pi-metric">
            <h3 style="color: #dc3545; margin: 0;">裏板なし</h3>
            <h2 style="margin: 0;">{summary['codes_without_backing']}</h2>
            <p style="margin: 0; color: #666;">Without Backing</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="pi-metric">
            <h3 style="color: #ffc107; margin: 0;">裏板材質</h3>
            <h2 style="margin: 0;">{summary['unique_backing_materials']}</h2>
            <p style="margin: 0; color: #666;">Material Types</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_dimension_calculator():
    """展開寸法計算機を表示"""
    pi_manager = get_pi_manager()

    st.markdown('<div class="pi-card">', unsafe_allow_html=True)
    st.subheader("🧮 展開寸法計算 / Dimension Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### 入力 / Input")

        # PIコード選択
        pi_codes = pi_manager.get_all_pi_codes()
        if pi_codes:
            selected_pi = st.selectbox(
                "PIコード選択 / Select PI Code",
                pi_codes,
                help="展開計算に使用するPIコードを選択"
            )
        else:
            st.warning("PIコードが登録されていません")
            selected_pi = None

        # 寸法入力
        finished_w = st.number_input(
            "完成W寸法 (mm) / Finished Width",
            min_value=10.0,
            max_value=2000.0,
            value=900.0,
            step=0.5,
            help="曲げ後の完成W寸法"
        )

        finished_h = st.number_input(
            "完成H寸法 (mm) / Finished Height",
            min_value=10.0,
            max_value=3000.0,
            value=600.0,
            step=0.5,
            help="曲げ後の完成H寸法"
        )

    with col2:
        st.write("#### 計算結果 / Calculation Result")

        if selected_pi:
            pi_info = pi_manager.get_pi_code(selected_pi)
            expanded_w, expanded_h = pi_manager.get_expansion_for_panel(
                selected_pi, finished_w, finished_h
            )

            # 計算結果表示
            st.markdown(f"""
            <div class="expansion-example">
                <h4>📐 展開寸法計算結果</h4>
                <table style="width: 100%;">
                    <tr>
                        <td><strong>完成寸法:</strong></td>
                        <td>{finished_w:.1f} × {finished_h:.1f} mm</td>
                    </tr>
                    <tr>
                        <td><strong>展開寸法:</strong></td>
                        <td>{expanded_w:.1f} × {expanded_h:.1f} mm</td>
                    </tr>
                    <tr>
                        <td><strong>W加算:</strong></td>
                        <td>{pi_info.w_expansion:+.1f} mm</td>
                    </tr>
                    <tr>
                        <td><strong>H加算:</strong></td>
                        <td>{pi_info.h_expansion:+.1f} mm</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            # PIコード詳細情報
            st.write("##### PIコード詳細")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**裏板:** {'あり' if pi_info.has_backing else 'なし'}")
                if pi_info.has_backing:
                    st.write(f"**裏板材質:** {pi_info.backing_material}")
                st.write(f"**プラスタ:** {'あり' if pi_info.has_plaster else 'なし'}")
            with col_b:
                st.write(f"**板厚:** {pi_info.thickness}mm")
                if pi_info.description:
                    st.write(f"**備考:** {pi_info.description}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_pi_management():
    """PIコード管理セクションを表示"""
    pi_manager = get_pi_manager()

    st.markdown('<div class="pi-card">', unsafe_allow_html=True)
    st.subheader("⚙️ PIコード管理 / PI Code Management")

    # タブ形式で管理機能を整理
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 PIコード一覧",
        "➕ PIコード追加",
        "✏️ PIコード編集",
        "🗑️ PIコード削除"
    ])

    with tab1:
        render_pi_list(pi_manager)

    with tab2:
        render_add_pi_form(pi_manager)

    with tab3:
        render_edit_pi_form(pi_manager)

    with tab4:
        render_delete_pi_form(pi_manager)

    st.markdown('</div>', unsafe_allow_html=True)


def render_pi_list(pi_manager):
    """PIコード一覧を表示"""
    st.write("### 📋 PIコード一覧 / PI Code List")

    # 検索機能
    search_query = st.text_input(
        "🔍 検索 / Search",
        placeholder="PIコード、備考、材質で検索...",
        help="PIコード名、備考、裏板材質で検索できます"
    )

    # データフィルタリング
    if search_query:
        filtered_codes = pi_manager.search_pi_codes(search_query)
    else:
        filtered_codes = pi_manager.pi_codes

    if not filtered_codes:
        st.info("該当するPIコードが見つかりません")
        return

    # データフレーム作成
    df = pi_manager.export_to_dataframe()
    if search_query:
        # 検索結果に絞り込み
        pi_codes_list = [pc.pi_code for pc in filtered_codes]
        df = df[df['PIコード'].isin(pi_codes_list)]

    # データテーブル表示
    st.write(f"**表示件数:** {len(df)} / {len(pi_manager.pi_codes)} 件")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PIコード": st.column_config.TextColumn("PIコード", width="medium"),
            "H加算": st.column_config.NumberColumn("H加算", format="%.1f mm"),
            "W加算": st.column_config.NumberColumn("W加算", format="%.1f mm"),
            "板厚": st.column_config.NumberColumn("板厚", format="%.1f mm"),
        }
    )


def render_add_pi_form(pi_manager):
    """PIコード追加フォーム"""
    st.write("### ➕ 新規PIコード追加 / Add New PI Code")

    with st.form("add_pi_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            pi_code = st.text_input(
                "PIコード *",
                placeholder="例: 77131000",
                help="一意のPIコード識別子"
            )

            h_expansion = st.number_input(
                "H寸法加算値 (mm) *",
                min_value=-500.0,
                max_value=500.0,
                value=0.0,
                step=0.1,
                help="完成H寸法に加算する値"
            )

            w_expansion = st.number_input(
                "W寸法加算値 (mm) *",
                min_value=-500.0,
                max_value=500.0,
                value=0.0,
                step=0.1,
                help="完成W寸法に加算する値"
            )

            thickness = st.number_input(
                "板厚 (mm) *",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                step=0.1,
                help="材料の板厚"
            )

        with col2:
            has_backing = st.checkbox("裏板あり / Has Backing")

            backing_material = ""
            backing_h = 0.0
            backing_w = 0.0

            if has_backing:
                backing_material = st.text_input(
                    "裏板材質",
                    placeholder="例: 5SECC",
                    help="裏板の材質"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    backing_h = st.number_input(
                        "裏板H寸法",
                        min_value=-500.0,
                        max_value=500.0,
                        value=0.0,
                        step=0.1
                    )
                with col_b:
                    backing_w = st.number_input(
                        "裏板W寸法",
                        min_value=-500.0,
                        max_value=500.0,
                        value=0.0,
                        step=0.1
                    )

            has_plaster = st.checkbox("プラスタあり / Has Plaster", value=True)

            plaster_h = 0.0
            plaster_w = 0.0

            if has_plaster:
                col_c, col_d = st.columns(2)
                with col_c:
                    plaster_h = st.number_input(
                        "プラスタH寸法",
                        min_value=-500.0,
                        max_value=500.0,
                        value=0.0,
                        step=0.1
                    )
                with col_d:
                    plaster_w = st.number_input(
                        "プラスタW寸法",
                        min_value=-500.0,
                        max_value=500.0,
                        value=0.0,
                        step=0.1
                    )

        description = st.text_area(
            "備考 / Description",
            placeholder="PIコードの説明や用途...",
            help="PIコードの説明や用途を記入"
        )

        # プレビュー
        if pi_code:
            st.markdown("#### 📐 計算プレビュー")
            sample_w, sample_h = 900.0, 600.0
            preview_w = sample_w + w_expansion
            preview_h = sample_h + h_expansion

            st.info(f"サンプル寸法 {sample_w:.1f}×{sample_h:.1f}mm → 展開寸法 {preview_w:.1f}×{preview_h:.1f}mm")

        # 送信ボタン
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button(
                "💾 PIコード追加",
                type="primary",
                use_container_width=True
            )
        with col2:
            if st.form_submit_button("🔄 Reset", use_container_width=True):
                st.rerun()

        if submitted:
            if pi_code:
                new_pi = PICode(
                    pi_code=pi_code,
                    h_expansion=h_expansion,
                    w_expansion=w_expansion,
                    has_backing=has_backing,
                    backing_material=backing_material,
                    backing_h=backing_h,
                    backing_w=backing_w,
                    has_plaster=has_plaster,
                    plaster_h=plaster_h,
                    plaster_w=plaster_w,
                    thickness=thickness,
                    description=description
                )

                if pi_manager.add_pi_code(new_pi):
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ <strong>成功！</strong> PIコード {pi_code} を追加しました<br>
                        <strong>Success!</strong> Added PI code {pi_code}
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                else:
                    st.error(f"❌ PIコード {pi_code} は既に存在します / PI code {pi_code} already exists")
            else:
                st.error("⚠️ PIコードは必須です / PI code is required")


def render_edit_pi_form(pi_manager):
    """PIコード編集フォーム"""
    st.write("### ✏️ PIコード編集 / Edit PI Code")

    if not pi_manager.pi_codes:
        st.info("編集するPIコードがありません / No PI codes to edit")
        return

    # PIコード選択
    pi_codes = pi_manager.get_all_pi_codes()
    selected_code = st.selectbox(
        "編集するPIコードを選択 / Select PI Code to Edit",
        pi_codes,
        help="編集したいPIコードを選択してください"
    )

    if selected_code:
        pi_info = pi_manager.get_pi_code(selected_code)
        if pi_info:
            # 現在の情報表示
            with st.expander("📋 現在のPIコード情報 / Current PI Code Info", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**PIコード:** {pi_info.pi_code}")
                    st.write(f"**H加算:** {pi_info.h_expansion}mm")
                    st.write(f"**W加算:** {pi_info.w_expansion}mm")
                with col2:
                    st.write(f"**裏板:** {'あり' if pi_info.has_backing else 'なし'}")
                    st.write(f"**裏板材質:** {pi_info.backing_material or 'N/A'}")
                    st.write(f"**プラスタ:** {'あり' if pi_info.has_plaster else 'なし'}")
                with col3:
                    st.write(f"**板厚:** {pi_info.thickness}mm")
                    st.write(f"**更新日:** {pi_info.last_updated[:10] if pi_info.last_updated else 'N/A'}")

            # 編集フォーム
            with st.form("edit_pi_form"):
                col1, col2 = st.columns(2)

                with col1:
                    new_h_expansion = st.number_input(
                        "H寸法加算値 (mm)",
                        min_value=-500.0,
                        max_value=500.0,
                        value=pi_info.h_expansion,
                        step=0.1
                    )

                    new_w_expansion = st.number_input(
                        "W寸法加算値 (mm)",
                        min_value=-500.0,
                        max_value=500.0,
                        value=pi_info.w_expansion,
                        step=0.1
                    )

                    new_thickness = st.number_input(
                        "板厚 (mm)",
                        min_value=0.1,
                        max_value=10.0,
                        value=pi_info.thickness,
                        step=0.1
                    )

                with col2:
                    new_has_backing = st.checkbox(
                        "裏板あり",
                        value=pi_info.has_backing
                    )

                    new_backing_material = st.text_input(
                        "裏板材質",
                        value=pi_info.backing_material
                    )

                    new_has_plaster = st.checkbox(
                        "プラスタあり",
                        value=pi_info.has_plaster
                    )

                new_description = st.text_area(
                    "備考",
                    value=pi_info.description
                )

                submitted = st.form_submit_button("💾 更新", type="primary")

                if submitted:
                    updates = {
                        'h_expansion': new_h_expansion,
                        'w_expansion': new_w_expansion,
                        'has_backing': new_has_backing,
                        'backing_material': new_backing_material,
                        'has_plaster': new_has_plaster,
                        'thickness': new_thickness,
                        'description': new_description
                    }

                    if pi_manager.update_pi_code(selected_code, updates):
                        st.success(f"✅ PIコード {selected_code} を更新しました / Updated PI code {selected_code}")
                        st.rerun()
                    else:
                        st.error("❌ 更新に失敗しました / Failed to update")


def render_delete_pi_form(pi_manager):
    """PIコード削除フォーム"""
    st.write("### 🗑️ PIコード削除 / Delete PI Code")

    if not pi_manager.pi_codes:
        st.info("削除するPIコードがありません / No PI codes to delete")
        return

    # PIコード選択
    pi_codes = pi_manager.get_all_pi_codes()
    selected_code = st.selectbox(
        "削除するPIコードを選択 / Select PI Code to Delete",
        pi_codes,
        help="⚠️ この操作は元に戻せません"
    )

    if selected_code:
        pi_info = pi_manager.get_pi_code(selected_code)
        if pi_info:
            # 削除確認表示
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ 削除確認 / Confirm Deletion</h4>
                <div style="margin-top: 1rem;">
                    <strong>PIコード:</strong> {pi_info.pi_code}<br>
                    <strong>H加算:</strong> {pi_info.h_expansion}mm<br>
                    <strong>W加算:</strong> {pi_info.w_expansion}mm<br>
                    <strong>裏板:</strong> {'あり' if pi_info.has_backing else 'なし'}<br>
                    <strong>備考:</strong> {pi_info.description or 'なし'}
                </div>
                <p style="margin-top: 1rem; margin-bottom: 0;">
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
                    if pi_manager.remove_pi_code(selected_code):
                        st.success(f"✅ PIコード {selected_code} を削除しました / Deleted PI code {selected_code}")
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
    """メイン関数"""
    setup_page_config()
    render_page_header()
    render_pi_summary()
    render_dimension_calculator()
    render_pi_management()

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p>💡 PIコードによる展開寸法計算で、より正確な切断計画を作成できます</p>
        <p>Create more accurate cutting plans with PI code-based dimension expansion</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()