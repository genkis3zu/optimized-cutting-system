#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI Code Manager - PIコード管理・参照ページ
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.database_models import MaterialDatabase, PiCodeData
from ui.page_headers import render_main_header


def initialize_database():
    """Initialize database if needed"""
    try:
        db = MaterialDatabase()
        return db
    except Exception as e:
        st.error(f"データベース初期化エラー: {e}")
        return None


def render_pi_code_search(db: MaterialDatabase):
    """PIコード検索機能"""
    st.subheader("🔍 PIコード検索")

    col1, col2 = st.columns([2, 1])

    with col1:
        search_term = st.text_input(
            "PIコード検索",
            placeholder="例: 18131000, 1813, LUX",
            help="PIコードの一部または全部を入力してください"
        )

    with col2:
        search_button = st.button("🔍 検索", type="primary")

    if search_button or search_term:
        try:
            pi_codes = db.search_pi_codes(search_term)

            if pi_codes:
                st.success(f"{len(pi_codes)}件のPIコードが見つかりました")

                # Convert to DataFrame for display
                data = []
                for pi in pi_codes:
                    data.append({
                        'PIコード': pi.pi_code,
                        'W展開': f"+{pi.width_expansion:.0f}mm",
                        'H展開': f"+{pi.height_expansion:.0f}mm",
                        '裏板': "有" if pi.back_plate else "無",
                        '裏板W': f"{pi.back_plate_w:.0f}mm" if pi.back_plate else "-",
                        '裏板H': f"{pi.back_plate_h:.0f}mm" if pi.back_plate else "-",
                        'プラスタ': "有" if pi.plaster else "無",
                        'プラスタW': f"{pi.plaster_w:.0f}mm" if pi.plaster else "-",
                        'プラスタH': f"{pi.plaster_h:.0f}mm" if pi.plaster else "-",
                        '板厚': f"{pi.plate_thickness:.1f}mm" if pi.plate_thickness > 0 else "-"
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Download option
                if len(data) > 0:
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📊 CSVダウンロード",
                        data=csv,
                        file_name=f"pi_codes_{search_term if search_term else 'all'}.csv",
                        mime="text/csv"
                    )

            else:
                st.warning("該当するPIコードが見つかりませんでした")

        except Exception as e:
            st.error(f"検索エラー: {e}")


def render_pi_code_details(db: MaterialDatabase):
    """PIコード詳細表示"""
    st.subheader("📋 PIコード詳細")

    # Get all PI codes for selectbox
    pi_codes_list = db.get_pi_codes_list()

    if not pi_codes_list:
        st.warning("PIコードがデータベースに登録されていません")
        return

    selected_pi_code = st.selectbox(
        "PIコードを選択:",
        options=pi_codes_list,
        index=0 if pi_codes_list else None,
        help="詳細を確認したいPIコードを選択してください"
    )

    if selected_pi_code:
        pi_data = db.get_pi_code(selected_pi_code)

        if pi_data:
            # Display detailed information
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("PIコード", pi_data.pi_code)
                st.metric("W展開", f"+{pi_data.width_expansion:.1f}mm")
                st.metric("H展開", f"+{pi_data.height_expansion:.1f}mm")

            with col2:
                st.metric("裏板", "有" if pi_data.back_plate else "無")
                if pi_data.back_plate:
                    st.metric("裏板W", f"{pi_data.back_plate_w:.1f}mm")
                    st.metric("裏板H", f"{pi_data.back_plate_h:.1f}mm")

            with col3:
                st.metric("プラスタ", "有" if pi_data.plaster else "無")
                if pi_data.plaster:
                    st.metric("プラスタW", f"{pi_data.plaster_w:.1f}mm")
                    st.metric("プラスタH", f"{pi_data.plaster_h:.1f}mm")

            if pi_data.plate_thickness > 0:
                st.metric("板厚", f"{pi_data.plate_thickness:.1f}mm")

            # Dimension expansion calculator
            st.subheader("🧮 寸法展開計算")

            col1, col2, col3 = st.columns(3)

            with col1:
                finished_w = st.number_input(
                    "完成W寸法 (mm)",
                    min_value=50.0,
                    max_value=3000.0,
                    value=968.0,
                    step=1.0
                )

            with col2:
                finished_h = st.number_input(
                    "完成H寸法 (mm)",
                    min_value=50.0,
                    max_value=3000.0,
                    value=712.0,
                    step=1.0
                )

            with col3:
                if st.button("🧮 計算実行"):
                    expanded_w, expanded_h = pi_data.expand_dimensions(finished_w, finished_h)

                    st.success("展開寸法計算完了")
                    st.write(f"**完成寸法**: {finished_w:.0f} × {finished_h:.0f} mm")
                    st.write(f"**展開寸法**: {expanded_w:.0f} × {expanded_h:.0f} mm")
                    st.write(f"**変化量**: +{expanded_w - finished_w:.0f}mm W, +{expanded_h - finished_h:.0f}mm H")

        else:
            st.error("PIコードデータの取得に失敗しました")


def render_pi_code_stats(db: MaterialDatabase):
    """PIコード統計情報"""
    st.subheader("📊 PIコードデータベース統計")

    try:
        stats = db.get_pi_code_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "総PIコード数",
                f"{stats['total_pi_codes']:,}件"
            )

        with col2:
            st.metric(
                "プラスタ付き",
                f"{stats['codes_with_plaster']:,}件"
            )

        with col3:
            st.metric(
                "裏板付き",
                f"{stats['codes_with_back_plate']:,}件"
            )

        with col4:
            if stats['avg_width_expansion'] > 0:
                st.metric(
                    "平均W展開",
                    f"+{stats['avg_width_expansion']:.1f}mm"
                )

        # Additional statistics
        if stats['avg_height_expansion'] > 0:
            st.metric(
                "平均H展開",
                f"+{stats['avg_height_expansion']:.1f}mm"
            )

    except Exception as e:
        st.error(f"統計情報の取得に失敗しました: {e}")


def render_pi_code_list(db: MaterialDatabase):
    """全PIコード一覧"""
    st.subheader("📋 全PIコード一覧")

    try:
        # Add pagination
        page_size = st.selectbox("表示件数", [20, 50, 100, 200], value=50)

        all_pi_codes = db.get_all_pi_codes()

        if not all_pi_codes:
            st.warning("PIコードがデータベースに登録されていません")
            return

        # Pagination
        total_pages = (len(all_pi_codes) - 1) // page_size + 1
        current_page = st.selectbox(
            f"ページ選択 (全{total_pages}ページ)",
            range(1, total_pages + 1),
            format_func=lambda x: f"ページ {x}"
        )

        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(all_pi_codes))
        page_pi_codes = all_pi_codes[start_idx:end_idx]

        # Convert to DataFrame
        data = []
        for pi in page_pi_codes:
            data.append({
                'PIコード': pi.pi_code,
                'W展開': f"+{pi.width_expansion:.0f}",
                'H展開': f"+{pi.height_expansion:.0f}",
                '裏板': "○" if pi.back_plate else "×",
                'プラスタ': "○" if pi.plaster else "×",
                '板厚': f"{pi.plate_thickness:.1f}" if pi.plate_thickness > 0 else "-"
            })

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            st.info(f"表示中: {start_idx + 1}-{end_idx}件 / 全{len(all_pi_codes)}件")

            # Download all data option
            if st.button("📊 全データCSVダウンロード"):
                all_data = []
                for pi in all_pi_codes:
                    all_data.append({
                        'PIコード': pi.pi_code,
                        'W展開': pi.width_expansion,
                        'H展開': pi.height_expansion,
                        '裏板': pi.back_plate,
                        '裏板W': pi.back_plate_w,
                        '裏板H': pi.back_plate_h,
                        'プラスタ': pi.plaster,
                        'プラスタW': pi.plaster_w,
                        'プラスタH': pi.plaster_h,
                        '板厚': pi.plate_thickness
                    })

                all_df = pd.DataFrame(all_data)
                csv = all_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📊 全PIコードCSVダウンロード",
                    data=csv,
                    file_name="all_pi_codes.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"PIコード一覧の取得に失敗しました: {e}")


def main():
    """Main page function"""
    render_main_header(
        "📋 PIコード管理",
        "PIコードの参照・検索・管理を行います"
    )

    # Initialize database
    db = initialize_database()
    if not db:
        return

    # Create tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 PIコード検索",
        "📋 詳細表示",
        "📊 統計情報",
        "📋 全一覧"
    ])

    with tab1:
        render_pi_code_search(db)

    with tab2:
        render_pi_code_details(db)

    with tab3:
        render_pi_code_stats(db)

    with tab4:
        render_pi_code_list(db)

    # Database info
    with st.sidebar:
        st.subheader("📊 データベース情報")
        try:
            stats = db.get_pi_code_stats()
            st.write(f"**総PIコード数**: {stats['total_pi_codes']:,}件")
            st.write(f"**プラスタ付き**: {stats['codes_with_plaster']:,}件")
            st.write(f"**裏板付き**: {stats['codes_with_back_plate']:,}件")
        except:
            st.write("統計情報を取得できませんでした")


if __name__ == "__main__":
    main()