"""
Data Management Page
データ管理ページ

Provides comprehensive data persistence management interface
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
from core.persistence_adapter import get_persistence_adapter
from core.database_manager import get_database_manager
from core.models import Panel
from ui.page_headers import render_unified_header, get_page_config
from ui.common_styles import get_common_css
from ui.metric_cards import render_metric_row, render_status_card


def render_data_management_page():
    """データ管理ページをレンダリング"""

    # Apply unified styling
    st.markdown(get_common_css(), unsafe_allow_html=True)

    # Unified page header
    config = get_page_config("data_management")
    render_unified_header(
        title_ja=config["title_ja"],
        title_en=config["title_en"],
        description=config["description"],
        icon=config["icon"]
    )

    # Initialize persistence adapter
    persistence = get_persistence_adapter()

    # System Status
    st.header("🔧 システム状態 / System Status")
    status = persistence.get_system_status()

    # Create status metrics
    metrics = [
        {
            'title': 'データベース',
            'value': '✅ 接続中' if status.get('database_available') else '❌ 利用不可',
            'subtitle': 'Database',
            'color': 'success' if status.get('database_available') else 'error'
        },
        {
            'title': 'JSONフォールバック',
            'value': '✅ 利用可能' if status.get('json_fallback_available') else '❌ 利用不可',
            'subtitle': 'JSON Fallback',
            'color': 'success' if status.get('json_fallback_available') else 'error'
        },
        {
            'title': '最終確認',
            'value': status.get('last_check', 'Unknown')[:19],
            'subtitle': 'Last Check',
            'color': 'neutral'
        }
    ]

    render_metric_row(metrics, 3)

    if status.get('database_stats'):
        st.subheader("📊 データベース統計 / Database Statistics")
        stats_df = pd.DataFrame([
            {"テーブル / Table": "材料 / Materials", "件数 / Count": status['database_stats'].get('materials', 0)},
            {"テーブル / Table": "PIコード / PI Codes", "件数 / Count": status['database_stats'].get('pi_codes', 0)},
            {"テーブル / Table": "プロジェクト / Projects", "件数 / Count": status['database_stats'].get('projects', 0)},
            {"テーブル / Table": "最適化履歴 / Optimization History", "件数 / Count": status['database_stats'].get('optimization_history', 0)}
        ])
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗂️ プロジェクト / Projects",
        "📈 最適化履歴 / History",
        "🏭 材料管理 / Materials",
        "🔧 PIコード / PI Codes",
        "💾 バックアップ / Backup"
    ])

    with tab1:
        render_project_management(persistence)

    with tab2:
        render_optimization_history(persistence)

    with tab3:
        render_material_management(persistence)

    with tab4:
        render_pi_code_management(persistence)

    with tab5:
        render_backup_management(persistence)


def render_project_management(persistence):
    """プロジェクト管理セクション"""
    st.subheader("🗂️ プロジェクト管理 / Project Management")

    # Save current session as project
    if 'panels' in st.session_state and st.session_state.panels:
        st.write("### 💾 現在のセッションを保存 / Save Current Session")

        col1, col2 = st.columns([2, 1])
        with col1:
            project_name = st.text_input(
                "プロジェクト名 / Project Name",
                value=f"Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                key="new_project_name"
            )
            project_description = st.text_area(
                "説明 / Description",
                key="new_project_description"
            )

        with col2:
            st.write("**現在のパネル / Current Panels**")
            st.metric("パネル数 / Panel Count", len(st.session_state.panels))
            st.metric("総数量 / Total Quantity", sum(p.quantity for p in st.session_state.panels))

        if st.button("💾 プロジェクト保存 / Save Project", type="primary"):
            if project_name:
                project_id = persistence.save_project(
                    project_name,
                    st.session_state.panels,
                    project_description
                )
                if project_id:
                    st.success(f"✅ プロジェクトが保存されました / Project saved (ID: {project_id})")
                else:
                    st.error("❌ プロジェクトの保存に失敗しました / Failed to save project")
            else:
                st.error("プロジェクト名を入力してください / Please enter project name")

    st.markdown("---")

    # Load existing projects
    st.write("### 📂 既存プロジェクト / Existing Projects")
    projects = persistence.get_projects()

    if projects:
        projects_df = pd.DataFrame(projects)
        projects_df['created_at'] = pd.to_datetime(projects_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

        # Display projects
        for _, project in projects_df.iterrows():
            with st.expander(f"📁 {project['name']} (ID: {project['id']})"):
                st.write(f"**説明 / Description:** {project['description']}")
                st.write(f"**作成日時 / Created:** {project['created_at']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📥 読み込み / Load", key=f"load_{project['id']}"):
                        project_data = persistence.load_project(project['id'])
                        if project_data:
                            name, panels, description = project_data
                            st.session_state.panels = panels
                            st.success(f"✅ プロジェクト '{name}' を読み込みました / Loaded project '{name}'")
                            st.rerun()
                        else:
                            st.error("❌ プロジェクトの読み込みに失敗しました / Failed to load project")

                with col2:
                    if st.button(f"📊 詳細 / Details", key=f"details_{project['id']}"):
                        project_data = persistence.load_project(project['id'])
                        if project_data:
                            name, panels, description = project_data
                            st.json({
                                "name": name,
                                "description": description,
                                "panel_count": len(panels),
                                "total_quantity": sum(p.quantity for p in panels)
                            })
    else:
        st.info("📂 保存されたプロジェクトはありません / No saved projects found")


def render_optimization_history(persistence):
    """最適化履歴セクション"""
    st.subheader("📈 最適化履歴 / Optimization History")

    # History controls
    col1, col2 = st.columns([1, 1])
    with col1:
        limit = st.selectbox("表示件数 / Display Limit", [10, 25, 50, 100], index=1)

    with col2:
        if st.button("🔄 履歴更新 / Refresh History"):
            st.rerun()

    # Get optimization history
    history = persistence.get_optimization_history(limit)

    if history:
        # Convert to DataFrame for better display
        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        history_df['placement_rate'] = history_df['placement_rate'].round(1)
        history_df['efficiency'] = history_df['efficiency'].round(1)
        history_df['processing_time'] = history_df['processing_time'].round(2)

        # Summary metrics
        st.write("### 📊 履歴サマリー / History Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_placement = history_df['placement_rate'].mean()
            st.metric("平均配置率 / Avg Placement", f"{avg_placement:.1f}%")

        with col2:
            avg_efficiency = history_df['efficiency'].mean()
            st.metric("平均効率 / Avg Efficiency", f"{avg_efficiency:.1f}%")

        with col3:
            avg_time = history_df['processing_time'].mean()
            st.metric("平均処理時間 / Avg Time", f"{avg_time:.1f}s")

        with col4:
            total_panels = history_df['total_panels'].sum()
            st.metric("総パネル数 / Total Panels", f"{total_panels:,}")

        # History table
        st.write("### 📋 最適化履歴 / Optimization History")
        display_columns = [
            'timestamp', 'algorithm_used', 'total_panels', 'placed_panels',
            'placement_rate', 'efficiency', 'sheets_used', 'processing_time'
        ]

        # Rename columns for display
        column_mapping = {
            'timestamp': '日時 / Timestamp',
            'algorithm_used': 'アルゴリズム / Algorithm',
            'total_panels': '総パネル / Total Panels',
            'placed_panels': '配置パネル / Placed Panels',
            'placement_rate': '配置率(%) / Placement Rate(%)',
            'efficiency': '効率(%) / Efficiency(%)',
            'sheets_used': 'シート数 / Sheets Used',
            'processing_time': '処理時間(s) / Processing Time(s)'
        }

        display_df = history_df[display_columns].rename(columns=column_mapping)
        st.dataframe(display_df, use_container_width=True, height=400)

        # Export option
        if st.button("📤 CSVエクスポート / Export CSV"):
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 履歴ダウンロード / Download History",
                data=csv,
                file_name=f"optimization_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    else:
        st.info("📈 最適化履歴はありません / No optimization history found")


def render_material_management(persistence):
    """材料管理セクション"""
    st.subheader("🏭 材料管理 / Material Management")

    # Get materials
    materials = persistence.get_materials()

    if materials:
        # Convert to DataFrame
        materials_data = []
        for material in materials:
            materials_data.append({
                "材料コード / Material Code": material.material_code,
                "材料種類 / Material Type": material.material_type,
                "厚み(mm) / Thickness(mm)": material.thickness,
                "幅(mm) / Width(mm)": material.width,
                "高さ(mm) / Height(mm)": material.height,
                "面積(m²) / Area(m²)": material.area,
                "在庫数 / Availability": material.availability,
                "コスト / Cost": f"¥{material.cost_per_sheet:,.0f}",
                "最終更新 / Last Updated": material.last_updated[:19] if material.last_updated else ""
            })

        materials_df = pd.DataFrame(materials_data)

        # Summary metrics
        st.write("### 📊 材料サマリー / Material Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_sheets = sum(m.availability for m in materials)
            st.metric("総シート数 / Total Sheets", f"{total_sheets:,}")

        with col2:
            material_types = len(set(m.material_type for m in materials))
            st.metric("材料種類 / Material Types", material_types)

        with col3:
            total_area = sum(m.area * m.availability for m in materials)
            st.metric("総面積(m²) / Total Area(m²)", f"{total_area:.1f}")

        # Materials table
        st.write("### 📋 材料一覧 / Material List")
        st.dataframe(materials_df, use_container_width=True, height=400)

        # Update availability
        st.write("### ✏️ 在庫更新 / Update Availability")
        col1, col2, col3 = st.columns(3)

        with col1:
            material_codes = [m.material_code for m in materials]
            selected_code = st.selectbox("材料コード / Material Code", material_codes)

        with col2:
            current_availability = next(m.availability for m in materials if m.material_code == selected_code)
            new_availability = st.number_input(
                "新しい在庫数 / New Availability",
                min_value=0,
                value=current_availability,
                step=1
            )

        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🔄 在庫更新 / Update Availability"):
                if persistence.update_material_availability(selected_code, new_availability):
                    st.success("✅ 在庫を更新しました / Availability updated")
                    st.rerun()
                else:
                    st.error("❌ 更新に失敗しました / Update failed")

    else:
        st.info("🏭 材料データがありません / No material data found")


def render_pi_code_management(persistence):
    """PIコード管理セクション"""
    st.subheader("🔧 PIコード管理 / PI Code Management")

    # PI Code search
    col1, col2 = st.columns([2, 1])
    with col1:
        search_code = st.text_input("PIコード検索 / Search PI Code", key="pi_search")

    with col2:
        if st.button("🔍 検索 / Search"):
            if search_code:
                pi_code = persistence.get_pi_code(search_code)
                if pi_code:
                    st.json({
                        "pi_code": pi_code.pi_code,
                        "width_expansion": pi_code.width_expansion,
                        "height_expansion": pi_code.height_expansion,
                        "has_backing": pi_code.has_backing,
                        "backing_material": pi_code.backing_material,
                        "backing_thickness": pi_code.backing_thickness,
                        "notes": pi_code.notes,
                        "last_updated": pi_code.last_updated
                    })
                else:
                    st.warning(f"PIコード '{search_code}' が見つかりません / PI Code '{search_code}' not found")

    # Get all PI codes (for display purposes, limit to summary)
    st.write("### 📊 PIコード統計 / PI Code Statistics")

    try:
        from core.pi_manager import get_pi_manager
        pi_manager = get_pi_manager()
        summary = pi_manager.get_summary()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総PIコード数 / Total PI Codes", summary['total_codes'])

        with col2:
            st.metric("バッキング材付き / With Backing", summary['codes_with_backing'])

        with col3:
            st.metric("材料種類 / Material Types", summary['unique_materials'])

        if summary['backing_materials']:
            st.write("**バッキング材料 / Backing Materials:**")
            st.write(", ".join(summary['backing_materials']))

    except Exception as e:
        st.error(f"PIコード統計の取得に失敗しました / Failed to get PI code statistics: {e}")


def render_backup_management(persistence):
    """バックアップ管理セクション"""
    st.subheader("💾 バックアップ管理 / Backup Management")

    st.write("### 🗂️ データバックアップ / Data Backup")
    st.info("💡 システム全体のデータ（データベース、JSON設定ファイル）をバックアップします")

    col1, col2 = st.columns([2, 1])
    with col1:
        backup_path = st.text_input(
            "バックアップ先 / Backup Destination",
            value="backups",
            help="バックアップファイルを保存するディレクトリ"
        )

    with col2:
        st.write("")  # Spacing
        if st.button("💾 バックアップ実行 / Execute Backup", type="primary"):
            with st.spinner("バックアップ中... / Creating backup..."):
                success = persistence.backup_all_data(backup_path)
                if success:
                    st.success("✅ バックアップが完了しました / Backup completed successfully")
                else:
                    st.error("❌ バックアップに失敗しました / Backup failed")

    st.markdown("---")

    # Database migration
    st.write("### 🔄 データベース移行 / Database Migration")
    st.info("💡 JSONファイルからSQLiteデータベースにデータを移行します")

    if st.button("🔄 JSON→DB移行 / Migrate JSON to DB"):
        try:
            db_manager = get_database_manager()
            with st.spinner("移行中... / Migrating..."):
                success = db_manager.migrate_from_json()
                if success:
                    st.success("✅ データ移行が完了しました / Migration completed")
                    st.rerun()
                else:
                    st.error("❌ データ移行に失敗しました / Migration failed")
        except Exception as e:
            st.error(f"移行エラー / Migration error: {e}")

    st.markdown("---")

    # System maintenance
    st.write("### 🔧 システムメンテナンス / System Maintenance")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 キャッシュクリア / Clear Cache"):
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            st.success("✅ キャッシュをクリアしました / Cache cleared")

    with col2:
        if st.button("🔄 システム再起動 / Restart System"):
            st.info("⚠️ ページを手動で再読み込みしてください / Please manually reload the page")


if __name__ == "__main__":
    render_data_management_page()