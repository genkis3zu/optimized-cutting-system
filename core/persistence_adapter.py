"""
Persistence Adapter for Steel Cutting Optimization System
既存システムとSQLiteデータベースの統合アダプター

Provides backward compatibility while migrating to database-based persistence
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from core.database_manager import get_database_manager, OptimizationRecord
from core.models import Panel, OptimizationConstraints
from core.material_manager import MaterialSheet
from core.pi_manager import PICode
import json


class PersistenceAdapter:
    """永続化統合アダプター"""

    def __init__(self, use_database: bool = True):
        self.use_database = use_database
        self.db_manager = get_database_manager() if use_database else None
        self.logger = logging.getLogger(__name__)

        # Try to migrate existing JSON data on first use
        if use_database and self.db_manager:
            self._migrate_if_needed()

    def _migrate_if_needed(self):
        """必要に応じてJSONデータを移行"""
        try:
            # Check if migration is needed (database is empty)
            stats = self.db_manager.get_database_stats()
            if stats.get('materials', 0) == 0 and stats.get('pi_codes', 0) == 0:
                self.logger.info("Database is empty, attempting migration from JSON files...")
                self.db_manager.migrate_from_json()
        except Exception as e:
            self.logger.error(f"Migration check failed: {e}")

    # Material Management - 材料管理
    def get_materials(self, material_type: Optional[str] = None) -> List[MaterialSheet]:
        """材料取得 (データベース優先、フォールバック対応)"""
        if self.use_database and self.db_manager:
            try:
                return self.db_manager.get_materials(material_type)
            except Exception as e:
                self.logger.error(f"Database material retrieval failed: {e}")

        # Fallback to JSON-based material manager
        try:
            from core.material_manager import get_material_manager
            material_manager = get_material_manager()
            if material_type:
                return [m for m in material_manager.inventory if m.material_type == material_type]
            return material_manager.inventory
        except Exception as e:
            self.logger.error(f"JSON material retrieval failed: {e}")
            return []

    def add_material(self, material: MaterialSheet) -> bool:
        """材料追加 (データベース + JSONバックアップ)"""
        success = False

        # Primary: Database
        if self.use_database and self.db_manager:
            try:
                success = self.db_manager.add_material(material)
            except Exception as e:
                self.logger.error(f"Database material addition failed: {e}")

        # Backup: JSON (for backward compatibility)
        try:
            from core.material_manager import get_material_manager
            material_manager = get_material_manager()
            json_success = material_manager.add_material_sheet(material)
            if not success:
                success = json_success  # Use JSON result if database failed
        except Exception as e:
            self.logger.error(f"JSON material addition failed: {e}")

        return success

    def update_material_availability(self, material_code: str, new_availability: int) -> bool:
        """材料在庫更新"""
        success = False

        if self.use_database and self.db_manager:
            try:
                success = self.db_manager.update_material_availability(material_code, new_availability)
            except Exception as e:
                self.logger.error(f"Database material update failed: {e}")

        # Also update JSON for consistency
        try:
            from core.material_manager import get_material_manager
            material_manager = get_material_manager()
            for material in material_manager.inventory:
                if material.material_code == material_code:
                    material.availability = new_availability
                    material.last_updated = datetime.now().isoformat()
                    material_manager._save_inventory()
                    if not success:
                        success = True
                    break
        except Exception as e:
            self.logger.error(f"JSON material update failed: {e}")

        return success

    # PI Code Management - PIコード管理
    def get_pi_code(self, pi_code: str) -> Optional[PICode]:
        """PIコード取得"""
        if self.use_database and self.db_manager:
            try:
                result = self.db_manager.get_pi_code(pi_code)
                if result:
                    return result
            except Exception as e:
                self.logger.error(f"Database PI code retrieval failed: {e}")

        # Fallback to JSON
        try:
            from core.pi_manager import get_pi_manager
            pi_manager = get_pi_manager()
            return pi_manager.get_pi_code(pi_code)
        except Exception as e:
            self.logger.error(f"JSON PI code retrieval failed: {e}")
            return None

    def add_pi_code(self, pi_code: PICode) -> bool:
        """PIコード追加"""
        success = False

        if self.use_database and self.db_manager:
            try:
                success = self.db_manager.add_pi_code(pi_code)
            except Exception as e:
                self.logger.error(f"Database PI code addition failed: {e}")

        # Also add to JSON
        try:
            from core.pi_manager import get_pi_manager
            pi_manager = get_pi_manager()
            json_success = pi_manager.add_pi_code(pi_code)
            if not success:
                success = json_success
        except Exception as e:
            self.logger.error(f"JSON PI code addition failed: {e}")

        return success

    # Project Management - プロジェクト管理
    def save_project(self, name: str, panels: List[Panel], description: str = "") -> int:
        """プロジェクト保存"""
        if self.use_database and self.db_manager:
            try:
                return self.db_manager.save_project(name, panels, description)
            except Exception as e:
                self.logger.error(f"Database project save failed: {e}")

        # Fallback: Save as JSON file
        try:
            import os
            project_dir = "data/projects"
            os.makedirs(project_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{project_dir}/{name}_{timestamp}.json"

            project_data = {
                "name": name,
                "description": description,
                "panels": [panel.__dict__ for panel in panels],
                "created_at": datetime.now().isoformat()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Project saved as JSON: {filename}")
            return hash(filename) % 1000000  # Return a pseudo-ID

        except Exception as e:
            self.logger.error(f"JSON project save failed: {e}")
            return 0

    def load_project(self, project_id: int) -> Optional[Tuple[str, List[Panel], str]]:
        """プロジェクト読み込み"""
        if self.use_database and self.db_manager:
            try:
                result = self.db_manager.load_project(project_id)
                if result:
                    return result
            except Exception as e:
                self.logger.error(f"Database project load failed: {e}")

        # Could implement JSON project loading here if needed
        return None

    def get_projects(self) -> List[Dict]:
        """プロジェクト一覧取得"""
        if self.use_database and self.db_manager:
            try:
                return self.db_manager.get_projects()
            except Exception as e:
                self.logger.error(f"Database project list failed: {e}")

        # Fallback: List JSON project files
        try:
            import os
            import glob
            project_files = glob.glob("data/projects/*.json")
            projects = []

            for file_path in project_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        projects.append({
                            'id': hash(file_path) % 1000000,
                            'name': data.get('name', os.path.basename(file_path)),
                            'description': data.get('description', ''),
                            'created_at': data.get('created_at', ''),
                            'updated_at': data.get('created_at', '')
                        })
                except Exception as e:
                    self.logger.error(f"Failed to read project file {file_path}: {e}")

            return projects

        except Exception as e:
            self.logger.error(f"JSON project listing failed: {e}")
            return []

    # Optimization History - 最適化履歴
    def save_optimization_result(self,
                                panels: List[Panel],
                                constraints: OptimizationConstraints,
                                results: Any,
                                algorithm_used: str,
                                processing_time: float,
                                metrics: Dict[str, Any]) -> int:
        """最適化結果保存"""
        if self.use_database and self.db_manager:
            try:
                record = OptimizationRecord(
                    id=None,
                    timestamp=datetime.now().isoformat(),
                    input_panels=json.dumps([panel.__dict__ for panel in panels], ensure_ascii=False),
                    constraints=json.dumps(constraints.__dict__, ensure_ascii=False),
                    results=json.dumps(results, ensure_ascii=False, default=str),
                    algorithm_used=algorithm_used,
                    processing_time=processing_time,
                    total_panels=metrics.get('total_panels', 0),
                    placed_panels=metrics.get('placed_panels', 0),
                    placement_rate=metrics.get('placement_rate', 0.0),
                    efficiency=metrics.get('efficiency', 0.0),
                    sheets_used=metrics.get('sheets_used', 0)
                )
                return self.db_manager.save_optimization_result(record)
            except Exception as e:
                self.logger.error(f"Database optimization save failed: {e}")

        # Fallback: Save as JSON file
        try:
            import os
            history_dir = "data/optimization_history"
            os.makedirs(history_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{history_dir}/optimization_{timestamp}.json"

            optimization_data = {
                "timestamp": datetime.now().isoformat(),
                "input_panels": [panel.__dict__ for panel in panels],
                "constraints": constraints.__dict__,
                "results": results,
                "algorithm_used": algorithm_used,
                "processing_time": processing_time,
                "metrics": metrics
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(optimization_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Optimization result saved as JSON: {filename}")
            return hash(filename) % 1000000

        except Exception as e:
            self.logger.error(f"JSON optimization save failed: {e}")
            return 0

    def get_optimization_history(self, limit: int = 50) -> List[Dict]:
        """最適化履歴取得"""
        if self.use_database and self.db_manager:
            try:
                return self.db_manager.get_optimization_history(limit)
            except Exception as e:
                self.logger.error(f"Database optimization history failed: {e}")

        # Fallback: Read JSON history files
        try:
            import os
            import glob
            history_files = glob.glob("data/optimization_history/*.json")
            history_files.sort(reverse=True)  # Most recent first

            history = []
            for file_path in history_files[:limit]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        history.append({
                            'id': hash(file_path) % 1000000,
                            'timestamp': data.get('timestamp', ''),
                            'algorithm_used': data.get('algorithm_used', ''),
                            'processing_time': data.get('processing_time', 0.0),
                            'total_panels': data.get('metrics', {}).get('total_panels', 0),
                            'placed_panels': data.get('metrics', {}).get('placed_panels', 0),
                            'placement_rate': data.get('metrics', {}).get('placement_rate', 0.0),
                            'efficiency': data.get('metrics', {}).get('efficiency', 0.0),
                            'sheets_used': data.get('metrics', {}).get('sheets_used', 0)
                        })
                except Exception as e:
                    self.logger.error(f"Failed to read history file {file_path}: {e}")

            return history

        except Exception as e:
            self.logger.error(f"JSON optimization history failed: {e}")
            return []

    # System Configuration - システム設定
    def set_config(self, key: str, value: str, description: str = "") -> bool:
        """システム設定保存"""
        if self.use_database and self.db_manager:
            try:
                return self.db_manager.set_config(key, value, description)
            except Exception as e:
                self.logger.error(f"Database config save failed: {e}")

        # Fallback: Use Streamlit session state
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                if 'system_config' not in st.session_state:
                    st.session_state.system_config = {}
                st.session_state.system_config[key] = value
                return True
        except Exception as e:
            self.logger.error(f"Session state config save failed: {e}")

        return False

    def get_config(self, key: str, default: str = "") -> str:
        """システム設定取得"""
        if self.use_database and self.db_manager:
            try:
                result = self.db_manager.get_config(key, default)
                if result != default:
                    return result
            except Exception as e:
                self.logger.error(f"Database config get failed: {e}")

        # Fallback: Use Streamlit session state
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and 'system_config' in st.session_state:
                return st.session_state.system_config.get(key, default)
        except Exception as e:
            self.logger.error(f"Session state config get failed: {e}")

        return default

    # Maintenance Operations - メンテナンス操作
    def backup_all_data(self, backup_dir: str) -> bool:
        """全データバックアップ"""
        import os
        import shutil
        from datetime import datetime

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"cutting_system_backup_{timestamp}")
            os.makedirs(backup_path, exist_ok=True)

            success = True

            # Backup database
            if self.use_database and self.db_manager:
                db_backup_path = os.path.join(backup_path, "cutting_system.db")
                if not self.db_manager.backup_database(db_backup_path):
                    success = False

            # Backup JSON files
            json_dirs = ["data", "config"]
            for dir_name in json_dirs:
                if os.path.exists(dir_name):
                    shutil.copytree(dir_name, os.path.join(backup_path, dir_name), dirs_exist_ok=True)

            self.logger.info(f"Full backup completed: {backup_path}")
            return success

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        status = {
            "database_enabled": self.use_database,
            "database_available": False,
            "json_fallback_available": True,
            "last_check": datetime.now().isoformat()
        }

        if self.use_database and self.db_manager:
            try:
                stats = self.db_manager.get_database_stats()
                status["database_available"] = True
                status["database_stats"] = stats
            except Exception as e:
                status["database_error"] = str(e)

        return status


# Singleton instance
_persistence_adapter = None

def get_persistence_adapter(use_database: bool = True) -> PersistenceAdapter:
    """永続化アダプターのシングルトンインスタンス取得"""
    global _persistence_adapter
    if _persistence_adapter is None:
        _persistence_adapter = PersistenceAdapter(use_database)
    return _persistence_adapter