"""
Database Manager for Steel Cutting Optimization System
鋼板切断最適化システム用データベース管理

Provides centralized SQLite-based persistence for all system data
"""

import sqlite3
import json
import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
from core.models import Panel
from core.material_manager import MaterialSheet
from core.pi_manager import PICode


@dataclass
class OptimizationRecord:
    """最適化実行記録"""
    id: Optional[int]
    timestamp: str
    input_panels: str  # JSON serialized panel data
    constraints: str   # JSON serialized constraints
    results: str       # JSON serialized optimization results
    algorithm_used: str
    processing_time: float
    total_panels: int
    placed_panels: int
    placement_rate: float
    efficiency: float
    sheets_used: int
    created_by: str = "system"


@dataclass
class ProjectRecord:
    """プロジェクト記録"""
    id: Optional[int]
    name: str
    description: str
    created_at: str
    updated_at: str
    panel_data: str    # JSON serialized
    active: bool = True


class DatabaseManager:
    """統合データベース管理クラス"""

    def __init__(self, db_path: str = "data/cutting_system.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """データベース初期化"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Materials table - 材料在庫管理
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_code TEXT UNIQUE NOT NULL,
                    material_type TEXT NOT NULL,
                    thickness REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    area REAL NOT NULL,
                    cost_per_sheet REAL DEFAULT 15000.0,
                    availability INTEGER DEFAULT 100,
                    supplier TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # PI Codes table - PIコード管理
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pi_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pi_code TEXT UNIQUE NOT NULL,
                    width_expansion REAL NOT NULL,
                    height_expansion REAL NOT NULL,
                    has_backing BOOLEAN DEFAULT 0,
                    backing_material TEXT DEFAULT '',
                    backing_thickness REAL DEFAULT 0.0,
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Projects table - プロジェクト管理
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    panel_data TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Optimization history - 最適化履歴
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    input_panels TEXT NOT NULL,
                    constraints TEXT NOT NULL,
                    results TEXT NOT NULL,
                    algorithm_used TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    total_panels INTEGER NOT NULL,
                    placed_panels INTEGER NOT NULL,
                    placement_rate REAL NOT NULL,
                    efficiency REAL NOT NULL,
                    sheets_used INTEGER NOT NULL,
                    created_by TEXT DEFAULT 'system',
                    project_id INTEGER,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)

            # Material usage tracking - 材料使用追跡
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS material_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id INTEGER NOT NULL,
                    material_code TEXT NOT NULL,
                    sheets_used INTEGER NOT NULL,
                    efficiency REAL NOT NULL,
                    waste_area REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (optimization_id) REFERENCES optimization_history (id),
                    FOREIGN KEY (material_code) REFERENCES materials (material_code)
                )
            """)

            # System configuration - システム設定
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_materials_type ON materials (material_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pi_codes_code ON pi_codes (pi_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimization_timestamp ON optimization_history (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_active ON projects (active)")

            conn.commit()
            self.logger.info("Database initialized successfully")

    @contextmanager
    def _get_connection(self):
        """安全なDB接続コンテキスト"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            conn.close()

    # Materials Management - 材料管理


    def add_material(self, material: MaterialSheet) -> bool:
        """材料追加"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO materials
                    (material_code, material_type, thickness, width, height, area,
                     cost_per_sheet, availability, supplier, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    material.material_code, material.material_type, material.thickness,
                    material.width, material.height, material.area,
                    material.cost_per_sheet, material.availability, material.supplier,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"Material code {material.material_code} already exists")
            return False
        except Exception as e:
            self.logger.error(f"Failed to add material: {e}")
            return False

    def get_materials(self, material_type: Optional[str] = None) -> List[MaterialSheet]:
        """材料取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if material_type:
                    cursor.execute("SELECT * FROM materials WHERE material_type = ? AND availability > 0", (material_type,))
                else:
                    cursor.execute("SELECT * FROM materials WHERE availability > 0")

                materials = []
                for row in cursor.fetchall():
                    materials.append(MaterialSheet(
                        material_code=row['material_code'],
                        material_type=row['material_type'],
                        thickness=row['thickness'],
                        width=row['width'],
                        height=row['height'],
                        area=row['area'],
                        cost_per_sheet=row['cost_per_sheet'],
                        availability=row['availability'],
                        supplier=row['supplier'],
                        last_updated=row['updated_at']
                    ))
                return materials
        except Exception as e:
            self.logger.error(f"Failed to get materials: {e}")
            return []

    def update_material_availability(self, material_code: str, new_availability: int) -> bool:
        """材料在庫数更新"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE materials
                    SET availability = ?, updated_at = ?
                    WHERE material_code = ?
                """, (new_availability, datetime.now().isoformat(), material_code))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to update material availability: {e}")
            return False

    # PI Code Management - PIコード管理


    def add_pi_code(self, pi_code: PICode) -> bool:
        """PIコード追加"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pi_codes
                    (pi_code, width_expansion, height_expansion, has_backing,
                     backing_material, backing_thickness, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pi_code.pi_code, pi_code.w_expansion, pi_code.h_expansion,
                    pi_code.has_backing, pi_code.backing_material, getattr(pi_code, 'backing_thickness', 0.0),
                    getattr(pi_code, 'notes', pi_code.description), datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"PI code {pi_code.pi_code} already exists")
            return False
        except Exception as e:
            self.logger.error(f"Failed to add PI code: {e}")
            return False

    def get_pi_code(self, pi_code: str) -> Optional[PICode]:
        """PIコード取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pi_codes WHERE pi_code = ?", (pi_code,))
                row = cursor.fetchone()
                if row:
                    return PICode(
                        pi_code=row['pi_code'],
                        w_expansion=row['width_expansion'],
                        h_expansion=row['height_expansion'],
                        has_backing=bool(row['has_backing']),
                        backing_material=row['backing_material'],
                        thickness=row['backing_thickness'],
                        description=row['notes'],
                        last_updated=row['updated_at']
                    )
                return None
        except Exception as e:
            self.logger.error(f"Failed to get PI code: {e}")
            return None

    # Project Management - プロジェクト管理


    def save_project(self, name: str, panels: List[Panel], description: str = "") -> int:
        """プロジェクト保存"""
        try:
            panel_data = json.dumps([asdict(panel) for panel in panels], ensure_ascii=False)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO projects (name, description, panel_data, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (name, description, panel_data, datetime.now().isoformat()))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to save project: {e}")
            return 0

    def load_project(self, project_id: int) -> Optional[Tuple[str, List[Panel], str]]:
        """プロジェクト読み込み"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM projects WHERE id = ? AND active = 1", (project_id,))
                row = cursor.fetchone()
                if row:
                    panel_data = json.loads(row['panel_data'])
                    panels = [Panel(**data) for data in panel_data]
                    return row['name'], panels, row['description']
                return None
        except Exception as e:
            self.logger.error(f"Failed to load project: {e}")
            return None

    def get_projects(self) -> List[Dict]:
        """プロジェクト一覧取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, description, created_at, updated_at
                    FROM projects WHERE active = 1 ORDER BY updated_at DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get projects: {e}")
            return []

    # Optimization History - 最適化履歴


    def save_optimization_result(self, record: OptimizationRecord) -> int:
        """最適化結果保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO optimization_history
                    (input_panels, constraints, results, algorithm_used, processing_time,
                     total_panels, placed_panels, placement_rate, efficiency, sheets_used, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.input_panels, record.constraints, record.results,
                    record.algorithm_used, record.processing_time, record.total_panels,
                    record.placed_panels, record.placement_rate, record.efficiency,
                    record.sheets_used, record.created_by
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to save optimization result: {e}")
            return 0

    def get_optimization_history(self, limit: int = 50) -> List[Dict]:
        """最適化履歴取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM optimization_history
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return []

    # System Configuration - システム設定


    def set_config(self, key: str, value: str, description: str = "") -> bool:
        """システム設定保存"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO system_config (key, value, description, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (key, value, description, datetime.now().isoformat()))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to set config: {e}")
            return False

    def get_config(self, key: str, default: str = "") -> str:
        """システム設定取得"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row['value'] if row else default
        except Exception as e:
            self.logger.error(f"Failed to get config: {e}")
            return default

    # Migration and Maintenance - 移行・メンテナンス


    def migrate_from_json(self) -> bool:
        """JSONファイルからデータ移行"""
        try:
            success_count = 0

            # Migrate materials
            from core.material_manager import get_material_manager
            material_manager = get_material_manager()
            for material in material_manager.inventory:
                if self.add_material(material):
                    success_count += 1

            # Migrate PI codes
            from core.pi_manager import get_pi_manager
            pi_manager = get_pi_manager()
            for pi_code in pi_manager.pi_codes:
                if self.add_pi_code(pi_code):
                    success_count += 1

            self.logger.info(f"Successfully migrated {success_count} records from JSON files")
            return True

        except Exception as e:
            self.logger.error(f"Failed to migrate from JSON: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """データベースバックアップ"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False

    def get_database_stats(self) -> Dict[str, int]:
        """データベース統計情報"""
        stats = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                tables = ['materials', 'pi_codes', 'projects', 'optimization_history']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[table] = cursor.fetchone()['count']

                return stats
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}


# Singleton instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """データベースマネージャーのシングルトンインスタンス取得"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
