"""
PI Code Management System
PIコード管理システム

Manages PI codes for expanding panel dimensions from finished size to cutting size.
完成寸法から切断寸法への展開計算用PIコード管理
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd


@dataclass
class PICode:
    """PIコード情報を格納するデータクラス"""
    pi_code: str                    # PIコード
    h_expansion: float = 0.0        # H寸法への加算値
    w_expansion: float = 0.0        # W寸法への加算値
    has_backing: bool = False       # 裏板フラグ (0=なし, 1=あり)
    backing_material: str = ""      # 裏板材質
    backing_h: float = 0.0          # 裏板H寸法
    backing_w: float = 0.0          # 裏板W寸法
    has_plaster: bool = True        # プラスタフラグ (0=なし, 1=あり)
    plaster_h: float = 0.0          # プラスタH寸法
    plaster_w: float = 0.0          # プラスタW寸法
    thickness: float = 0.5          # 板厚
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""           # 備考

    def __post_init__(self):
        """データ正規化処理"""
        self.pi_code = self.pi_code.strip().upper()
        self.description = self.description.strip()

    def get_expanded_dimensions(self, original_w: float, original_h: float) -> Tuple[float, float]:
        """
        完成寸法から展開寸法を計算

        Args:
            original_w: 完成時W寸法
            original_h: 完成時H寸法

        Returns:
            Tuple[展開W寸法, 展開H寸法]
        """
        expanded_w = original_w + self.w_expansion
        expanded_h = original_h + self.h_expansion

        # 最小寸法チェック
        expanded_w = max(expanded_w, 50.0)  # 最小50mm
        expanded_h = max(expanded_h, 50.0)  # 最小50mm

        return expanded_w, expanded_h


class PIManager:
    """PIコード管理クラス"""

    def __init__(self, use_database: bool = True):
        """
        Initialize PI Manager
        Args:
            use_database: If True, use SQLite database; if False, use JSON file (legacy)
        """
        self.use_database = use_database
        self.pi_codes: List[PICode] = []
        if use_database:
            self.load_pi_codes_from_database()
        else:
            # Legacy JSON file support
            self.data_file = "data/pi_codes.json"
            self._ensure_directory()
            self.load_pi_codes()

    def load_pi_codes_from_database(self) -> bool:
        """SQLiteデータベースからPIコードを読み込み"""
        try:
            from core.persistence_adapter import get_persistence_adapter
            persistence = get_persistence_adapter()

            # Get all PI codes from database
            import sqlite3
            conn = sqlite3.connect('data/cutting_system.db')
            cursor = conn.cursor()

            cursor.execute('''
                SELECT pi_code, width_expansion, height_expansion, has_backing,
                       backing_material, backing_thickness, notes
                FROM pi_codes
            ''')

            rows = cursor.fetchall()
            self.pi_codes = []

            for row in rows:
                # Convert database format to PICode object
                pi_code_obj = PICode(
                    pi_code=row[0],
                    w_expansion=row[1] or 0.0,
                    h_expansion=row[2] or 0.0,
                    has_backing=bool(row[3]) if row[3] is not None else False,
                    backing_material=row[4] or "",
                    thickness=row[5] or 0.5,
                    description=row[6] or ""
                )
                self.pi_codes.append(pi_code_obj)

            conn.close()
            print(f"データベースから{len(self.pi_codes)}個のPIコードを読み込みました")
            return True

        except Exception as e:
            print(f"データベースからのPIコード読み込みエラー: {e}")
            # Fallback to JSON if database fails
            self.data_file = "data/pi_codes.json"
            self._ensure_directory()
            return self.load_pi_codes()

    def _ensure_directory(self):
        """データディレクトリの作成"""
        if hasattr(self, 'data_file'):
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)

    def load_pi_codes(self) -> bool:
        """PIコードデータをファイルから読み込み"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.pi_codes = [PICode(**item) for item in data]
            else:
                # 初期データとしてサンプルから読み込み
                self._load_sample_data()
            return True
        except Exception as e:
            print(f"PIコードデータ読み込みエラー: {e}")
            self.pi_codes = []
            return False

    def save_pi_codes(self) -> bool:
        """PIコードデータをファイルに保存"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                data = [asdict(pi_code) for pi_code in self.pi_codes]
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"PIコードデータ保存エラー: {e}")
            return False

    def add_pi_code(self, pi_code: PICode) -> bool:
        """PIコードを追加"""
        if any(existing.pi_code == pi_code.pi_code for existing in self.pi_codes):
            return False  # 既に存在

        pi_code.last_updated = datetime.now().isoformat()
        self.pi_codes.append(pi_code)
        self.save_pi_codes()
        return True

    def update_pi_code(self, pi_code: str, updates: Dict) -> bool:
        """PIコードを更新"""
        for existing in self.pi_codes:
            if existing.pi_code == pi_code:
                for key, value in updates.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.last_updated = datetime.now().isoformat()
                self.save_pi_codes()
                return True
        return False

    def remove_pi_code(self, pi_code: str) -> bool:
        """PIコードを削除"""
        original_length = len(self.pi_codes)
        self.pi_codes = [pc for pc in self.pi_codes if pc.pi_code != pi_code]
        if len(self.pi_codes) < original_length:
            self.save_pi_codes()
            return True
        return False

    def get_pi_code(self, pi_code: str) -> Optional[PICode]:
        """PIコードを取得"""
        for pc in self.pi_codes:
            if pc.pi_code == pi_code:
                return pc
        return None

    def get_all_pi_codes(self) -> List[str]:
        """すべてのPIコード一覧を取得"""
        return [pc.pi_code for pc in self.pi_codes]

    def search_pi_codes(self, query: str) -> List[PICode]:
        """PIコードを検索"""
        query = query.lower().strip()
        if not query:
            return self.pi_codes

        results = []
        for pc in self.pi_codes:
            if (query in pc.pi_code.lower() or
                query in pc.description.lower() or
                query in pc.backing_material.lower()):
                results.append(pc)
        return results

    def get_expansion_for_panel(self, pi_code: str, w: float, h: float) -> Tuple[float, float]:
        """
        パネルの展開寸法を計算

        Args:
            pi_code: PIコード
            w: 完成W寸法
            h: 完成H寸法

        Returns:
            Tuple[展開W寸法, 展開H寸法]
        """
        pi = self.get_pi_code(pi_code)
        if pi:
            return pi.get_expanded_dimensions(w, h)
        else:
            # PIコードが見つからない場合は元寸法をそのまま返す
            return w, h

    def get_pi_summary(self) -> Dict:
        """PIコードサマリー情報を取得"""
        total_codes = len(self.pi_codes)
        codes_with_backing = len([pc for pc in self.pi_codes if pc.has_backing])
        unique_materials = len(set(pc.backing_material for pc in self.pi_codes if pc.backing_material))

        return {
            'total_codes': total_codes,
            'codes_with_backing': codes_with_backing,
            'codes_without_backing': total_codes - codes_with_backing,
            'unique_backing_materials': unique_materials,
            'backing_materials': list(set(pc.backing_material for pc in self.pi_codes if pc.backing_material))
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """PIコードデータをDataFrameに変換"""
        if not self.pi_codes:
            return pd.DataFrame()

        data = []
        for pc in self.pi_codes:
            data.append({
                'PIコード': pc.pi_code,
                'H加算': pc.h_expansion,
                'W加算': pc.w_expansion,
                '裏板': '○' if pc.has_backing else '',
                '裏板材質': pc.backing_material,
                '裏板H': pc.backing_h if pc.has_backing else '',
                '裏板W': pc.backing_w if pc.has_backing else '',
                'プラスタ': '○' if pc.has_plaster else '',
                'プラスタH': pc.plaster_h,
                'プラスタW': pc.plaster_w,
                '板厚': pc.thickness,
                '備考': pc.description,
                '更新日': pc.last_updated[:10] if pc.last_updated else ''
            })

        return pd.DataFrame(data)

    def _load_sample_data(self):
        """サンプルデータからPI.txtを読み込み"""
        sample_file = "sample_data/PI.txt"
        if not os.path.exists(sample_file):
            return

        try:
            # TSVファイルとして読み込み
            df = pd.read_csv(sample_file, sep='\t', encoding='utf-8')

            for _, row in df.iterrows():
                try:
                    pi_code = str(row.iloc[0]).strip()
                    if not pi_code or pi_code == 'ＰＩコード':
                        continue

                    # 数値データの安全な変換
                    def safe_float(val, default=0.0):
                        try:
                            if pd.isna(val) or val == '':
                                return default
                            return float(val)
                        except:
                            return default

                    def safe_bool(val):
                        try:
                            if pd.isna(val) or val == '':
                                return False
                            return int(val) == 1
                        except:
                            return False

                    pi = PICode(
                        pi_code=pi_code,
                        h_expansion=safe_float(row.iloc[1], 0.0),
                        w_expansion=safe_float(row.iloc[2], 0.0),
                        has_backing=safe_bool(row.iloc[3]),
                        backing_material=str(row.iloc[4]) if not pd.isna(row.iloc[4]) else "",
                        backing_h=safe_float(row.iloc[5], 0.0),
                        backing_w=safe_float(row.iloc[6], 0.0),
                        has_plaster=safe_bool(row.iloc[7]),
                        plaster_h=safe_float(row.iloc[8], 0.0),
                        plaster_w=safe_float(row.iloc[9], 0.0),
                        thickness=safe_float(row.iloc[10], 0.5),
                        description=str(row.iloc[11]) if len(row) > 11 and not pd.isna(row.iloc[11]) else ""
                    )

                    self.pi_codes.append(pi)

                except Exception as e:
                    print(f"行のスキップ: {e}")
                    continue

            # 初期データを保存
            self.save_pi_codes()
            print(f"サンプルデータから{len(self.pi_codes)}個のPIコードを読み込みました")

        except Exception as e:
            print(f"サンプルデータ読み込みエラー: {e}")


# グローバルインスタンス
_pi_manager = None

def get_pi_manager() -> PIManager:
    """PIManagerのシングルトンインスタンスを取得"""
    global _pi_manager
    if _pi_manager is None:
        _pi_manager = PIManager()
    return _pi_manager