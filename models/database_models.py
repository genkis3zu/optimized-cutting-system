#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database models for material and PI code data
材料とPIコードデータのデータベースモデル
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import sqlite3
import csv
from pathlib import Path


@dataclass
class MaterialData:
    """
    Material specification from sizaidata.txt
    sizaidata.txtからの材料仕様
    """
    material_code: str          # 資材コード
    material_type: str          # 材質 (SE/E24, SGCC, etc.)
    thickness: float           # T (板厚)
    width: float              # W (幅)
    height: float             # H (高さ)
    spec_name: str            # syuko (仕様名)
    unit: int                 # 単位 (通常2)
    area: float               # 面積
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def display_name(self) -> str:
        """Display-friendly material name"""
        return f"{self.material_type} {self.thickness}t {self.width}x{self.height}"

    @property
    def is_standard_sheet(self) -> bool:
        """Check if this is a standard cutting sheet (1500x3100)"""
        return (abs(self.width - 1500) < 10 and abs(self.height - 3100) < 10) or \
               (abs(self.width - 3100) < 10 and abs(self.height - 1500) < 10)


@dataclass
class PiCodeData:
    """
    PI code dimension expansion data from pi.txt
    pi.txtからのPIコード寸法展開データ
    """
    pi_code: str               # PIコード
    height_expansion: float    # H寸法展開
    width_expansion: float     # W寸法展開
    back_plate: int           # 裏板フラグ (0/1)
    back_plate_h: float = 0.0 # 裏板H寸法
    back_plate_w: float = 0.0 # 裏板W寸法
    plaster: int = 0          # プラスタフラグ (0/1)
    plaster_h: float = 0.0    # プラスタH寸法
    plaster_w: float = 0.0    # プラスタW寸法
    plate_thickness: float = 0.0  # 板厚
    created_at: datetime = field(default_factory=datetime.now)

    def expand_dimensions(self, finished_w: float, finished_h: float) -> Tuple[float, float]:
        """
        Calculate expanded dimensions from finished dimensions
        完成寸法から展開寸法を計算

        Args:
            finished_w: 完成W寸法
            finished_h: 完成H寸法

        Returns:
            Tuple[expanded_w, expanded_h]: 展開W寸法, 展開H寸法
        """
        expanded_w = finished_w + self.width_expansion
        expanded_h = finished_h + self.height_expansion

        # Apply plaster adjustments if enabled
        if self.plaster == 1:
            expanded_w += self.plaster_w
            expanded_h += self.plaster_h

        # Ensure minimum dimensions
        expanded_w = max(50.0, expanded_w)  # Minimum 50mm
        expanded_h = max(50.0, expanded_h)  # Minimum 50mm

        return expanded_w, expanded_h


class MaterialDatabase:
    """
    Database manager for material and PI code data
    材料とPIコードデータのデータベース管理
    """

    def __init__(self, db_path: str = "data/materials.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Materials table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS materials (
                    material_code TEXT PRIMARY KEY,
                    material_type TEXT NOT NULL,
                    thickness REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    spec_name TEXT NOT NULL,
                    unit INTEGER NOT NULL,
                    area REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # PI codes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pi_codes (
                    pi_code TEXT PRIMARY KEY,
                    height_expansion REAL NOT NULL,
                    width_expansion REAL NOT NULL,
                    back_plate INTEGER DEFAULT 0,
                    back_plate_h REAL DEFAULT 0.0,
                    back_plate_w REAL DEFAULT 0.0,
                    plaster INTEGER DEFAULT 0,
                    plaster_h REAL DEFAULT 0.0,
                    plaster_w REAL DEFAULT 0.0,
                    plate_thickness REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    def load_materials_from_file(self, file_path: str) -> int:
        """
        Load material data from sizaidata.txt
        sizaidata.txtから材料データを読み込み

        Returns:
            Number of materials loaded
        """
        loaded_count = 0

        try:
            # Try different encodings
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"Could not decode file with any encoding: {file_path}")
                return 0

            with sqlite3.connect(self.db_path) as conn:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line or i == 0:  # Skip header
                        continue

                    try:
                        # Parse tab-separated values
                        parts = line.split('\t')
                        if len(parts) < 7:  # Minimum required fields
                            continue

                        # Skip lines with header content or invalid data
                        material_code = parts[0].strip()
                        if not material_code or material_code == '資材コー':
                            continue

                        material_type = parts[1].strip()
                        thickness_str = parts[2].strip()
                        width_str = parts[3].strip()
                        height_str = parts[4].strip()
                        spec_name = parts[5].strip()

                        # Handle unit field (may have empty strings)
                        unit = 2  # Default unit
                        try:
                            unit_str = parts[6].strip()
                            if unit_str:
                                unit = int(unit_str)
                        except (ValueError, IndexError):
                            pass

                        # Handle area field (may be at different positions due to empty columns)
                        area = 0.0
                        for j in range(7, len(parts)):
                            area_str = parts[j].strip()
                            if area_str:
                                try:
                                    area = float(area_str)
                                    break
                                except ValueError:
                                    continue

                        # Validate and convert numeric fields
                        if not thickness_str or not width_str or not height_str:
                            continue

                        thickness = float(thickness_str)
                        width = float(width_str)
                        height = float(height_str)

                        # Calculate area if not provided
                        if area == 0.0:
                            area = width * height

                        material = MaterialData(
                            material_code=material_code,
                            material_type=material_type,
                            thickness=thickness,
                            width=width,
                            height=height,
                            spec_name=spec_name,
                            unit=unit,
                            area=area
                        )

                        # Insert or replace
                        conn.execute('''
                            INSERT OR REPLACE INTO materials
                            (material_code, material_type, thickness, width, height, spec_name, unit, area)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            material.material_code, material.material_type, material.thickness,
                            material.width, material.height, material.spec_name,
                            material.unit, material.area
                        ))

                        loaded_count += 1

                    except (ValueError, IndexError) as e:
                        # Only print first few warnings to avoid spam
                        if loaded_count < 5:
                            print(f"Warning: Failed to parse line {i+1}: {line[:50]}... - {e}")
                        continue

                conn.commit()

        except FileNotFoundError:
            print(f"Material data file not found: {file_path}")
        except Exception as e:
            print(f"Error loading material data: {e}")

        return loaded_count

    def load_pi_codes_from_file(self, file_path: str) -> int:
        """
        Load PI code data from pi.txt
        pi.txtからPIコードデータを読み込み

        Returns:
            Number of PI codes loaded
        """
        loaded_count = 0

        try:
            # Try different encodings
            for encoding in ['utf-8', 'shift-jis', 'cp932']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"Could not decode PI file with any encoding: {file_path}")
                return 0

            with sqlite3.connect(self.db_path) as conn:
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line or i == 0:  # Skip header
                        continue

                    try:
                        # Parse tab-separated values
                        parts = line.split('\t')
                        if len(parts) < 3:  # Minimum required fields
                            continue

                        # Skip header lines
                        pi_code_str = parts[0].strip()
                        if not pi_code_str or 'コード' in pi_code_str:
                            continue

                        height_exp_str = parts[1].strip() if len(parts) > 1 else ""
                        width_exp_str = parts[2].strip() if len(parts) > 2 else ""

                        # Skip lines with non-numeric expansions or material codes mixed in
                        try:
                            height_expansion = float(height_exp_str) if height_exp_str else 0.0
                            width_expansion = float(width_exp_str) if width_exp_str else 0.0
                        except ValueError:
                            # Skip lines where expansion values contain text (like material codes)
                            continue

                        # Handle back plate info
                        back_plate = 0
                        back_plate_h = 0.0
                        back_plate_w = 0.0

                        if len(parts) > 3:
                            try:
                                back_plate_str = parts[3].strip()
                                if back_plate_str and not back_plate_str.isalpha():  # Skip text fields
                                    back_plate = int(float(back_plate_str))
                            except (ValueError, IndexError):
                                pass

                        if len(parts) > 4:
                            try:
                                back_plate_h = float(parts[4].strip()) if parts[4].strip() else 0.0
                            except (ValueError, IndexError):
                                pass

                        if len(parts) > 5:
                            try:
                                back_plate_w = float(parts[5].strip()) if parts[5].strip() else 0.0
                            except (ValueError, IndexError):
                                pass

                        # Handle plaster info
                        plaster = 0
                        plaster_h = 0.0
                        plaster_w = 0.0

                        if len(parts) > 6:
                            try:
                                plaster = int(float(parts[6].strip())) if parts[6].strip() else 0
                            except (ValueError, IndexError):
                                pass

                        if len(parts) > 7:
                            try:
                                plaster_h = float(parts[7].strip()) if parts[7].strip() else 0.0
                            except (ValueError, IndexError):
                                pass

                        if len(parts) > 8:
                            try:
                                plaster_w = float(parts[8].strip()) if parts[8].strip() else 0.0
                            except (ValueError, IndexError):
                                pass

                        # Handle plate thickness
                        plate_thickness = 0.0
                        if len(parts) > 9:
                            try:
                                plate_thickness = float(parts[9].strip()) if parts[9].strip() else 0.0
                            except (ValueError, IndexError):
                                pass

                        pi_code = PiCodeData(
                            pi_code=pi_code_str,
                            height_expansion=height_expansion,
                            width_expansion=width_expansion,
                            back_plate=back_plate,
                            back_plate_h=back_plate_h,
                            back_plate_w=back_plate_w,
                            plaster=plaster,
                            plaster_h=plaster_h,
                            plaster_w=plaster_w,
                            plate_thickness=plate_thickness
                        )

                        # Insert or replace
                        conn.execute('''
                            INSERT OR REPLACE INTO pi_codes
                            (pi_code, height_expansion, width_expansion, back_plate, back_plate_h, back_plate_w,
                             plaster, plaster_h, plaster_w, plate_thickness)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            pi_code.pi_code, pi_code.height_expansion, pi_code.width_expansion,
                            pi_code.back_plate, pi_code.back_plate_h, pi_code.back_plate_w,
                            pi_code.plaster, pi_code.plaster_h, pi_code.plaster_w,
                            pi_code.plate_thickness
                        ))

                        loaded_count += 1

                    except (ValueError, IndexError) as e:
                        # Only print first few warnings to avoid spam
                        if loaded_count < 5:
                            print(f"Warning: Failed to parse PI line {i+1}: {line[:50]}... - {e}")
                        continue

                conn.commit()

        except FileNotFoundError:
            print(f"PI code data file not found: {file_path}")
        except Exception as e:
            print(f"Error loading PI code data: {e}")

        return loaded_count

    def get_material_by_code(self, material_code: str) -> Optional[MaterialData]:
        """Get material by code"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM materials WHERE material_code = ?',
                (material_code,)
            )
            row = cursor.fetchone()

            if row:
                return MaterialData(
                    material_code=row['material_code'],
                    material_type=row['material_type'],
                    thickness=row['thickness'],
                    width=row['width'],
                    height=row['height'],
                    spec_name=row['spec_name'],
                    unit=row['unit'],
                    area=row['area']
                )
        return None

    def get_pi_code(self, pi_code: str) -> Optional[PiCodeData]:
        """Get PI code data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM pi_codes WHERE pi_code = ?',
                (pi_code,)
            )
            row = cursor.fetchone()

            if row:
                return PiCodeData(
                    pi_code=row['pi_code'],
                    height_expansion=row['height_expansion'],
                    width_expansion=row['width_expansion'],
                    back_plate=row['back_plate'],
                    back_plate_h=row['back_plate_h'],
                    back_plate_w=row['back_plate_w'],
                    plaster=row['plaster'],
                    plaster_h=row['plaster_h'],
                    plaster_w=row['plaster_w'],
                    plate_thickness=row['plate_thickness']
                )
        return None

    def get_all_materials(self) -> List[MaterialData]:
        """Get all materials"""
        materials = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM materials ORDER BY material_type, thickness')

            for row in cursor.fetchall():
                materials.append(MaterialData(
                    material_code=row['material_code'],
                    material_type=row['material_type'],
                    thickness=row['thickness'],
                    width=row['width'],
                    height=row['height'],
                    spec_name=row['spec_name'],
                    unit=row['unit'],
                    area=row['area']
                ))

        return materials

    def get_material_types(self) -> List[str]:
        """Get unique material types"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT material_type FROM materials ORDER BY material_type')
            return [row[0] for row in cursor.fetchall()]

    def get_standard_sheets(self) -> List[MaterialData]:
        """Get standard sheet materials (1500x3100)"""
        materials = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM materials
                WHERE (width = 1500 AND height = 3100) OR (width = 3100 AND height = 1500)
                ORDER BY material_type, thickness
            ''')

            for row in cursor.fetchall():
                materials.append(MaterialData(
                    material_code=row['material_code'],
                    material_type=row['material_type'],
                    thickness=row['thickness'],
                    width=row['width'],
                    height=row['height'],
                    spec_name=row['spec_name'],
                    unit=row['unit'],
                    area=row['area']
                ))

        return materials

    def get_all_pi_codes(self) -> List[PiCodeData]:
        """Get all PI codes from database"""
        pi_codes = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM pi_codes ORDER BY pi_code')

            for row in cursor.fetchall():
                pi_codes.append(PiCodeData(
                    pi_code=row['pi_code'],
                    height_expansion=row['height_expansion'],
                    width_expansion=row['width_expansion'],
                    back_plate=row['back_plate'],
                    back_plate_h=row['back_plate_h'],
                    back_plate_w=row['back_plate_w'],
                    plaster=row['plaster'],
                    plaster_h=row['plaster_h'],
                    plaster_w=row['plaster_w'],
                    plate_thickness=row['plate_thickness']
                ))

        return pi_codes

    def get_pi_codes_list(self) -> List[str]:
        """Get list of all PI codes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT DISTINCT pi_code FROM pi_codes ORDER BY pi_code')
            return [row[0] for row in cursor.fetchall()]

    def search_pi_codes(self, search_term: str = "") -> List[PiCodeData]:
        """Search PI codes by partial match"""
        pi_codes = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if search_term:
                cursor = conn.execute('''
                    SELECT * FROM pi_codes
                    WHERE pi_code LIKE ?
                    ORDER BY pi_code
                ''', (f'%{search_term}%',))
            else:
                cursor = conn.execute('SELECT * FROM pi_codes ORDER BY pi_code')

            for row in cursor.fetchall():
                pi_codes.append(PiCodeData(
                    pi_code=row['pi_code'],
                    height_expansion=row['height_expansion'],
                    width_expansion=row['width_expansion'],
                    back_plate=row['back_plate'],
                    back_plate_h=row['back_plate_h'],
                    back_plate_w=row['back_plate_w'],
                    plaster=row['plaster'],
                    plaster_h=row['plaster_h'],
                    plaster_w=row['plaster_w'],
                    plate_thickness=row['plate_thickness']
                ))

        return pi_codes

    def get_pi_code_stats(self) -> Dict[str, any]:
        """Get PI code database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM pi_codes')
            total_codes = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM pi_codes WHERE plaster = 1')
            with_plaster = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(*) FROM pi_codes WHERE back_plate = 1')
            with_back_plate = cursor.fetchone()[0]

            cursor = conn.execute('''
                SELECT AVG(width_expansion), AVG(height_expansion)
                FROM pi_codes
                WHERE width_expansion > 0 AND height_expansion > 0
            ''')
            avg_expansions = cursor.fetchone()

            return {
                'total_pi_codes': total_codes,
                'codes_with_plaster': with_plaster,
                'codes_with_back_plate': with_back_plate,
                'avg_width_expansion': avg_expansions[0] if avg_expansions[0] else 0,
                'avg_height_expansion': avg_expansions[1] if avg_expansions[1] else 0
            }

    def initialize_from_sample_data(self) -> Dict[str, int]:
        """
        Initialize database from sample data files
        サンプルデータファイルからデータベースを初期化

        Returns:
            Dictionary with loading statistics
        """
        results = {
            'materials_loaded': 0,
            'pi_codes_loaded': 0
        }

        # Load materials
        material_file = Path('sample_data/sizaidata.txt')
        if material_file.exists():
            results['materials_loaded'] = self.load_materials_from_file(str(material_file))

        # Load PI codes
        pi_file = Path('sample_data/pi.txt')
        if pi_file.exists():
            results['pi_codes_loaded'] = self.load_pi_codes_from_file(str(pi_file))

        return results