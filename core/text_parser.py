"""
Robust text parser for Japanese steel cutting data
日本語鋼板切断データ用の堅牢なテキストパーサー
"""

import re
import json
import csv
import io
import unicodedata
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from core.models import Panel
import logging

try:
    import jaconv
    try:
        import mojimoji
        MOJIMOJI_SUPPORT = True
    except ImportError:
        MOJIMOJI_SUPPORT = False
    JAPANESE_SUPPORT = True
except ImportError:
    JAPANESE_SUPPORT = False
    MOJIMOJI_SUPPORT = False

# Log status after imports
if not JAPANESE_SUPPORT:
    logging.warning("Japanese text processing libraries not available. Install jaconv and mojimoji for full support.")
elif not MOJIMOJI_SUPPORT:
    logging.info("jaconv installed. For full Japanese text support, install mojimoji as well.")


@dataclass
class ParseError:
    """Error information for parsing failures"""
    line_number: int
    raw_data: str
    error_message: str
    suggested_fix: Optional[str] = None


@dataclass
class ParseResult:
    """Result of text parsing operation"""
    panels: List[Panel]
    errors: List[ParseError]
    warnings: List[str]
    format_detected: str
    total_lines: int
    success_rate: float
    
    @property
    def is_successful(self) -> bool:
        """Check if parsing was generally successful"""
        return len(self.panels) > 0 and self.success_rate > 0.5


class RobustTextParser:
    """
    Robust text data parser with Japanese support
    日本語対応の堅牢なテキストデータパーサー
    """
    
    def __init__(self, encoding: str = 'utf-8', unit_system: str = 'mm'):
        self.encoding = encoding
        self.unit_system = unit_system
        self.logger = logging.getLogger(__name__)

        # Material mapping based on actual inventory
        self.material_mapping = {
            # Japanese generic terms
            'ステンレス': 'SECC',
            'ステンレス鋼': 'SECC',
            '炭素鋼': 'SECC',
            '一般構造用鋼': 'SECC',
            '鉄': 'SECC',
            '鋼板': 'SECC',

            # Actual inventory material codes
            'SE/E24': 'SECC',    # Maps to SECC
            'SE/E8': 'SECC',     # Maps to SECC
            'S203': 'S-203',     # Normalize naming

            # KW code normalization (all to no-hyphen format)
            'KW90': 'KW90',
            'KW-90': 'KW90',     # Remove hyphen
            'KW100': 'KW100',
            'KW-100': 'KW100',   # Remove hyphen
            'KW300': 'KW300',
            'KW-300': 'KW300',   # Remove hyphen
            'KW400': 'KW400',
            'KW-400': 'KW400',   # Remove hyphen

            # Keep existing codes as-is
            'SECC': 'SECC',
            'SGCC': 'SGCC',      # For 0.4mm blank materials
            'E-238P': 'E-238P',
            'E-201P': 'E-201P',
            'E-203P': 'E-203P',
            'S-201': 'S-201',
            'S-203': 'S-203',
            'S-232': 'S-232',
            'S-WHT': 'S-WHT',
            'E-232D': 'E-232D',
            'E-7017': 'E-7017',
            'E-1259P': 'E-1259P',
            'E-2054P': 'E-2054P',
            'LG-011': 'LG-011',
            'GS/E24': 'GS-E24',
            'GS/E8': 'GS-E8'
        }

        # Common delimiters for auto-detection
        self.delimiters = [',', '\t', ';', '|', ' ']

        # Sample data format mappings
        self.cutting_data_headers = {
            '製造番号': 'manufacturing_no',
            'PI': 'pi_code',
            '部材名': 'part_name',
            'W': 'width',
            'H': 'height',
            '数量': 'quantity',
            '色': 'color',
            '板厚': 'thickness',
            '識別番号': 'id_number'
        }

        self.material_data_headers = {
            '資材コー': 'material_code',
            '材質': 'material_type',
            'T': 'thickness',
            'W': 'width',
            'H': 'height',
            '面積': 'area'
        }
        
    def normalize_japanese_text(self, text: str) -> str:
        """
        Normalize Japanese text input
        日本語テキスト入力の正規化
        """
        if not JAPANESE_SUPPORT:
            return text
            
        try:
            # Convert full-width numbers and symbols to half-width
            if MOJIMOJI_SUPPORT:
                text = mojimoji.zen_to_han(text, kana=False)
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            # Convert Japanese decimal point
            text = text.replace('．', '.')
            return text
        except Exception as e:
            self.logger.warning(f"Japanese text normalization failed: {e}")
            return text
    
    def detect_format(self, raw_data: str) -> str:
        """
        Auto-detect data format
        データフォーマットの自動検出
        """
        # Clean sample for analysis
        sample = raw_data[:500].strip()
        
        # JSON detection
        if sample.startswith('{') or sample.startswith('['):
            try:
                json.loads(sample)
                return 'json'
            except:
                pass
        
        # Count delimiter occurrences
        delimiter_scores = {}
        lines = sample.split('\n')[:5]  # Check first 5 lines
        
        for delimiter in self.delimiters:
            scores = []
            for line in lines:
                if line.strip():
                    scores.append(line.count(delimiter))
            
            if scores:
                # Check consistency (similar count across lines)
                avg_count = sum(scores) / len(scores)
                consistency = 1 - (max(scores) - min(scores)) / (avg_count + 1)
                delimiter_scores[delimiter] = avg_count * consistency
        
        if delimiter_scores:
            best_delimiter = max(delimiter_scores.keys(), key=lambda k: delimiter_scores[k])
            
            if best_delimiter == ',':
                return 'csv'
            elif best_delimiter == '\t':
                # Check if it's manufacturing TSV format
                if any('製造番号' in line and 'PI' in line for line in lines):
                    return 'manufacturing_tsv'
                return 'tsv'
            elif delimiter_scores[best_delimiter] > 1:
                return f'delimited_{best_delimiter}'
        
        return 'fixed_width'
    
    def parse_csv_line(self, line: str, line_num: int) -> Tuple[Optional[Panel], Optional[ParseError]]:
        """Parse a single CSV line into Panel object"""
        try:
            # Split by comma, handle quoted fields
            reader = csv.reader(io.StringIO(line))
            fields = next(reader)
            
            if len(fields) < 6:
                return None, ParseError(
                    line_num, line,
                    f"Insufficient fields: expected 6+, got {len(fields)}",
                    "Format: id,width,height,quantity,material,thickness"
                )
            
            # Extract and validate fields
            panel_id = fields[0].strip()
            width = self._parse_number(fields[1])
            height = self._parse_number(fields[2])
            quantity = int(float(fields[3]))
            material = self._normalize_material(fields[4].strip())
            thickness = self._parse_number(fields[5])
            
            # Optional fields
            priority = int(float(fields[6])) if len(fields) > 6 and fields[6].strip() else 1
            allow_rotation = self._parse_boolean(fields[7]) if len(fields) > 7 else True
            pi_code = fields[8].strip() if len(fields) > 8 and fields[8].strip() else ""
            
            # Unit conversion if needed
            if self.unit_system == 'inch':
                width *= 25.4
                height *= 25.4
                thickness *= 25.4
            
            panel = Panel(
                id=panel_id,
                width=width,
                height=height,
                quantity=quantity,
                material=material,
                thickness=thickness,
                priority=priority,
                allow_rotation=allow_rotation,
                pi_code=pi_code
            )
            
            return panel, None
            
        except ValueError as e:
            return None, ParseError(
                line_num, line,
                f"Value error: {str(e)}",
                "Check numeric values and data types"
            )
        except Exception as e:
            return None, ParseError(
                line_num, line,
                f"Parse error: {str(e)}",
                "Check data format and encoding"
            )
    
    def parse_tsv_line(self, line: str, line_num: int) -> Tuple[Optional[Panel], Optional[ParseError]]:
        """Parse tab-separated line"""
        # Replace tabs with commas for CSV parser
        csv_line = line.replace('\t', ',')
        return self.parse_csv_line(csv_line, line_num)
    
    def parse_json_data(self, raw_data: str) -> Tuple[List[Panel], List[ParseError]]:
        """Parse JSON format data"""
        panels = []
        errors = []
        
        try:
            data = json.loads(raw_data)
            
            if isinstance(data, dict) and 'panels' in data:
                data = data['panels']
            
            if not isinstance(data, list):
                errors.append(ParseError(
                    0, raw_data[:100],
                    "JSON must be array or object with 'panels' key"
                ))
                return panels, errors
            
            for i, item in enumerate(data):
                try:
                    panel = Panel(
                        id=str(item.get('id', f'panel_{i+1}')),
                        width=float(item['width']),
                        height=float(item['height']),
                        quantity=int(item.get('quantity', 1)),
                        material=self._normalize_material(item.get('material', 'SS400')),
                        thickness=float(item.get('thickness', 6.0)),
                        priority=int(item.get('priority', 1)),
                        allow_rotation=bool(item.get('allow_rotation', True)),
                        pi_code=str(item.get('pi_code', ''))
                    )
                    panels.append(panel)
                except (KeyError, ValueError, TypeError) as e:
                    errors.append(ParseError(
                        i+1, str(item),
                        f"JSON item error: {str(e)}"
                    ))
        
        except json.JSONDecodeError as e:
            errors.append(ParseError(
                0, raw_data[:100],
                f"Invalid JSON: {str(e)}"
            ))
        
        return panels, errors
    
    def parse_fixed_width(self, raw_data: str) -> Tuple[List[Panel], List[ParseError]]:
        """Parse fixed-width format data"""
        panels = []
        errors = []
        
        lines = raw_data.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                # Attempt to extract fields using regex patterns
                # Pattern for: ID + numbers (width, height, qty, thickness)
                pattern = r'(\S+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\S+)\s+(\d+(?:\.\d+)?)'
                match = re.match(pattern, line)
                
                if match:
                    panel_id, width, height, quantity, material, thickness = match.groups()
                    
                    panel = Panel(
                        id=panel_id,
                        width=float(width),
                        height=float(height),
                        quantity=int(quantity),
                        material=self._normalize_material(material),
                        thickness=float(thickness)
                    )
                    panels.append(panel)
                else:
                    errors.append(ParseError(
                        line_num, line,
                        "Could not parse fixed-width format",
                        "Expected: ID width height quantity material thickness"
                    ))
            
            except Exception as e:
                errors.append(ParseError(
                    line_num, line,
                    f"Fixed-width parse error: {str(e)}"
                ))

        return panels, errors

    def parse_manufacturing_tsv(self, raw_data: str) -> Tuple[List[Panel], List[ParseError]]:
        """
        Parse manufacturing TSV format like data0923.txt
        製造用TSV形式（data0923.txt等）をパース

        Expected format columns:
        製造番号 PI 部材名 W H 寸法3 数量 識別番号 品名 色 板厚 ...
        """
        panels = []
        errors = []

        lines = raw_data.strip().split('\n')
        header_found = False

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # Skip header lines
            if not header_found:
                if '製造番号' in line and 'PI' in line and 'W' in line and 'H' in line:
                    header_found = True
                continue

            if not header_found:
                continue

            try:
                # Split by tab
                fields = line.split('\t')

                if len(fields) < 11:  # Minimum required fields
                    errors.append(ParseError(
                        line_num, line,
                        f"Insufficient fields: expected 11+, got {len(fields)}",
                        "Format: 製造番号 PI 部材名 W H 寸法3 数量 識別番号 品名 色 板厚 ..."
                    ))
                    continue

                # Extract fields according to the sample data format
                manufacturing_no = fields[0].strip()
                pi_code = fields[1].strip()
                part_name = fields[2].strip()
                width = self._parse_number(fields[3])  # W (完成寸法)
                height = self._parse_number(fields[4])  # H (完成寸法)
                # fields[5] is 寸法3 (usually 0)
                quantity = int(float(fields[6]))  # 数量
                # fields[7] is 識別番号
                product_name = fields[8].strip()  # 品名
                color = fields[9].strip()  # 色
                thickness = self._parse_number(fields[10])  # 板厚

                # Create panel ID from manufacturing number and part name
                panel_id = f"{manufacturing_no}_{part_name}" if part_name else manufacturing_no

                # Extract material from product name or use default
                material = self._extract_material_from_name(product_name)
                if not material:
                    material = 'SECC'  # Default for steel panels

                panel = Panel(
                    id=panel_id,
                    width=width,
                    height=height,
                    quantity=quantity,
                    material=material,
                    thickness=thickness,
                    priority=1,
                    allow_rotation=True,
                    pi_code=pi_code
                )

                panels.append(panel)

            except (ValueError, IndexError) as e:
                errors.append(ParseError(
                    line_num, line,
                    f"Manufacturing TSV parse error: {str(e)}",
                    "Check manufacturing data format"
                ))
            except Exception as e:
                errors.append(ParseError(
                    line_num, line,
                    f"Unexpected error: {str(e)}"
                ))

        return panels, errors

    def _extract_material_from_name(self, product_name: str) -> str:
        """Extract material type from product name"""
        if not product_name:
            return ""

        product_name = product_name.upper()

        # Common material patterns in Japanese manufacturing
        material_patterns = {
            'SECC': ['SECC', 'ＳＥＣＣ', '鋼板', 'STEEL'],
            'SGCC': ['SGCC', 'ＳＧＣＣ', 'ガルバ', 'GALVA'],
            'SPCC': ['SPCC', 'ＳＰＣＣ'],
            'SS400': ['SS400', 'ＳＳ４００'],
            'SUS': ['SUS', 'ＳＵＳ', 'ステンレス', 'STAINLESS'],
            'AL': ['AL', 'ＡＬ', 'アルミ', 'ALUMINUM']
        }

        for material, patterns in material_patterns.items():
            for pattern in patterns:
                if pattern in product_name:
                    return material

        # Default fallback for steel panels
        if any(keyword in product_name for keyword in ['パネル', 'PANEL', 'LUX', 'SLUX']):
            return 'SECC'

        return ""

    def parse_cutting_data_tsv(self, raw_data: str) -> Tuple[List[Panel], List[ParseError]]:
        """
        Parse data0923.txt format - Japanese cutting data
        """
        panels = []
        errors = []
        lines = raw_data.strip().split('\n')

        # Find header line (contains 製造番号)
        header_line_idx = -1
        for i, line in enumerate(lines):
            if '製造番号' in line and 'W' in line and 'H' in line:
                header_line_idx = i
                break

        if header_line_idx == -1:
            errors.append(ParseError(
                0, raw_data[:100],
                "Could not find header line with required fields (製造番号, W, H)"
            ))
            return panels, errors

        # Parse header to get field positions
        header_fields = lines[header_line_idx].split('\t')
        field_map = {}
        for i, field in enumerate(header_fields):
            field = field.strip()
            if field in self.cutting_data_headers:
                field_map[self.cutting_data_headers[field]] = i

        # Verify required fields
        required_fields = ['manufacturing_no', 'width', 'height', 'quantity', 'thickness']
        missing_fields = [f for f in required_fields if f not in field_map]
        if missing_fields:
            errors.append(ParseError(
                header_line_idx + 1, lines[header_line_idx],
                f"Missing required fields: {missing_fields}"
            ))
            return panels, errors

        # Parse data lines
        for line_num, line in enumerate(lines[header_line_idx + 1:], header_line_idx + 2):
            line = line.strip()
            if not line:
                continue

            try:
                fields = line.split('\t')
                if len(fields) < max(field_map.values()) + 1:
                    errors.append(ParseError(
                        line_num, line,
                        f"Insufficient fields: expected at least {max(field_map.values()) + 1}, got {len(fields)}"
                    ))
                    continue

                # Extract values using field mapping
                manufacturing_no = fields[field_map['manufacturing_no']].strip()
                width = self._parse_number(fields[field_map['width']])
                height = self._parse_number(fields[field_map['height']])
                quantity = int(float(fields[field_map['quantity']]))
                thickness = self._parse_number(fields[field_map['thickness']])

                # Extract PI code if available
                pi_code = ""
                if 'pi_code' in field_map and len(fields) > field_map['pi_code']:
                    pi_code = fields[field_map['pi_code']].strip()

                # Extract material from color field if available
                material = 'SS400'  # default
                if 'color' in field_map and len(fields) > field_map['color']:
                    color_field = fields[field_map['color']].strip()
                    if color_field:  # Only process non-empty color field
                        normalized_material = self._normalize_material(color_field)
                        if normalized_material:  # Only use if normalization returns non-empty result
                            material = normalized_material

                # Create panel ID from manufacturing number and optional part name
                panel_id = manufacturing_no
                if 'part_name' in field_map and len(fields) > field_map['part_name']:
                    part_name = fields[field_map['part_name']].strip()
                    if part_name:
                        panel_id = f"{manufacturing_no}_{part_name}"

                panel = Panel(
                    id=panel_id,
                    width=width,
                    height=height,
                    quantity=quantity,
                    material=material,
                    thickness=thickness,
                    priority=1,
                    allow_rotation=True,
                    pi_code=pi_code
                )
                panels.append(panel)

            except (ValueError, IndexError) as e:
                errors.append(ParseError(
                    line_num, line,
                    f"Cutting data parse error: {str(e)}"
                ))

        return panels, errors

    def parse_material_data_tsv(self, raw_data: str) -> Tuple[List[Dict], List[ParseError]]:
        """
        Parse sizaidata.txt format - Material inventory data
        Returns list of material stock info rather than panels
        """
        materials = []
        errors = []
        lines = raw_data.strip().split('\n')

        # Find header line (contains 資材コー)
        header_line_idx = -1
        for i, line in enumerate(lines):
            if '資材コー' in line and '材質' in line:
                header_line_idx = i
                break

        if header_line_idx == -1:
            errors.append(ParseError(
                0, raw_data[:100],
                "Could not find header line with required fields (資材コー, 材質)"
            ))
            return materials, errors

        # Parse header to get field positions
        header_fields = lines[header_line_idx].split('\t')
        field_map = {}
        for i, field in enumerate(header_fields):
            field = field.strip()
            if field in self.material_data_headers:
                field_map[self.material_data_headers[field]] = i

        # Parse data lines
        for line_num, line in enumerate(lines[header_line_idx + 1:], header_line_idx + 2):
            line = line.strip()
            if not line:
                continue

            try:
                fields = line.split('\t')
                if len(fields) < max(field_map.values()) + 1:
                    errors.append(ParseError(
                        line_num, line,
                        f"Insufficient fields: expected at least {max(field_map.values()) + 1}, got {len(fields)}"
                    ))
                    continue

                material_info = {}
                for std_field, field_idx in field_map.items():
                    if field_idx < len(fields):
                        value = fields[field_idx].strip()
                        if std_field in ['thickness', 'width', 'height', 'area']:
                            material_info[std_field] = self._parse_number(value) if value else 0.0
                        else:
                            material_info[std_field] = value

                # Handle blank material type with 0.4mm thickness -> SGCC
                if 'material_type' in material_info and 'thickness' in material_info:
                    mat_type = material_info['material_type']
                    thickness = material_info['thickness']

                    if (not mat_type or mat_type == '') and abs(thickness - 0.4) < 0.01:
                        material_info['material_type'] = 'SGCC'
                    elif mat_type:
                        material_info['material_type'] = self._normalize_material(mat_type)

                materials.append(material_info)

            except (ValueError, IndexError) as e:
                errors.append(ParseError(
                    line_num, line,
                    f"Material data parse error: {str(e)}"
                ))

        return materials, errors

    def detect_sample_data_format(self, raw_data: str) -> str:
        """
        Detect if this is sample data format (cutting data or material data)
        """
        sample = raw_data[:1000]

        if '製造番号' in sample and 'PI' in sample and '部材名' in sample:
            return 'cutting_data_tsv'
        elif '資材コー' in sample and '材質' in sample:
            return 'material_data_tsv'

        return 'unknown'

    def _parse_number(self, value: str) -> float:
        """Parse number with Japanese decimal handling"""
        if not value:
            raise ValueError("Empty numeric value")
        
        # Normalize Japanese text
        value = self.normalize_japanese_text(str(value).strip())
        
        # Remove any non-numeric characters except decimal point
        value = re.sub(r'[^\d\.-]', '', value)
        
        if not value:
            raise ValueError("No numeric content found")
        
        return float(value)
    
    def _parse_boolean(self, value: str) -> bool:
        """Parse boolean value with Japanese support"""
        if not value:
            return True
        
        value = value.strip().lower()
        true_values = ['true', 't', '1', 'yes', 'y', 'はい', '可', 'ok']
        false_values = ['false', 'f', '0', 'no', 'n', 'いいえ', '不可', 'ng']
        
        if value in true_values:
            return True
        elif value in false_values:
            return False
        else:
            return True  # Default to True
    
    def _normalize_material(self, material: str) -> str:
        """Normalize material names with Japanese mapping"""
        material = material.strip()
        original = material

        # Return empty string if input is empty (let caller handle default)
        if not material:
            return ""

        # First, normalize KW-XXX format to KWXXX (remove hyphen)
        if material.startswith('KW-') and len(material) > 3:
            # Convert KW-300 to KW300, KW-90 to KW90, etc.
            normalized_kw = 'KW' + material[3:]
            logging.debug(f"Material normalization: {original} → {normalized_kw} (KW hyphen removal)")
            return normalized_kw

        # Check Japanese mapping
        for jp_name, std_name in self.material_mapping.items():
            if jp_name in material:
                logging.debug(f"Material normalization: {original} → {std_name} (mapping)")
                return std_name

        # Return as-is if no mapping found
        result = material.upper()
        if result != original:
            logging.debug(f"Material normalization: {original} → {result} (uppercase)")
        return result
    
    def validate_and_fix_panels(self, panels: List[Panel]) -> Tuple[List[Panel], List[str]]:
        """
        Validate and attempt to fix panel data
        パネルデータの検証と修正
        """
        fixed_panels = []
        warnings = []
        
        for panel in panels:
            try:
                # Try rotation if panel doesn't fit standard sheet
                if panel.width > 1500 or panel.height > 3100:
                    if panel.allow_rotation and panel.height <= 1500 and panel.width <= 3100:
                        # Rotate panel
                        panel.width, panel.height = panel.height, panel.width
                        warnings.append(f"Panel {panel.id} rotated to fit sheet constraints")
                    else:
                        warnings.append(f"Panel {panel.id} exceeds maximum sheet size")
                        continue
                
                # Validate constraints
                panel.validate_size()
                fixed_panels.append(panel)
                
            except ValueError as e:
                warnings.append(f"Panel {panel.id} validation failed: {str(e)}")
        
        return fixed_panels, warnings
    
    def parse_to_panels(self, raw_data: str, format_hint: Optional[str] = None) -> ParseResult:
        """
        Main parsing method - converts text data to Panel objects
        メイン解析メソッド - テキストデータをPanelオブジェクトに変換
        """
        if not raw_data or not raw_data.strip():
            return ParseResult([], [], ["No data provided"], "empty", 0, 0.0)
        
        # Normalize input
        raw_data = self.normalize_japanese_text(raw_data)
        
        # Detect format
        sample_format = self.detect_sample_data_format(raw_data)
        if sample_format != 'unknown':
            format_type = sample_format
        else:
            format_type = format_hint or self.detect_format(raw_data)

        panels = []
        errors = []
        warnings = []

        # Parse based on detected format
        if format_type == 'json':
            panels, errors = self.parse_json_data(raw_data)
        elif format_type == 'manufacturing_tsv':
            panels, errors = self.parse_manufacturing_tsv(raw_data)
        elif format_type == 'cutting_data_tsv':
            panels, errors = self.parse_cutting_data_tsv(raw_data)
        elif format_type == 'material_data_tsv':
            # Material data doesn't create panels directly, but we can report the materials found
            materials, errors = self.parse_material_data_tsv(raw_data)
            warnings.append(f"Material data format detected: {len(materials)} materials found. Use parse_material_data() method for material inventory.")
        else:
            # Line-based parsing for CSV/TSV/fixed-width
            lines = raw_data.strip().split('\n')
            total_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                panel = None
                error = None
                
                if format_type == 'csv':
                    panel, error = self.parse_csv_line(line, line_num)
                elif format_type == 'tsv':
                    panel, error = self.parse_tsv_line(line, line_num)
                elif format_type.startswith('delimited_'):
                    delimiter = format_type.split('_')[1]
                    csv_line = line.replace(delimiter, ',')
                    panel, error = self.parse_csv_line(csv_line, line_num)
                else:  # fixed_width
                    temp_panels, temp_errors = self.parse_fixed_width(line)
                    panels.extend(temp_panels)
                    errors.extend(temp_errors)
                    continue
                
                if panel:
                    panels.append(panel)
                if error:
                    errors.append(error)
        
        # Validate and fix panels
        panels, validation_warnings = self.validate_and_fix_panels(panels)
        warnings.extend(validation_warnings)
        
        # Calculate success rate
        total_lines = len(raw_data.strip().split('\n'))
        success_rate = len(panels) / max(total_lines, 1)
        
        return ParseResult(
            panels=panels,
            errors=errors,
            warnings=warnings,
            format_detected=format_type,
            total_lines=total_lines,
            success_rate=success_rate
        )


# Convenience functions
def parse_csv_file(file_path: str) -> ParseResult:
    """Parse CSV file to panels"""
    parser = RobustTextParser()
    with open(file_path, 'r', encoding='utf-8') as f:
        return parser.parse_to_panels(f.read(), 'csv')


def parse_text_data(text: str, format_hint: Optional[str] = None) -> ParseResult:
    """Quick parse function for text data"""
    parser = RobustTextParser()
    return parser.parse_to_panels(text, format_hint)


def parse_cutting_data_file(file_path: str) -> ParseResult:
    """Parse Japanese cutting data file (data0923.txt format)"""
    parser = RobustTextParser()
    with open(file_path, 'r', encoding='utf-8') as f:
        return parser.parse_to_panels(f.read(), 'cutting_data_tsv')


def parse_material_data_file(file_path: str) -> Tuple[List[Dict], List[ParseError]]:
    """Parse Japanese material data file (sizaidata.txt format)"""
    parser = RobustTextParser()
    with open(file_path, 'r', encoding='utf-8') as f:
        return parser.parse_material_data_tsv(f.read())


def parse_sample_data_files(cutting_file: str, material_file: str) -> Tuple[ParseResult, List[Dict], List[ParseError]]:
    """
    Parse both sample data files together
    Returns: (cutting_panels_result, material_inventory, combined_errors)
    """
    cutting_result = parse_cutting_data_file(cutting_file)
    materials, material_errors = parse_material_data_file(material_file)

    # Combine errors
    all_errors = cutting_result.errors + material_errors

    return cutting_result, materials, all_errors