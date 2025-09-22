"""
Material Inventory Management System
材料在庫管理システム

Manages material inventory with persistent storage and validation
"""

import json
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class MaterialSheet:
    """Material sheet specification"""
    material_code: str
    material_type: str
    thickness: float
    width: float
    height: float
    area: float
    cost_per_sheet: float = 15000.0
    availability: int = 100
    supplier: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    @property
    def display_name(self) -> str:
        """Human-readable display name"""
        return f"{self.material_type} {self.thickness}mm ({self.width}×{self.height})"

    def matches_requirements(self, material_type: str, thickness: float,
                           min_width: float, min_height: float) -> bool:
        """Check if this sheet can satisfy panel requirements"""
        return (
            self.material_type == material_type and
            abs(self.thickness - thickness) < 0.1 and  # Allow small thickness tolerance
            self.width >= min_width and
            self.height >= min_height and
            self.availability > 0
        )


class MaterialInventoryManager:
    """
    Material inventory management with persistent storage
    材料在庫管理 (永続化対応)
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.inventory_file = os.path.join(config_dir, "material_inventory.json")
        self.mapping_file = os.path.join(config_dir, "material_mapping.json")
        self.logger = logging.getLogger(__name__)

        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)

        # Load or initialize inventory
        self.inventory: List[MaterialSheet] = []
        self.material_types: Set[str] = set()
        self._load_inventory()

        # Load material mapping
        self.material_mapping = self._load_material_mapping()

    def _load_inventory(self):
        """Load inventory from file"""
        try:
            if os.path.exists(self.inventory_file):
                with open(self.inventory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.inventory = [MaterialSheet(**item) for item in data]
                    self.material_types = set(sheet.material_type for sheet in self.inventory)
                    self.logger.info(f"Loaded {len(self.inventory)} material sheets")
            else:
                self.logger.info("No existing inventory file, starting with empty inventory")
        except Exception as e:
            self.logger.error(f"Failed to load inventory: {e}")
            self.inventory = []
            self.material_types = set()

    def _save_inventory(self):
        """Save inventory to file"""
        try:
            data = [asdict(sheet) for sheet in self.inventory]
            with open(self.inventory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(self.inventory)} material sheets")
        except Exception as e:
            self.logger.error(f"Failed to save inventory: {e}")

    def _load_material_mapping(self) -> Dict[str, str]:
        """Load material code mapping"""
        default_mapping = {
            # Actual inventory material codes from sample data
            'SE/E24': 'SECC',
            'SE/E8': 'SECC',
            'S203': 'S-203',

            # KW code bidirectional mapping for compatibility
            'KW90': 'KW-90',     # Normalize to hyphen format
            'KW-90': 'KW-90',    # Keep consistent
            'KW100': 'KW-100',   # Normalize to hyphen format
            'KW-100': 'KW-100',  # Keep consistent
            'KW300': 'KW-300',   # Normalize to hyphen format
            'KW-300': 'KW-300',  # Keep consistent
            'KW400': 'KW-400',   # Normalize to hyphen format
            'KW-400': 'KW-400',  # Keep consistent

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

        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    return {**default_mapping, **mapping}
        except Exception as e:
            self.logger.warning(f"Failed to load material mapping: {e}")

        return default_mapping

    def _save_material_mapping(self):
        """Save material mapping to file"""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.material_mapping, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save material mapping: {e}")

    def load_from_sample_data(self, material_file: str) -> int:
        """Load materials from sample data file"""
        try:
            from core.text_parser import parse_material_data_file
            materials, errors = parse_material_data_file(material_file)

            added_count = 0
            for material_data in materials:
                sheet = MaterialSheet(
                    material_code=material_data.get('material_code', ''),
                    material_type=material_data.get('material_type', ''),
                    thickness=material_data.get('thickness', 0.0),
                    width=material_data.get('width', 0.0),
                    height=material_data.get('height', 0.0),
                    area=material_data.get('area', 0.0)
                )

                # Check if already exists
                if not any(s.material_code == sheet.material_code for s in self.inventory):
                    self.inventory.append(sheet)
                    self.material_types.add(sheet.material_type)
                    added_count += 1

            self._save_inventory()
            self.logger.info(f"Added {added_count} new material sheets from sample data")
            return added_count

        except Exception as e:
            self.logger.error(f"Failed to load from sample data: {e}")
            return 0

    def add_material_sheet(self, sheet: MaterialSheet) -> bool:
        """Add a material sheet to inventory"""
        try:
            # Check for duplicate material codes
            if any(s.material_code == sheet.material_code for s in self.inventory):
                self.logger.warning(f"Material code {sheet.material_code} already exists")
                return False

            sheet.last_updated = datetime.now().isoformat()
            self.inventory.append(sheet)
            self.material_types.add(sheet.material_type)
            self._save_inventory()
            return True

        except Exception as e:
            self.logger.error(f"Failed to add material sheet: {e}")
            return False

    def update_material_sheet(self, material_code: str, updates: Dict) -> bool:
        """Update an existing material sheet"""
        try:
            for sheet in self.inventory:
                if sheet.material_code == material_code:
                    for key, value in updates.items():
                        if hasattr(sheet, key):
                            setattr(sheet, key, value)
                    sheet.last_updated = datetime.now().isoformat()
                    self._save_inventory()
                    return True

            self.logger.warning(f"Material code {material_code} not found")
            return False

        except Exception as e:
            self.logger.error(f"Failed to update material sheet: {e}")
            return False

    def remove_material_sheet(self, material_code: str) -> bool:
        """Remove a material sheet from inventory"""
        try:
            original_count = len(self.inventory)
            self.inventory = [s for s in self.inventory if s.material_code != material_code]

            if len(self.inventory) < original_count:
                # Update material types set
                self.material_types = set(sheet.material_type for sheet in self.inventory)
                self._save_inventory()
                return True
            else:
                self.logger.warning(f"Material code {material_code} not found")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remove material sheet: {e}")
            return False

    def get_material_sheet(self, material_code: str) -> Optional[MaterialSheet]:
        """Get a specific material sheet"""
        return next((s for s in self.inventory if s.material_code == material_code), None)

    def get_sheets_by_type(self, material_type: str) -> List[MaterialSheet]:
        """Get all sheets of a specific material type"""
        return [s for s in self.inventory if s.material_type == material_type]

    def find_compatible_sheets(self, material_type: str, thickness: float,
                             min_width: float, min_height: float) -> List[MaterialSheet]:
        """Find sheets that can accommodate the given requirements"""
        compatible = []
        for sheet in self.inventory:
            if sheet.matches_requirements(material_type, thickness, min_width, min_height):
                compatible.append(sheet)

        # Sort by area (smaller sheets first for efficiency)
        return sorted(compatible, key=lambda s: s.area)

    def validate_panel_against_inventory(self, material_type: str, thickness: float,
                                       width: float, height: float) -> Tuple[bool, str]:
        """Validate if a panel can be cut from available inventory"""
        # Normalize material code before validation
        normalized_material = self.normalize_material_code(material_type)

        # Check if material type exists
        if normalized_material not in self.material_types:
            available_types = ", ".join(sorted(self.material_types))
            return False, f"Material type '{material_type}' not available. Available types: {available_types}"

        # Find compatible sheets using normalized material
        compatible_sheets = self.find_compatible_sheets(normalized_material, thickness, width, height)

        if not compatible_sheets:
            return False, f"No available sheets can accommodate {width}×{height}mm with thickness {thickness}mm in material {normalized_material}"

        return True, f"Can be cut from {len(compatible_sheets)} available sheet(s)"

    def get_inventory_summary(self) -> Dict:
        """Get summary of current inventory"""
        summary = {
            'total_sheets': len(self.inventory),
            'material_types': len(self.material_types),
            'by_material_type': {},
            'by_thickness': {},
            'total_area': 0,
            'total_value': 0
        }

        for sheet in self.inventory:
            # By material type
            if sheet.material_type not in summary['by_material_type']:
                summary['by_material_type'][sheet.material_type] = {
                    'count': 0, 'total_area': 0, 'total_value': 0
                }

            summary['by_material_type'][sheet.material_type]['count'] += 1
            summary['by_material_type'][sheet.material_type]['total_area'] += sheet.area
            summary['by_material_type'][sheet.material_type]['total_value'] += sheet.cost_per_sheet

            # By thickness
            thickness_key = f"{sheet.thickness}mm"
            if thickness_key not in summary['by_thickness']:
                summary['by_thickness'][thickness_key] = 0
            summary['by_thickness'][thickness_key] += 1

            # Totals
            summary['total_area'] += sheet.area
            summary['total_value'] += sheet.cost_per_sheet

        return summary

    def normalize_material_code(self, input_code: str) -> str:
        """Normalize material code using mapping"""
        return self.material_mapping.get(input_code, input_code)

    def add_material_mapping(self, from_code: str, to_code: str):
        """Add a new material mapping"""
        self.material_mapping[from_code] = to_code
        self._save_material_mapping()

    def get_all_material_types(self) -> List[str]:
        """Get all available material types"""
        return sorted(list(self.material_types))

    def export_inventory(self, file_path: str) -> bool:
        """Export inventory to file"""
        try:
            data = {
                'exported_at': datetime.now().isoformat(),
                'total_sheets': len(self.inventory),
                'sheets': [asdict(sheet) for sheet in self.inventory]
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            self.logger.error(f"Failed to export inventory: {e}")
            return False


# Global instance
_material_manager = None

def get_material_manager() -> MaterialInventoryManager:
    """Get global material manager instance"""
    global _material_manager
    if _material_manager is None:
        _material_manager = MaterialInventoryManager()
    return _material_manager