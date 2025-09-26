"""
Result Formatter for Optimization Results
最適化結果のフォーマット処理
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core.models import Panel, PlacementResult


@dataclass
class MaterialAssignment:
    """Material assignment for a panel row"""
    row_number: int
    panel: Panel
    sheet_material: str
    sheet_size: str
    material_code: str
    sheet_quantity: int
    combined_rows: List[int]  # Row numbers that share this sheet
    sheet_area: float
    used_area: float
    efficiency: float


class ResultFormatter:
    """Format optimization results in the required output format"""

    def __init__(self):
        self.material_codes = {
            'KW-90': {
                '0.5X968X612': '064766',
                '0.5X1268X2062': '064741',
                '0.5X968X2062': '064711',
                '0.5X1068X2456': '064724',
                '0.5X868X2062': '064701',
                '0.5X1268X2077': '064742',
                '0.5X968X2077': '064712',
                '0.5X1268X2556': '064747',
                '0.5X968X2856': '064709',
            },
            'KWN-20': {
                '0.5X968X2062': '021411',
                '0.5X968X2456': '021414',
            },
            'SE/E8': {
                '0.5X968X2062': '021411',
                '0.5X968X2456': '021414',
            }
        }

    def format_results(
        self,
        original_df: pd.DataFrame,
        optimization_results: List[PlacementResult]
    ) -> pd.DataFrame:
        """
        Format optimization results to match the required output format

        Args:
            original_df: Original input dataframe with panel data
            optimization_results: List of placement results from optimization

        Returns:
            DataFrame with formatted results matching result.txt format
        """
        # Create a copy of the original dataframe
        result_df = original_df.copy()

        # Map column names to match result.txt format exactly
        column_mapping = {
            '製造番号': '製番',
            'PI': 'ＰＩコード',
            '部材名': '品名',
            'W': 'Ｗ寸法',
            'H': 'Ｈ寸法',
            'NCNO': 'NCNO',
            '色': '色',
            '板厚': '板厚',
            '展開Ｈ': '展開Ｈ',
            '展開Ｗ': '展開Ｗ'
        }

        # Rename existing columns to match result.txt format
        for old_name, new_name in column_mapping.items():
            if old_name in result_df.columns:
                result_df.rename(columns={old_name: new_name}, inplace=True)

        # Add new columns for results (matching result.txt format exactly)
        result_df['鋼板サイズ'] = ''
        result_df['資材コード'] = ''
        result_df['数量'] = 0
        result_df['ｺﾒﾝﾄ'] = ''
        result_df['面積'] = 0
        result_df['製品総面積'] = 0
        result_df['素材総面積'] = 0
        result_df['歩留まり率'] = ''
        result_df['差'] = 0

        # Create mapping of panel IDs to placement results
        placement_map = {}
        for result in optimization_results:
            for panel in result.panels:
                # Handle both Panel and PlacedPanel objects
                if hasattr(panel, 'panel') and hasattr(panel.panel, 'id'):
                    # PlacedPanel object
                    panel_id = panel.panel.id
                elif hasattr(panel, 'id'):
                    # Panel object
                    panel_id = panel.id
                else:
                    continue

                if panel_id not in placement_map:
                    placement_map[panel_id] = []
                placement_map[panel_id].append({
                    'sheet': result.sheet,
                    'result': result,
                    'material': result.sheet.material if hasattr(result.sheet, 'material') else 'Unknown'
                })

        # Group panels by sheet to find combinations
        sheet_groups = {}
        for idx, row in result_df.iterrows():
            panel_id = f"P{idx+1}"  # Assuming panel IDs are P1, P2, etc.

            if panel_id in placement_map:
                placements = placement_map[panel_id]
                if placements:
                    placement = placements[0]  # Use first placement
                    sheet = placement['sheet']
                    sheet_key = f"{sheet.material}_{sheet.width}x{sheet.height}"

                    if sheet_key not in sheet_groups:
                        sheet_groups[sheet_key] = []
                    sheet_groups[sheet_key].append(idx + 1)  # Store 1-based row number

        # Process each row
        for idx, row in result_df.iterrows():
            row_num = idx + 1
            panel_id = f"P{row_num}"

            if panel_id in placement_map:
                placements = placement_map[panel_id]
                if placements:
                    placement = placements[0]
                    sheet = placement['sheet']
                    material = placement['material']

                    # Format sheet size
                    thickness = row.get('板厚', 0.5)
                    sheet_size = f"{material} {thickness}X{sheet.width}X{sheet.height}"
                    result_df.at[idx, '鋼板サイズ'] = sheet_size

                    # Look up material code
                    material_key = material.replace('-', '').upper()
                    size_key = f"{thickness}X{sheet.width}X{sheet.height}"

                    if material_key in self.material_codes:
                        if size_key in self.material_codes[material_key]:
                            result_df.at[idx, '資材コード'] = self.material_codes[material_key][size_key]

                    # Set original panel quantity from input data
                    original_quantity = row.get('数量', row.get('quantity', 1))
                    if original_quantity <= 0:
                        original_quantity = 1  # fallback
                    result_df.at[idx, '数量'] = original_quantity

                    # Set sheet quantity (always 1 for individual sheet assignment)
                    result_df.at[idx, 'シート数量'] = 1

                    # Find combined rows (panels on same sheet)
                    sheet_key = f"{material}_{sheet.width}x{sheet.height}"
                    if sheet_key in sheet_groups:
                        combined = sheet_groups[sheet_key]
                        if len(combined) > 1:
                            # Format comment with combined row numbers
                            other_rows = [str(r) for r in combined if r != row_num]
                            if other_rows:
                                result_df.at[idx, 'ｺﾒﾝﾄ'] = ' '.join(other_rows)

                    # Calculate areas
                    sheet_area = sheet.width * sheet.height
                    panel_area = row.get('展開Ｗ', 0) * row.get('展開Ｈ', 0)

                    result_df.at[idx, '面積'] = sheet_area
                    result_df.at[idx, '製品総面積'] = panel_area * row.get('数量', 1)
                    result_df.at[idx, '素材総面積'] = sheet_area

                    # Calculate efficiency
                    if sheet_area > 0:
                        efficiency = (result_df.at[idx, '製品総面積'] / sheet_area) * 100
                        result_df.at[idx, '歩留まり率'] = f"{efficiency:.0f}%"
                        result_df.at[idx, '差'] = sheet_area - result_df.at[idx, '製品総面積']
            else:
                # Panel not placed - mark as remaining material
                result_df.at[idx, 'ｺﾒﾝﾄ'] = '残材/多数子'
                panel_area = row.get('展開Ｗ', 0) * row.get('展開Ｈ', 0)
                result_df.at[idx, '製品総面積'] = panel_area * row.get('数量', 1)

        # Fix duplicate column names by creating second quantity column
        if '数量' in result_df.columns:
            # Create a copy for the second quantity column position (for sheet quantity)
            result_df['シート数量'] = result_df['数量']

        # Reorder columns to match result.txt format exactly
        result_txt_columns = [
            '製番', 'ＰＩコード', '品名', 'Ｗ寸法', 'Ｈ寸法', 'NCNO', '数量', '色', '板厚',
            '展開Ｈ', '展開Ｗ', '鋼板サイズ', '資材コード', 'シート数量', 'ｺﾒﾝﾄ', '面積',
            '製品総面積', '素材総面積', '歩留まり率', '差'
        ]

        # Only include columns that exist in the DataFrame
        available_columns = [col for col in result_txt_columns if col in result_df.columns]
        result_df = result_df[available_columns]

        return result_df

    def save_to_file(
        self,
        result_df: pd.DataFrame,
        output_path: str = "output/optimization_result.txt"
    ):
        """Save formatted results to a tab-delimited text file"""
        result_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        return output_path
