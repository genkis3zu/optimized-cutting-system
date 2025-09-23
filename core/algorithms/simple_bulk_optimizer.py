"""
Simple Bulk Optimizer - Focus on High Quantity Panels
シンプルバルク最適化 - 高数量パネルに特化
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from core.models import (
    Panel, SteelSheet, PlacementResult, PlacedPanel,
    OptimizationConstraints
)
from core.optimizer import OptimizationAlgorithm


class SimpleBulkOptimizer(OptimizationAlgorithm):
    """
    シンプルバルク最適化アルゴリズム
    - 高数量パネルを優先的にグリッド配置
    - 重複配置エラーを回避
    - 実装が確実で安定
    """

    def __init__(self):
        super().__init__("Simple_Bulk")

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """処理時間推定"""
        return min(panel_count * 0.01, 30.0)

    def optimize(self, panels: List[Panel], sheet: SteelSheet, constraints: OptimizationConstraints) -> PlacementResult:
        """シンプルバルク最適化"""
        if not panels:
            return self._empty_result(sheet)

        start_time = time.time()
        self.logger.info(f"Simple Bulk開始: {len(panels)} パネル種類")

        try:
            # 全パネルを個別に展開
            individual_panels = []
            for panel in panels:
                for i in range(panel.quantity):
                    individual_panel = self._create_individual_panel(panel, i+1)
                    individual_panels.append(individual_panel)

            self.logger.info(f"展開後総パネル数: {len(individual_panels)}")

            # 高数量パネルを特定してバルク処理
            bulk_panels, remaining_panels = self._separate_bulk_panels(individual_panels)

            placed_panels = []
            used_area = 0.0

            # Strategy 1: バルクパネルを優先的にグリッド配置
            bulk_placement = []
            if bulk_panels:
                bulk_placement = self._place_bulk_panels_grid(bulk_panels, sheet)
                self.logger.info(f"バルク配置: {len(bulk_placement)}個")

            # Strategy 2: 残りパネルを従来方式で配置
            remaining_placement = []
            if remaining_panels:
                remaining_placement = self._place_remaining_panels(remaining_panels, sheet)
                self.logger.info(f"残りパネル配置: {len(remaining_placement)}個")

            # 最も多くのパネルを配置できる戦略を選択
            if len(bulk_placement) >= len(remaining_placement):
                placed_panels.extend(bulk_placement)
                used_area += sum(p.actual_width * p.actual_height for p in bulk_placement)
                self.logger.info(f"バルク戦略選択: {len(bulk_placement)}個配置")
            else:
                placed_panels.extend(remaining_placement)
                used_area += sum(p.actual_width * p.actual_height for p in remaining_placement)
                self.logger.info(f"残りパネル戦略選択: {len(remaining_placement)}個配置")

            processing_time = time.time() - start_time
            efficiency = used_area / sheet.area if sheet.area > 0 else 0.0

            total_input_panels = len(individual_panels)
            total_placed = len(placed_panels)

            self.logger.info(f"Simple Bulk完了: {total_placed}/{total_input_panels} パネル配置 ({efficiency:.1%})")

            return PlacementResult(
                sheet_id=1,
                material_block=sheet.material,
                sheet=sheet,
                panels=placed_panels,
                efficiency=efficiency,
                waste_area=sheet.area - used_area,
                cut_length=self._calculate_cut_length(placed_panels),
                cost=sheet.cost_per_sheet,
                algorithm="Simple_Bulk",
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Simple Bulk エラー: {e}")
            processing_time = time.time() - start_time
            return self._empty_result(sheet, processing_time)

    def _separate_bulk_panels(self, individual_panels: List[Panel]) -> Tuple[List[Panel], List[Panel]]:
        """同一仕様のパネルをバルクグループとして分離"""
        # パネルIDの基本部分でグループ化 (例: "562210_LUX_1" -> "562210_LUX")
        groups = defaultdict(list)

        for panel in individual_panels:
            base_id = panel.id.rsplit('_', 1)[0]  # 末尾の番号を除去
            groups[base_id].append(panel)

        bulk_panels = []
        remaining_panels = []

        for base_id, group in groups.items():
            if len(group) >= 4:  # 4個以上でバルク扱い
                # 同一仕様かチェック
                first_panel = group[0]
                if all(self._panels_are_identical(first_panel, p) for p in group[1:]):
                    bulk_panels.extend(group)
                    self.logger.info(f"バルクグループ: {base_id} - {len(group)}個")
                else:
                    remaining_panels.extend(group)
            else:
                remaining_panels.extend(group)

        return bulk_panels, remaining_panels

    def _panels_are_identical(self, panel1: Panel, panel2: Panel) -> bool:
        """2つのパネルが同一仕様かチェック"""
        return (
            getattr(panel1, 'cutting_width', panel1.width) == getattr(panel2, 'cutting_width', panel2.width) and
            getattr(panel1, 'cutting_height', panel1.height) == getattr(panel2, 'cutting_height', panel2.height) and
            panel1.material == panel2.material and
            panel1.allow_rotation == panel2.allow_rotation
        )

    def _place_bulk_panels_grid(self, bulk_panels: List[Panel], sheet: SteelSheet) -> List[PlacedPanel]:
        """バルクパネルをグリッド配置"""
        if not bulk_panels:
            return []

        # 代表パネルの寸法を取得
        representative_panel = bulk_panels[0]
        panel_w = getattr(representative_panel, 'cutting_width', representative_panel.width)
        panel_h = getattr(representative_panel, 'cutting_height', representative_panel.height)

        self.logger.info(f"グリッド配置: {panel_w:.0f}x{panel_h:.0f}mm パネル {len(bulk_panels)}個")

        # グリッドの計算（回転なしでシンプルに）
        cols = int(sheet.width // panel_w)
        rows = int(sheet.height // panel_h)
        grid_capacity = cols * rows

        self.logger.info(f"グリッド容量: {cols}列 x {rows}行 = {grid_capacity}個")

        if grid_capacity == 0:
            self.logger.warning("グリッド配置不可能")
            return []

        # グリッド配置実行
        placed_panels = []
        panels_to_place = min(len(bulk_panels), grid_capacity)

        for i in range(panels_to_place):
            row = i // cols
            col = i % cols

            x_pos = col * panel_w
            y_pos = row * panel_h

            placed_panel = PlacedPanel(
                panel=bulk_panels[i],
                x=x_pos,
                y=y_pos,
                rotated=False
            )
            placed_panels.append(placed_panel)

        efficiency = (panels_to_place * panel_w * panel_h) / sheet.area
        self.logger.info(f"グリッド配置完了: {panels_to_place}個配置 ({efficiency:.1%}効率)")

        return placed_panels

    def _place_remaining_panels(self, remaining_panels: List[Panel], sheet: SteelSheet) -> List[PlacedPanel]:
        """残りパネルを配置"""
        if not remaining_panels:
            return []

        # 面積降順でソート
        sorted_panels = sorted(remaining_panels,
                             key=lambda p: getattr(p, 'cutting_area', p.area),
                             reverse=True)

        placed_panels = []
        current_x = 0.0
        current_y = 0.0
        row_height = 0.0

        # 左詰め配置
        for panel in sorted_panels:  # すべてのパネルを試行
            panel_w = getattr(panel, 'cutting_width', panel.width)
            panel_h = getattr(panel, 'cutting_height', panel.height)

            # 次の行への移動チェック
            if current_x + panel_w > sheet.width:
                current_x = 0.0
                current_y += row_height
                row_height = 0.0

            # シートに収まるかチェック
            if current_y + panel_h <= sheet.height and current_x + panel_w <= sheet.width:
                placed_panel = PlacedPanel(
                    panel=panel,
                    x=current_x,
                    y=current_y,
                    rotated=False
                )
                placed_panels.append(placed_panel)

                current_x += panel_w
                row_height = max(row_height, panel_h)

        return placed_panels

    def _create_individual_panel(self, panel: Panel, index: int) -> Panel:
        """個別パネルインスタンスを作成"""
        individual_panel = Panel(
            id=f"{panel.id}_{index}",
            width=panel.width,
            height=panel.height,
            quantity=1,
            material=panel.material,
            thickness=panel.thickness,
            priority=panel.priority,
            allow_rotation=panel.allow_rotation
        )

        # カッティング寸法をコピー
        if hasattr(panel, '_cutting_width'):
            individual_panel._cutting_width = panel._cutting_width
            individual_panel._cutting_height = panel._cutting_height

        return individual_panel

    def _calculate_cut_length(self, placed_panels: List[PlacedPanel]) -> float:
        """カット長さ計算"""
        total_length = 0.0
        for panel in placed_panels:
            perimeter = 2 * (panel.actual_width + panel.actual_height)
            total_length += perimeter
        return total_length

    def _empty_result(self, sheet: SteelSheet, processing_time: float = 0.0) -> PlacementResult:
        """空の結果を作成"""
        return PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=[],
            efficiency=0.0,
            waste_area=sheet.area,
            cut_length=0.0,
            cost=sheet.cost_per_sheet,
            algorithm="Simple_Bulk",
            processing_time=processing_time
        )