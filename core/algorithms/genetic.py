"""
Genetic Algorithm for Steel Cutting Optimization
鋼板切断最適化用遺伝的アルゴリズム

Implements a genetic algorithm for bin packing with guillotine constraints
ギロチン制約付きビンパッキング用遺伝的アルゴリズム実装
"""

import random
import math
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.models import Panel, SteelSheet, PlacementResult, PlacedPanel, OptimizationConstraints
from core.optimizer import OptimizationAlgorithm
from core.algorithms.ffd import GuillotineBinPacker  # Base placer for chromosome evaluation


@dataclass
class Chromosome:
    """遺伝子（パネル配置順序）"""
    sequence: List[int]  # Panel indices
    fitness: float = 0.0
    efficiency: float = 0.0
    sheets_used: int = 0

    def __len__(self):
        return len(self.sequence)

    def copy(self):
        return Chromosome(
            sequence=self.sequence.copy(),
            fitness=self.fitness,
            efficiency=self.efficiency,
            sheets_used=self.sheets_used
        )


class GeneticAlgorithm(OptimizationAlgorithm):
    """
    遺伝的アルゴリズム実装
    Genetic Algorithm for cutting optimization
    """

    def __init__(self,
                 population_size: int = 20,
                 generations: int = 30,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 3,
                 tournament_size: int = 3):
        super().__init__("GA")

        # GA Parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        # Algorithm state
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        self.fitness_history = []

    def estimate_time(self, panel_count: int, complexity: float) -> float:
        """
        Estimate processing time for genetic algorithm
        遺伝的アルゴリズムの処理時間見積もり
        """
        # GA time depends on population size and generations
        base_time = 0.1  # Base time per generation
        estimated_time = base_time * self.generations * (panel_count / 10.0)
        return min(estimated_time, 60.0)  # Cap at 60 seconds

    def optimize(self,
                 panels: List[Panel],
                 sheet: SteelSheet,
                 constraints: OptimizationConstraints) -> PlacementResult:
        """
        メイン最適化処理
        Main optimization process
        """
        if not panels:
            return PlacementResult(
                sheet_id=1,
                material_block=sheet.material,
                sheet=sheet,
                panels=[],
                efficiency=0.0,
                waste_area=sheet.area,
                cut_length=0.0,
                cost=sheet.cost_per_sheet,
                algorithm="GA",
                processing_time=0.0
            )

        self.logger.info(f"GA開始: {len(panels)}パネル, 世代数={self.generations}, 集団サイズ={self.population_size}")

        start_time = time.time()

        # Initialize population
        self._initialize_population(panels)

        # Evolution process
        for generation in range(self.generations):
            # Evaluate population
            self._evaluate_population(panels, sheet, constraints)

            # Track best solution
            best_in_generation = max(self.population, key=lambda x: x.fitness)
            if self.best_chromosome is None or best_in_generation.fitness > self.best_chromosome.fitness:
                self.best_chromosome = best_in_generation.copy()

            self.fitness_history.append(best_in_generation.fitness)

            # Early stopping if no improvement
            if generation > 20 and len(set(self.fitness_history[-10:])) == 1:
                self.logger.info(f"早期停止: 世代{generation}")
                break

            # Create next generation
            if generation < self.generations - 1:
                self.population = self._create_next_generation()

            if generation % 20 == 0:
                self.logger.info(f"世代{generation}: 最適効率={best_in_generation.efficiency:.1%}")

        # Convert best chromosome to placement result
        result = self._chromosome_to_result(self.best_chromosome, panels, sheet, constraints)

        elapsed = time.time() - start_time
        self.logger.info(f"GA完了: {elapsed:.2f}秒, 最終効率={result.efficiency:.1%}")

        return result

    def _initialize_population(self, panels: List[Panel]):
        """初期集団生成"""
        self.population = []
        panel_indices = list(range(len(panels)))

        for _ in range(self.population_size):
            # Create random permutation
            sequence = panel_indices.copy()
            random.shuffle(sequence)

            # Add some heuristic-based individuals
            if len(self.population) < self.population_size // 4:
                # Size-based sorting (largest first)
                sequence = sorted(panel_indices,
                                key=lambda i: panels[i].cutting_area,
                                reverse=True)
            elif len(self.population) < self.population_size // 2:
                # Priority-based sorting
                sequence = sorted(panel_indices,
                                key=lambda i: (panels[i].priority, panels[i].cutting_area),
                                reverse=True)

            self.population.append(Chromosome(sequence=sequence))

    def _evaluate_population(self, panels: List[Panel], sheet: SteelSheet, constraints: OptimizationConstraints):
        """集団評価（並列処理）"""


        def evaluate_chromosome(chromosome):
            return self._evaluate_chromosome(chromosome, panels, sheet, constraints)

        # 並列評価
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(evaluate_chromosome, chrom): chrom
                      for chrom in self.population}

            for future in as_completed(futures):
                chromosome = futures[future]
                try:
                    efficiency, sheets_used = future.result()
                    chromosome.efficiency = efficiency
                    chromosome.sheets_used = sheets_used
                    chromosome.fitness = self._calculate_fitness(efficiency, sheets_used)
                except Exception as e:
                    self.logger.warning(f"評価エラー: {e}")
                    chromosome.fitness = 0.0

    def _evaluate_chromosome(self, chromosome: Chromosome, panels: List[Panel],
                           sheet: SteelSheet, constraints: OptimizationConstraints) -> Tuple[float, int]:
        """個体評価"""
        try:
            # Create ordered panel list based on chromosome
            ordered_panels = [panels[i] for i in chromosome.sequence]

            # Use GuillotineBinPacker for actual placement
            placer = GuillotineBinPacker(sheet.width, sheet.height)
            sheets_used = 0
            total_panels = sum(panel.quantity for panel in ordered_panels)
            placed_panels = 0

            # Process panels in order
            remaining_panels = []
            for panel in ordered_panels:
                remaining_panels.extend([panel] * panel.quantity)

            while remaining_panels and sheets_used < 10:  # Limit sheets
                placer = GuillotineBinPacker(sheet.width, sheet.height)
                sheets_used += 1

                panels_this_sheet = []
                i = 0
                while i < len(remaining_panels):
                    if placer.place_panel(remaining_panels[i]):
                        panels_this_sheet.append(remaining_panels.pop(i))
                        placed_panels += 1
                    else:
                        i += 1

                if not panels_this_sheet:  # No panels could be placed
                    break

            # Calculate efficiency
            efficiency = placed_panels / total_panels if total_panels > 0 else 0.0

            return efficiency, sheets_used

        except Exception as e:
            self.logger.warning(f"Chromosome evaluation error: {e}")
            return 0.0, 999

    def _calculate_fitness(self, efficiency: float, sheets_used: int) -> float:
        """適応度計算"""
        # Multi-objective fitness: maximize efficiency, minimize sheets
        efficiency_weight = 0.7
        sheet_penalty_weight = 0.3

        efficiency_score = efficiency * efficiency_weight
        sheet_penalty = max(0, 1 - (sheets_used - 1) * 0.1) * sheet_penalty_weight

        return efficiency_score + sheet_penalty

    def _create_next_generation(self) -> List[Chromosome]:
        """次世代生成"""
        next_generation = []

        # Elitism: keep best individuals
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in range(min(self.elite_size, len(sorted_population))):
            next_generation.append(sorted_population[i].copy())

        # Generate offspring
        while len(next_generation) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                offspring1, offspring2 = self._crossover(parent1, parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    self._mutate(offspring1)
                if random.random() < self.mutation_rate:
                    self._mutate(offspring2)

                next_generation.extend([offspring1, offspring2])
            else:
                # Just mutation
                parent = self._tournament_selection()
                offspring = parent.copy()
                self._mutate(offspring)
                next_generation.append(offspring)

        return next_generation[:self.population_size]

    def _tournament_selection(self) -> Chromosome:
        """トーナメント選択"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """順序交叉（OX）"""
        size = len(parent1.sequence)

        # Select random crossover points
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)

        # Create offspring
        offspring1_seq = [-1] * size
        offspring2_seq = [-1] * size

        # Copy segments
        offspring1_seq[start:end] = parent1.sequence[start:end]
        offspring2_seq[start:end] = parent2.sequence[start:end]

        # Fill remaining positions
        self._fill_remaining_ox(offspring1_seq, parent2.sequence, end)
        self._fill_remaining_ox(offspring2_seq, parent1.sequence, end)

        return Chromosome(offspring1_seq), Chromosome(offspring2_seq)

    def _fill_remaining_ox(self, offspring: List[int], parent: List[int], start_pos: int):
        """OX交叉の残り位置を埋める"""
        size = len(offspring)
        parent_pos = start_pos % size
        offspring_pos = start_pos % size

        while -1 in offspring:
            if parent[parent_pos] not in offspring:
                offspring[offspring_pos] = parent[parent_pos]
                offspring_pos = (offspring_pos + 1) % size
            parent_pos = (parent_pos + 1) % size

    def _mutate(self, chromosome: Chromosome):
        """突然変異（2点スワップ）"""
        if len(chromosome.sequence) < 2:
            return

        # Random swap
        i, j = random.sample(range(len(chromosome.sequence)), 2)
        chromosome.sequence[i], chromosome.sequence[j] = chromosome.sequence[j], chromosome.sequence[i]

        # Reset fitness
        chromosome.fitness = 0.0

    def _chromosome_to_result(self, chromosome: Chromosome, panels: List[Panel],
                            sheet: SteelSheet, constraints: OptimizationConstraints) -> PlacementResult:
        """最適個体をPlacementResultに変換"""
        if not chromosome:
            return PlacementResult(
                sheet_id=1,
                material_block=sheet.material,
                sheet=sheet,
                panels=[],
                efficiency=0.0,
                waste_area=sheet.area,
                cut_length=0.0,
                cost=sheet.cost_per_sheet,
                algorithm="GA",
                processing_time=0.0
            )

        # Create ordered panel list
        ordered_panels = [panels[i] for i in chromosome.sequence]

        # Place panels using best sequence
        placer = GuillotineBinPacker(sheet.width, sheet.height)
        all_placed_panels = []
        total_panels = sum(panel.quantity for panel in ordered_panels)

        # Expand panels by quantity
        panel_instances = []
        for panel in ordered_panels:
            panel_instances.extend([panel] * panel.quantity)

        # Place panels
        for panel in panel_instances:
            if placer.place_panel(panel):
                all_placed_panels.extend(placer.placed_panels[-1:])  # Add last placed

        # Calculate metrics
        used_area = sum(p.actual_width * p.actual_height for p in all_placed_panels)
        total_area = sheet.width * sheet.height
        efficiency = used_area / total_area if total_area > 0 else 0.0

        return PlacementResult(
            sheet_id=1,
            material_block=sheet.material,
            sheet=sheet,
            panels=all_placed_panels,
            efficiency=efficiency,
            waste_area=total_area - used_area,
            cut_length=self._calculate_cut_length(all_placed_panels),
            cost=sheet.cost_per_sheet,
            algorithm="GA"
        )

    def _calculate_cut_length(self, placed_panels: List[PlacedPanel]) -> float:
        """切断長計算"""
        if not placed_panels:
            return 0.0

        # Simplified cut length calculation
        total_length = 0.0
        for panel in placed_panels:
            # Approximate: perimeter of each panel
            perimeter = 2 * (panel.actual_width + panel.actual_height)
            total_length += perimeter

        return total_length

    def group_by_material(self, panels: List[Panel]) -> Dict[str, List[Panel]]:
        """材質別グループ化"""
        groups = {}
        for panel in panels:
            material = panel.material
            if material not in groups:
                groups[material] = []
            groups[material].append(panel)
        return groups


def create_genetic_algorithm(population_size: int = 20,
                           generations: int = 30,
                           mutation_rate: float = 0.1) -> GeneticAlgorithm:
    """
    遺伝的アルゴリズムインスタンス作成

    Args:
        population_size: 集団サイズ
        generations: 世代数
        mutation_rate: 突然変異率
    """
    return GeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate
    )
