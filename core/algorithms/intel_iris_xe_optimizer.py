"""
Intel Iris Xe Graphics GPU-Accelerated Genetic Algorithm Optimizer

Implements GPU acceleration for steel cutting optimization using Intel Iris Xe Graphics.
Features thermal monitoring, adaptive workload management, and robust CPU fallback.

Key Features:
- Zero-copy unified memory utilization
- Thermal throttling detection and mitigation
- Adaptive population sizing based on GPU capabilities
- Robust error handling with CPU fallback
- Performance monitoring and optimization
"""

import logging
import time
import psutil
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

from core.models import Panel, PlacedPanel
from core.algorithms.genetic import GeneticAlgorithm
from core.algorithms.gpu_fallback_manager import (
    GPUFallbackManager, ExecutionContext, ExecutionMode, FallbackReason
)

logger = logging.getLogger(__name__)

@dataclass
class GPUPerformanceMetrics:
    """Performance metrics for GPU operations"""
    kernel_execution_time: float = 0.0
    memory_transfer_time: float = 0.0
    total_gpu_time: float = 0.0
    cpu_fallback_time: float = 0.0
    thermal_throttling_events: int = 0
    gpu_utilization: float = 0.0
    speedup_factor: float = 1.0

@dataclass
class ThermalState:
    """Thermal monitoring state"""
    gpu_temperature: float = 0.0
    cpu_temperature: float = 0.0
    is_throttling: bool = False
    throttling_start_time: Optional[float] = None
    max_safe_temperature: float = 85.0  # Conservative limit for sustained workloads

class IntelIrisXeOptimizer(GeneticAlgorithm):
    """
    GPU-accelerated genetic algorithm optimizer for Intel Iris Xe Graphics.

    Extends the base genetic algorithm with GPU acceleration while maintaining
    full compatibility and CPU fallback capability.
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        enable_gpu: bool = True,
        thermal_monitoring: bool = True,
        adaptive_workload: bool = True
    ):
        super().__init__(population_size, generations, mutation_rate)

        self.enable_gpu = enable_gpu and OPENCL_AVAILABLE
        self.thermal_monitoring = thermal_monitoring
        self.adaptive_workload = adaptive_workload

        # GPU context and resources
        self.opencl_context: Optional[cl.Context] = None
        self.command_queue: Optional[cl.CommandQueue] = None
        self.device: Optional[cl.Device] = None
        self.kernel_program: Optional[cl.Program] = None

        # Performance monitoring
        self.performance_metrics = GPUPerformanceMetrics()
        self.thermal_state = ThermalState()

        # Fallback manager
        self.fallback_manager = GPUFallbackManager(
            thermal_monitoring=thermal_monitoring,
            performance_tracking=True,
            automatic_fallback=True,
            thermal_limit=85.0,
            memory_limit_mb=4096.0
        )

        # GPU capabilities
        self.max_workgroup_size = 32  # Optimal for Intel Iris Xe
        self.max_memory_mb = 0
        self.gpu_available = False

        # Initialize GPU if available
        if self.enable_gpu:
            self._initialize_gpu()

        # Register executors with fallback manager
        self.fallback_manager.register_executors(
            gpu_executor=self._gpu_optimize,
            cpu_executor=self._cpu_optimize
        )

    def _initialize_gpu(self) -> bool:
        """Initialize OpenCL context and compile kernels for Intel Iris Xe"""
        try:
            # Find Intel GPU device
            platforms = cl.get_platforms()
            intel_device = None

            for platform in platforms:
                if "Intel" in platform.name:
                    devices = platform.get_devices(cl.device_type.GPU)
                    for device in devices:
                        if "Iris" in device.name or "Xe" in device.name:
                            intel_device = device
                            break
                    if intel_device:
                        break

            if not intel_device:
                logger.warning("Intel Iris Xe Graphics not found, GPU acceleration disabled")
                return False

            # Create OpenCL context and command queue
            self.device = intel_device
            self.opencl_context = cl.Context([intel_device])
            self.command_queue = cl.CommandQueue(self.opencl_context)

            # Get device capabilities
            self.max_workgroup_size = min(
                intel_device.max_work_group_size,
                32  # Optimal for Iris Xe
            )
            self.max_memory_mb = intel_device.global_mem_size // (1024 * 1024)

            logger.info(f"Intel Iris Xe GPU initialized:")
            logger.info(f"  Device: {intel_device.name}")
            logger.info(f"  Memory: {self.max_memory_mb} MB")
            logger.info(f"  Max workgroup size: {self.max_workgroup_size}")
            logger.info(f"  Compute units: {intel_device.max_compute_units}")

            # Compile OpenCL kernels
            self._compile_kernels()

            self.gpu_available = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GPU: {e}")
            self.gpu_available = False
            return False

    def _compile_kernels(self):
        """Compile OpenCL kernels from source file"""
        try:
            kernel_file = Path(__file__).parent / "gpu_genetic_kernels.cl"

            if not kernel_file.exists():
                raise FileNotFoundError(f"OpenCL kernel file not found: {kernel_file}")

            with open(kernel_file, 'r') as f:
                kernel_source = f.read()

            # Compile kernels with Intel Iris Xe optimizations
            compiler_options = [
                "-cl-fast-relaxed-math",  # Fast math for performance
                "-cl-mad-enable",         # Multiply-add optimization
                "-DWORKGROUP_SIZE=32",    # Optimal for Iris Xe
            ]

            self.kernel_program = cl.Program(
                self.opencl_context,
                kernel_source
            ).build(options=" ".join(compiler_options))

            logger.info("OpenCL kernels compiled successfully")

        except Exception as e:
            logger.error(f"Failed to compile OpenCL kernels: {e}")
            raise

    def _monitor_thermal_state(self) -> bool:
        """Monitor thermal state and detect throttling"""
        if not self.thermal_monitoring:
            return True

        try:
            # Get CPU temperature (proxy for integrated GPU thermal state)
            temps = psutil.sensors_temperatures()

            cpu_temp = 0.0
            if 'coretemp' in temps:
                cpu_temp = max(sensor.current for sensor in temps['coretemp'])
            elif 'cpu_thermal' in temps:
                cpu_temp = temps['cpu_thermal'][0].current

            self.thermal_state.cpu_temperature = cpu_temp
            self.thermal_state.gpu_temperature = cpu_temp  # Shared thermal envelope

            # Check for throttling
            was_throttling = self.thermal_state.is_throttling
            self.thermal_state.is_throttling = cpu_temp > self.thermal_state.max_safe_temperature

            if self.thermal_state.is_throttling and not was_throttling:
                self.thermal_state.throttling_start_time = time.time()
                self.performance_metrics.thermal_throttling_events += 1
                logger.warning(f"Thermal throttling detected: {cpu_temp:.1f}°C")

            return not self.thermal_state.is_throttling

        except Exception as e:
            logger.warning(f"Thermal monitoring failed: {e}")
            return True  # Assume safe if monitoring fails

    def _estimate_gpu_benefit(self, num_panels: int, population_size: int) -> float:
        """Estimate GPU acceleration benefit for given workload"""
        # GPU overhead (kernel compilation, memory setup)
        gpu_overhead_ms = 200 + (num_panels * 0.1)

        # CPU execution time estimate
        cpu_time_ms = (num_panels * population_size * 0.05) + (population_size * 2)

        # GPU execution time estimate (with speedup factor)
        speedup_factor = min(20, max(2, population_size / 10))  # 2-20x speedup range
        gpu_time_ms = (cpu_time_ms / speedup_factor) + gpu_overhead_ms

        # Return benefit ratio (>1.5 means worthwhile)
        return cpu_time_ms / gpu_time_ms if gpu_time_ms > 0 else 0

    def _should_use_gpu(self, num_panels: int, population_size: int) -> bool:
        """Determine if GPU acceleration should be used"""
        if not self.gpu_available:
            return False

        # Check thermal state
        if not self._monitor_thermal_state():
            logger.info("GPU disabled due to thermal throttling")
            return False

        # Check minimum workload size
        if population_size < 30:
            return False  # Too small for GPU benefit

        # Check memory requirements
        estimated_memory_mb = (num_panels + population_size) * 0.001  # Rough estimate
        if estimated_memory_mb > self.max_memory_mb * 0.8:
            logger.warning(f"Workload too large for GPU memory: {estimated_memory_mb:.1f}MB")
            return False

        # Check benefit estimation
        benefit = self._estimate_gpu_benefit(num_panels, population_size)
        return benefit > 1.5  # Require 50% speedup minimum

    def _prepare_gpu_data(self, panels: List[Panel], population: np.ndarray) -> Dict[str, cl.Buffer]:
        """Prepare data structures for GPU execution"""
        try:
            # Convert panels to GPU-friendly format
            panel_data = np.array([
                [p.width, p.height, p.thickness,
                 hash(p.material) % 256, int(p.allow_rotation), i]
                for i, p in enumerate(panels)
            ], dtype=np.float32)

            # Population genes (panel ordering)
            population_genes = population.astype(np.int32).flatten()

            # Output buffers
            fitness_results = np.zeros(len(population), dtype=np.float32)
            sheet_results = np.zeros(len(population) * 10, dtype=np.float32)  # Max 10 sheets per individual

            # Create OpenCL buffers using unified memory (zero-copy)
            buffers = {
                'panels': cl.Buffer(
                    self.opencl_context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                    hostbuf=panel_data
                ),
                'population': cl.Buffer(
                    self.opencl_context,
                    cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR,
                    hostbuf=population_genes
                ),
                'fitness': cl.Buffer(
                    self.opencl_context,
                    cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                    hostbuf=fitness_results
                ),
                'sheets': cl.Buffer(
                    self.opencl_context,
                    cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR,
                    hostbuf=sheet_results
                )
            }

            return buffers

        except Exception as e:
            logger.error(f"Failed to prepare GPU data: {e}")
            raise

    def _execute_gpu_evaluation(
        self,
        panels: List[Panel],
        population: np.ndarray,
        sheet_width: float = 1500.0,
        sheet_height: float = 3100.0
    ) -> np.ndarray:
        """Execute population evaluation on GPU"""
        start_time = time.time()

        try:
            # Prepare GPU data
            gpu_buffers = self._prepare_gpu_data(panels, population)

            # Set kernel arguments
            evaluation_kernel = self.kernel_program.evaluate_population_fitness
            evaluation_kernel.set_args(
                gpu_buffers['panels'],
                gpu_buffers['population'],
                gpu_buffers['fitness'],
                gpu_buffers['sheets'],
                np.int32(len(population)),
                np.int32(len(panels)),
                np.float32(sheet_width),
                np.float32(sheet_height)
            )

            # Execute kernel
            global_size = (len(population),)
            local_size = (min(self.max_workgroup_size, len(population)),)

            event = cl.enqueue_nd_range_kernel(
                self.command_queue,
                evaluation_kernel,
                global_size,
                local_size
            )
            event.wait()

            # Read results (zero-copy with unified memory)
            fitness_results = np.empty(len(population), dtype=np.float32)
            cl.enqueue_copy(self.command_queue, fitness_results, gpu_buffers['fitness']).wait()

            # Update performance metrics
            self.performance_metrics.kernel_execution_time = time.time() - start_time
            self.performance_metrics.total_gpu_time = self.performance_metrics.kernel_execution_time

            return fitness_results

        except Exception as e:
            logger.error(f"GPU evaluation failed: {e}")
            raise

    def optimize(self, panels: List[Panel], sheet, constraints) -> Dict[str, Any]:
        """
        Main optimization method with GPU acceleration and fallback management.

        Uses the fallback manager for intelligent GPU/CPU execution decisions.
        """
        start_time = time.time()

        # Adaptive population sizing based on GPU capabilities
        if self.adaptive_workload and self.gpu_available:
            optimal_population = min(
                self.population_size,
                max(50, len(panels) * 2)  # Scale population with problem size
            )
            if optimal_population != self.population_size:
                logger.info(f"Adapted population size: {self.population_size} → {optimal_population}")
                self.population_size = optimal_population

        # Create execution context
        execution_mode = ExecutionMode.CPU_ONLY if not self.enable_gpu else ExecutionMode.AUTO
        context = ExecutionContext(
            num_panels=len(panels),
            population_size=self.population_size,
            generations=self.generations,
            available_memory_mb=self.fallback_manager._get_available_memory_mb(),
            current_temperature=self.fallback_manager.current_temperature,
            execution_mode=execution_mode
        )

        # Execute with automatic fallback
        try:
            result = self.fallback_manager.execute_with_fallback(
                context, panels, sheet, constraints
            )

            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics.total_gpu_time = execution_time

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _gpu_optimize(self, panels: List[Panel], sheet, constraints) -> Dict[str, Any]:
        """GPU-accelerated optimization implementation"""
        try:
            start_time = time.time()

            # Initialize population
            population = self._initialize_gpu_population(len(panels))
            best_fitness = 0
            best_individual = None
            best_placement = None

            logger.info(f"Starting GPU genetic algorithm: {self.population_size} individuals, {self.generations} generations")

            for generation in range(self.generations):
                # GPU-accelerated fitness evaluation
                fitness_scores = self._gpu_evaluate_population(panels, population, sheet)

                # Find best individual
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_individual = population[max_fitness_idx].copy()

                    # Generate placement for best individual
                    best_placement = self._generate_placement_from_genes(
                        panels, best_individual, sheet
                    )

                # Selection and reproduction
                population = self._genetic_operations(population, fitness_scores)

                # Log progress every 10 generations
                if generation % 10 == 0:
                    avg_fitness = np.mean(fitness_scores)
                    logger.info(f"Generation {generation}: avg={avg_fitness:.2f}, best={best_fitness:.2f}")

            execution_time = time.time() - start_time
            self.performance_metrics.total_gpu_time = execution_time

            # Create result
            result = {
                'placement_result': best_placement,
                'best_fitness': float(best_fitness),
                'execution_time': execution_time,
                'generations_completed': self.generations,
                'gpu_acceleration': True,
                'performance_metrics': {
                    'kernel_time': self.performance_metrics.kernel_execution_time,
                    'total_time': execution_time,
                    'speedup_estimate': self._estimate_speedup(len(panels), self.population_size)
                }
            }

            logger.info(f"GPU optimization completed in {execution_time:.2f}s, best fitness: {best_fitness:.2f}")
            return result

        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
            # Fallback to CPU automatically handled by fallback manager
            raise

    def _cpu_optimize(self, panels: List[Panel], sheet, constraints) -> Dict[str, Any]:
        """CPU optimization implementation (fallback)"""
        start_time = time.time()
        result = super().optimize(panels, sheet, constraints)
        execution_time = time.time() - start_time

        # Wrap result in expected format if it's not already
        if isinstance(result, dict):
            result['gpu_acceleration'] = False
            result['execution_time'] = execution_time
        else:
            # Convert PlacementResult to expected dict format
            result = {
                'placement_result': result,
                'best_fitness': result.efficiency * 100,  # Convert to percentage
                'execution_time': execution_time,
                'gpu_acceleration': False,
                'performance_metrics': {
                    'kernel_time': 0.0,
                    'total_time': execution_time,
                    'speedup_estimate': 1.0
                }
            }

        return result

    def _estimate_cpu_time(self, num_panels: int, population_size: int) -> float:
        """Estimate CPU execution time for comparison"""
        # Rough estimation based on algorithm complexity
        return (num_panels * population_size * 0.0001) + (self.generations * 0.1)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return {
            'gpu_available': self.gpu_available,
            'gpu_device': self.device.name if self.device else None,
            'performance_metrics': {
                'kernel_time': self.performance_metrics.kernel_execution_time,
                'total_gpu_time': self.performance_metrics.total_gpu_time,
                'speedup_factor': self.performance_metrics.speedup_factor,
                'thermal_events': self.performance_metrics.thermal_throttling_events
            },
            'thermal_state': {
                'current_temp': self.thermal_state.cpu_temperature,
                'is_throttling': self.thermal_state.is_throttling,
                'max_safe_temp': self.thermal_state.max_safe_temperature
            }
        }

    def _initialize_gpu_population(self, num_panels: int) -> np.ndarray:
        """Initialize genetic algorithm population with random panel orderings for GPU"""
        population = np.zeros((self.population_size, num_panels), dtype=np.int32)

        for i in range(self.population_size):
            population[i] = np.random.permutation(num_panels)

        return population

    def _gpu_evaluate_population(self, panels: List[Panel], population: np.ndarray, sheet) -> np.ndarray:
        """Evaluate population fitness using GPU-accelerated kernels"""
        try:
            # Use simple fitness evaluation for now (more reliable)
            panel_areas = np.array([p.width * p.height for p in panels], dtype=np.float32)
            sheet_area = float(sheet.width * sheet.height)

            # Prepare GPU data
            population_genes = population.astype(np.int32).flatten()
            fitness_results = np.zeros(self.population_size, dtype=np.float32)

            # Create OpenCL buffers
            panel_areas_buffer = cl.Buffer(
                self.opencl_context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=panel_areas
            )

            population_buffer = cl.Buffer(
                self.opencl_context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=population_genes
            )

            fitness_buffer = cl.Buffer(
                self.opencl_context,
                cl.mem_flags.WRITE_ONLY,
                fitness_results.nbytes
            )

            # Execute kernel
            kernel = self.kernel_program.simple_fitness_evaluation
            kernel.set_args(
                panel_areas_buffer,
                population_buffer,
                fitness_buffer,
                np.int32(self.population_size),
                np.int32(len(panels)),
                np.float32(sheet_area)
            )

            global_size = (self.population_size,)
            # Ensure local size divides global size evenly
            if self.population_size <= self.max_workgroup_size:
                local_size = (self.population_size,)
            else:
                # Find largest divisor of population_size that's <= max_workgroup_size
                local_size = None
                for size in range(min(self.max_workgroup_size, 32), 0, -1):
                    if self.population_size % size == 0:
                        local_size = (size,)
                        break
                if local_size is None:
                    local_size = None  # Let OpenCL choose

            start_kernel_time = time.time()
            event = cl.enqueue_nd_range_kernel(
                self.command_queue,
                kernel,
                global_size,
                local_size
            )
            event.wait()

            # Read results
            cl.enqueue_copy(self.command_queue, fitness_results, fitness_buffer).wait()

            self.performance_metrics.kernel_execution_time = time.time() - start_kernel_time

            return fitness_results

        except Exception as e:
            logger.error(f"GPU evaluation failed: {e}")
            # Fallback to CPU evaluation
            return self._cpu_evaluate_population(panels, population, sheet)

    def _cpu_evaluate_population(self, panels: List[Panel], population: np.ndarray, sheet) -> np.ndarray:
        """CPU fallback for population evaluation"""
        fitness_scores = np.zeros(self.population_size)

        for i, individual in enumerate(population):
            # Simple fitness: sum of panel areas in order (normalized)
            total_area = sum(panels[gene_idx].width * panels[gene_idx].height
                           for gene_idx in individual if gene_idx < len(panels))
            sheet_area = sheet.width * sheet.height
            fitness_scores[i] = min((total_area / sheet_area) * 100, 100.0)

        return fitness_scores

    def _genetic_operations(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """Perform selection, crossover, and mutation operations"""
        new_population = np.zeros_like(population)

        # Tournament selection
        for i in range(self.population_size):
            # Select two random individuals
            idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)

            # Choose the fitter one
            if fitness_scores[idx1] > fitness_scores[idx2]:
                parent = population[idx1].copy()
            else:
                parent = population[idx2].copy()

            # Mutation: swap two random positions
            if np.random.random() < self.mutation_rate:
                pos1, pos2 = np.random.choice(len(parent), 2, replace=False)
                parent[pos1], parent[pos2] = parent[pos2], parent[pos1]

            new_population[i] = parent

        return new_population

    def _generate_placement_from_genes(self, panels: List[Panel], genes: np.ndarray, sheet) -> 'PlacementResult':
        """Generate actual placement result from genetic encoding"""
        try:
            # Import here to avoid circular imports
            from core.models import PlacementResult, PlacedPanel

            placed_panels = []
            current_x, current_y = 0, 0

            for gene_idx in genes:
                if gene_idx >= len(panels):
                    continue

                panel = panels[gene_idx]

                # Simple placement strategy for demonstration
                if current_x + panel.width <= sheet.width:
                    placed_panel = PlacedPanel(
                        panel=panel,
                        x=current_x,
                        y=current_y,
                        rotated=False
                    )
                    placed_panels.append(placed_panel)
                    current_x += panel.width
                else:
                    # Move to next row
                    current_x = 0
                    if placed_panels:
                        current_y += placed_panels[-1].panel.height
                    else:
                        current_y += panel.height

                    if current_y + panel.height <= sheet.height:
                        placed_panel = PlacedPanel(
                            panel=panel,
                            x=current_x,
                            y=current_y,
                            rotated=False
                        )
                        placed_panels.append(placed_panel)
                        current_x += panel.width

            # Calculate efficiency
            total_panel_area = sum(p.panel.width * p.panel.height for p in placed_panels)
            sheet_area = sheet.width * sheet.height
            efficiency = (total_panel_area / sheet_area) * 100

            return PlacementResult(
                sheet_id=1,
                material_block="Steel",
                sheet=sheet,
                panels=placed_panels,
                efficiency=efficiency / 100.0,  # Convert to 0-1 range
                waste_area=sheet.width * sheet.height - total_panel_area,
                cut_length=0.0,  # TODO: Calculate actual cut length
                cost=sheet.cost_per_sheet
            )

        except Exception as e:
            logger.error(f"Failed to generate placement: {e}")
            raise

    def _estimate_speedup(self, num_panels: int, population_size: int) -> float:
        """Estimate GPU speedup factor based on workload"""
        # Simple heuristic for speedup estimation
        base_speedup = min(20, max(2, population_size / 10))
        complexity_factor = min(2, num_panels / 50)
        return base_speedup * complexity_factor

    def cleanup(self):
        """Clean up GPU resources"""
        if self.opencl_context:
            try:
                self.command_queue.finish()
                self.command_queue = None
                self.opencl_context = None
                self.kernel_program = None
                logger.info("GPU resources cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up GPU resources: {e}")

        # Cleanup fallback manager
        self.fallback_manager.cleanup()


def create_intel_iris_xe_optimizer(**kwargs) -> IntelIrisXeOptimizer:
    """Factory function to create Intel Iris Xe optimizer with validation"""
    if not OPENCL_AVAILABLE:
        logger.warning("PyOpenCL not available, GPU acceleration disabled")
        kwargs['enable_gpu'] = False

    return IntelIrisXeOptimizer(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test GPU initialization
    optimizer = create_intel_iris_xe_optimizer(
        population_size=100,
        generations=50,
        enable_gpu=True,
        thermal_monitoring=True
    )

    # Print GPU capabilities
    stats = optimizer.get_performance_stats()
    print("GPU Capabilities:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    optimizer.cleanup()