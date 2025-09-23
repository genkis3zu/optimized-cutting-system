"""
GPU Fallback Manager

Robust error handling and CPU fallback system for GPU-accelerated genetic algorithms.
Provides seamless transition between GPU and CPU execution based on various conditions.

Features:
- GPU error detection and recovery
- Thermal throttling monitoring
- Memory pressure management
- Performance degradation detection
- Automatic fallback decision making
- Execution strategy selection
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import traceback

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution mode options"""
    GPU_ONLY = "gpu_only"
    CPU_ONLY = "cpu_only"
    HYBRID = "hybrid"
    AUTO = "auto"

class FallbackReason(Enum):
    """Reasons for GPU to CPU fallback"""
    GPU_ERROR = "gpu_error"
    THERMAL_THROTTLING = "thermal_throttling"
    MEMORY_PRESSURE = "memory_pressure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIVER_ISSUE = "driver_issue"
    USER_PREFERENCE = "user_preference"
    WORKLOAD_TOO_SMALL = "workload_too_small"

@dataclass
class ExecutionContext:
    """Context for execution decision making"""
    num_panels: int
    population_size: int
    generations: int
    available_memory_mb: float
    current_temperature: float
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    thermal_limit: float = 85.0
    memory_limit_mb: float = 4096.0

@dataclass
class FallbackEvent:
    """Record of fallback event"""
    timestamp: float
    reason: FallbackReason
    context: ExecutionContext
    error_message: Optional[str] = None
    recovery_successful: bool = False

@dataclass
class PerformanceMetrics:
    """Performance tracking for fallback decisions"""
    gpu_execution_times: List[float] = field(default_factory=list)
    cpu_execution_times: List[float] = field(default_factory=list)
    gpu_error_count: int = 0
    cpu_error_count: int = 0
    thermal_events: int = 0
    fallback_events: List[FallbackEvent] = field(default_factory=list)

class GPUFallbackManager:
    """
    Manages GPU execution with robust fallback to CPU.

    Provides intelligent decision making for execution strategy based on
    system conditions, performance history, and error patterns.
    """

    def __init__(
        self,
        thermal_monitoring: bool = True,
        performance_tracking: bool = True,
        automatic_fallback: bool = True,
        max_gpu_errors: int = 3,
        thermal_limit: float = 85.0,
        memory_limit_mb: float = 4096.0
    ):
        self.thermal_monitoring = thermal_monitoring
        self.performance_tracking = performance_tracking
        self.automatic_fallback = automatic_fallback
        self.max_gpu_errors = max_gpu_errors
        self.thermal_limit = thermal_limit
        self.memory_limit_mb = memory_limit_mb

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.current_execution_mode = ExecutionMode.AUTO

        # Error tracking
        self.consecutive_gpu_errors = 0
        self.last_gpu_error_time = 0.0
        self.gpu_disabled_until = 0.0

        # Thermal monitoring
        self.thermal_monitor_active = False
        self.thermal_monitor_thread = None
        self.current_temperature = 45.0

        # Callbacks
        self.gpu_executor: Optional[Callable] = None
        self.cpu_executor: Optional[Callable] = None

        if self.thermal_monitoring:
            self._start_thermal_monitoring()

    def register_executors(
        self,
        gpu_executor: Callable,
        cpu_executor: Callable
    ):
        """Register GPU and CPU execution functions"""
        self.gpu_executor = gpu_executor
        self.cpu_executor = cpu_executor

    def _start_thermal_monitoring(self):
        """Start background thermal monitoring"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - thermal monitoring disabled")
            return

        self.thermal_monitor_active = True
        self.thermal_monitor_thread = threading.Thread(
            target=self._thermal_monitor_loop,
            daemon=True
        )
        self.thermal_monitor_thread.start()
        logger.info("Thermal monitoring started")

    def _thermal_monitor_loop(self):
        """Background thermal monitoring loop"""
        while self.thermal_monitor_active:
            try:
                temp = self._get_cpu_temperature()
                if temp > 0:
                    self.current_temperature = temp

                    # Check for thermal throttling
                    if temp > self.thermal_limit:
                        self.metrics.thermal_events += 1
                        logger.warning(f"🌡️ Thermal limit exceeded: {temp:.1f}°C")

                time.sleep(2.0)  # Monitor every 2 seconds

            except Exception as e:
                logger.debug(f"Thermal monitoring error: {e}")
                time.sleep(5.0)  # Back off on errors

    def _get_cpu_temperature(self) -> float:
        """Get current CPU temperature"""
        if not PSUTIL_AVAILABLE:
            return 45.0

        try:
            temps = psutil.sensors_temperatures()

            if 'coretemp' in temps:
                return max(sensor.current for sensor in temps['coretemp'])
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
            else:
                return 45.0

        except Exception:
            return 45.0

    def _get_available_memory_mb(self) -> float:
        """Get available memory in MB"""
        if not PSUTIL_AVAILABLE:
            return 4096.0

        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)
        except Exception:
            return 4096.0

    def should_use_gpu(self, context: ExecutionContext) -> bool:
        """
        Determine if GPU should be used based on current conditions.

        Args:
            context: Execution context with problem parameters

        Returns:
            True if GPU should be used, False for CPU fallback
        """
        # Check if GPU is temporarily disabled
        if time.time() < self.gpu_disabled_until:
            logger.info(f"GPU disabled until {self.gpu_disabled_until - time.time():.1f}s")
            return False

        # Check execution mode preference
        if context.execution_mode == ExecutionMode.CPU_ONLY:
            return False
        elif context.execution_mode == ExecutionMode.GPU_ONLY:
            return True

        # Automatic decision making
        return self._automatic_gpu_decision(context)

    def _automatic_gpu_decision(self, context: ExecutionContext) -> bool:
        """Make automatic GPU vs CPU decision"""
        decision_factors = []

        # Factor 1: Workload size
        if context.population_size < 30:
            decision_factors.append(("workload_too_small", False))
        elif context.population_size >= 100:
            decision_factors.append(("workload_size", True))
        else:
            decision_factors.append(("workload_size", True))

        # Factor 2: Thermal condition
        if self.current_temperature > self.thermal_limit:
            decision_factors.append(("thermal_limit", False))
        elif self.current_temperature > self.thermal_limit - 5:
            decision_factors.append(("thermal_warning", False))
        else:
            decision_factors.append(("thermal_ok", True))

        # Factor 3: Memory availability
        available_memory = self._get_available_memory_mb()
        estimated_usage = self._estimate_memory_usage(context)

        if estimated_usage > available_memory * 0.9:
            decision_factors.append(("memory_pressure", False))
        elif estimated_usage > self.memory_limit_mb:
            decision_factors.append(("memory_limit", False))
        else:
            decision_factors.append(("memory_ok", True))

        # Factor 4: Recent GPU errors
        if self.consecutive_gpu_errors >= self.max_gpu_errors:
            decision_factors.append(("gpu_errors", False))
        elif self.consecutive_gpu_errors > 0:
            decision_factors.append(("gpu_warnings", False))
        else:
            decision_factors.append(("gpu_stable", True))

        # Factor 5: Performance history
        if self.performance_tracking and len(self.metrics.gpu_execution_times) > 3:
            gpu_avg = sum(self.metrics.gpu_execution_times[-3:]) / 3
            cpu_avg = sum(self.metrics.cpu_execution_times[-3:]) / 3 if self.metrics.cpu_execution_times else gpu_avg * 5

            if gpu_avg < cpu_avg * 1.5:  # GPU should be at least 50% faster
                decision_factors.append(("performance_benefit", True))
            else:
                decision_factors.append(("performance_poor", False))

        # Make decision based on factors
        positive_factors = sum(1 for _, decision in decision_factors if decision)
        total_factors = len(decision_factors)

        use_gpu = positive_factors > total_factors / 2

        logger.debug(f"GPU decision factors: {decision_factors}")
        logger.debug(f"GPU decision: {use_gpu} ({positive_factors}/{total_factors})")

        return use_gpu

    def _estimate_memory_usage(self, context: ExecutionContext) -> float:
        """Estimate memory usage for given context (in MB)"""
        # Rough estimation based on problem parameters
        panel_memory = context.num_panels * 0.001  # 1KB per panel
        population_memory = context.population_size * context.num_panels * 0.004  # 4 bytes per gene
        gpu_overhead = 100  # 100MB GPU overhead

        return panel_memory + population_memory + gpu_overhead

    @contextmanager
    def execution_context(self, context: ExecutionContext):
        """Context manager for tracked execution"""
        start_time = time.time()
        execution_mode = "gpu" if self.should_use_gpu(context) else "cpu"

        logger.info(f"🚀 Starting {execution_mode.upper()} execution")
        logger.info(f"   Problem: {context.num_panels} panels, {context.population_size} population")
        logger.info(f"   Temperature: {self.current_temperature:.1f}°C")
        logger.info(f"   Memory: {self._get_available_memory_mb():.0f}MB available")

        try:
            yield execution_mode

            # Record successful execution
            execution_time = time.time() - start_time

            if execution_mode == "gpu":
                self.metrics.gpu_execution_times.append(execution_time)
                self.consecutive_gpu_errors = 0  # Reset error count on success
            else:
                self.metrics.cpu_execution_times.append(execution_time)

            logger.info(f"✅ {execution_mode.upper()} execution completed in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time

            if execution_mode == "gpu":
                self._handle_gpu_error(e, context)
            else:
                self.metrics.cpu_error_count += 1
                logger.error(f"CPU execution failed: {e}")

            logger.error(f"❌ {execution_mode.upper()} execution failed after {execution_time:.2f}s")
            raise

    def _handle_gpu_error(self, error: Exception, context: ExecutionContext):
        """Handle GPU execution error with fallback logic"""
        self.consecutive_gpu_errors += 1
        self.metrics.gpu_error_count += 1
        self.last_gpu_error_time = time.time()

        # Determine fallback reason
        error_str = str(error).lower()

        if "memory" in error_str or "out of memory" in error_str:
            reason = FallbackReason.MEMORY_PRESSURE
        elif "thermal" in error_str or "throttling" in error_str:
            reason = FallbackReason.THERMAL_THROTTLING
        elif "driver" in error_str or "platform" in error_str:
            reason = FallbackReason.DRIVER_ISSUE
        else:
            reason = FallbackReason.GPU_ERROR

        # Record fallback event
        fallback_event = FallbackEvent(
            timestamp=time.time(),
            reason=reason,
            context=context,
            error_message=str(error)
        )
        self.metrics.fallback_events.append(fallback_event)

        # Decide on GPU disable duration
        if self.consecutive_gpu_errors >= self.max_gpu_errors:
            disable_duration = 300.0  # 5 minutes
            self.gpu_disabled_until = time.time() + disable_duration
            logger.warning(f"🚫 GPU disabled for {disable_duration}s due to repeated errors")
        elif reason == FallbackReason.THERMAL_THROTTLING:
            disable_duration = 60.0  # 1 minute for thermal issues
            self.gpu_disabled_until = time.time() + disable_duration
            logger.warning(f"🌡️ GPU disabled for {disable_duration}s due to thermal throttling")

        logger.error(f"GPU error #{self.consecutive_gpu_errors}: {error}")

    def execute_with_fallback(
        self,
        context: ExecutionContext,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute with automatic GPU/CPU fallback.

        Args:
            context: Execution context
            *args, **kwargs: Arguments for execution functions

        Returns:
            Execution result
        """
        if not self.gpu_executor or not self.cpu_executor:
            raise ValueError("GPU and CPU executors must be registered")

        with self.execution_context(context) as execution_mode:
            if execution_mode == "gpu":
                try:
                    return self.gpu_executor(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU execution failed, falling back to CPU: {e}")

                    # Immediate fallback to CPU
                    fallback_event = FallbackEvent(
                        timestamp=time.time(),
                        reason=FallbackReason.GPU_ERROR,
                        context=context,
                        error_message=str(e),
                        recovery_successful=True
                    )
                    self.metrics.fallback_events.append(fallback_event)

                    return self.cpu_executor(*args, **kwargs)
            else:
                return self.cpu_executor(*args, **kwargs)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and fallback history"""
        gpu_times = self.metrics.gpu_execution_times
        cpu_times = self.metrics.cpu_execution_times

        return {
            'execution_counts': {
                'gpu_executions': len(gpu_times),
                'cpu_executions': len(cpu_times),
                'total_executions': len(gpu_times) + len(cpu_times)
            },
            'performance': {
                'gpu_avg_time': sum(gpu_times) / len(gpu_times) if gpu_times else 0,
                'cpu_avg_time': sum(cpu_times) / len(cpu_times) if cpu_times else 0,
                'gpu_speedup': (sum(cpu_times) / len(cpu_times)) / (sum(gpu_times) / len(gpu_times))
                             if gpu_times and cpu_times else 1.0
            },
            'errors': {
                'gpu_errors': self.metrics.gpu_error_count,
                'cpu_errors': self.metrics.cpu_error_count,
                'consecutive_gpu_errors': self.consecutive_gpu_errors,
                'thermal_events': self.metrics.thermal_events
            },
            'fallback_events': len(self.metrics.fallback_events),
            'current_state': {
                'temperature': self.current_temperature,
                'available_memory_mb': self._get_available_memory_mb(),
                'gpu_disabled_until': self.gpu_disabled_until,
                'execution_mode': self.current_execution_mode.value
            }
        }

    def cleanup(self):
        """Clean up resources"""
        self.thermal_monitor_active = False
        if self.thermal_monitor_thread and self.thermal_monitor_thread.is_alive():
            self.thermal_monitor_thread.join(timeout=1.0)
        logger.info("Fallback manager cleaned up")


# Example usage and testing
if __name__ == "__main__":
    import random
    import time

    logging.basicConfig(level=logging.INFO)

    def mock_gpu_executor(*args, **kwargs):
        """Mock GPU executor that sometimes fails"""
        time.sleep(0.1)  # Simulate GPU work
        if random.random() < 0.2:  # 20% failure rate
            raise RuntimeError("Mock GPU error")
        return "GPU result"

    def mock_cpu_executor(*args, **kwargs):
        """Mock CPU executor that's slower but reliable"""
        time.sleep(0.5)  # Simulate slower CPU work
        return "CPU result"

    # Test fallback manager
    manager = GPUFallbackManager(
        thermal_limit=80.0,
        max_gpu_errors=2
    )

    manager.register_executors(mock_gpu_executor, mock_cpu_executor)

    # Test multiple executions
    for i in range(10):
        context = ExecutionContext(
            num_panels=100,
            population_size=50,
            generations=10,
            available_memory_mb=4096.0,
            current_temperature=70.0
        )

        try:
            result = manager.execute_with_fallback(context)
            print(f"Execution {i+1}: {result}")
        except Exception as e:
            print(f"Execution {i+1} failed: {e}")

    # Print statistics
    stats = manager.get_execution_stats()
    print("\nExecution Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    manager.cleanup()