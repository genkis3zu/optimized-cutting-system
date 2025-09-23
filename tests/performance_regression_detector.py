"""
Performance Regression Detection System for Steel Cutting Optimization
性能回帰検出システム

Monitors and detects performance regressions in optimization algorithms:
- Execution time tracking and analysis
- Memory usage monitoring
- Placement rate regression detection
- Algorithm efficiency benchmarking
- Historical performance comparison
"""

import time
import psutil
import os
import json
import statistics
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import threading
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.models import Panel, PlacementResult, OptimizationConstraints
from tests.placement_guarantee_framework import PlacementTestCase, PlacementTestResult


class PerformanceMetrics(NamedTuple):
    """Performance metrics for a single test execution"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    placement_rate: float
    efficiency: float
    panels_per_second: float
    sheets_used: int


@dataclass
class PerformanceBenchmark:
    """Performance benchmark for a specific test case"""
    test_name: str
    algorithm: str
    panel_count: int
    baseline_metrics: PerformanceMetrics
    threshold_metrics: PerformanceMetrics  # Alert thresholds
    timestamp: datetime


@dataclass
class RegressionAlert:
    """Performance regression alert"""
    test_name: str
    algorithm: str
    metric_name: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    baseline_value: float
    current_value: float
    regression_percent: float
    threshold_exceeded: bool
    timestamp: datetime
    description: str


class SystemResourceMonitor:
    """Monitors system resources during test execution"""

    def __init__(self, sampling_interval: float = 0.1):
        """Initialize monitor with sampling interval in seconds"""
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        if not self.samples:
            return {
                'avg_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'avg_cpu_percent': 0.0,
                'peak_cpu_percent': 0.0
            }

        memory_samples = [s['memory_mb'] for s in self.samples]
        cpu_samples = [s['cpu_percent'] for s in self.samples]

        return {
            'avg_memory_mb': statistics.mean(memory_samples),
            'peak_memory_mb': max(memory_samples),
            'avg_cpu_percent': statistics.mean(cpu_samples),
            'peak_cpu_percent': max(cpu_samples)
        }

    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()

        while self.monitoring:
            try:
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()

                self.samples.append({
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent,
                    'timestamp': time.time()
                })

                time.sleep(self.sampling_interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class PerformanceRegressionDetector:
    """
    Comprehensive performance regression detection system

    Features:
    - Baseline establishment and maintenance
    - Real-time performance monitoring
    - Regression threshold configuration
    - Historical trend analysis
    - Automated alerting system
    """

    def __init__(self, baseline_file: Path = None, alert_thresholds: Dict[str, float] = None):
        """
        Initialize regression detector

        Args:
            baseline_file: Path to file storing performance baselines
            alert_thresholds: Dictionary of metric thresholds for alerts
        """
        self.baseline_file = baseline_file or (project_root / "tests" / "performance_baselines.json")
        self.baselines: Dict[str, PerformanceBenchmark] = {}
        self.alert_thresholds = alert_thresholds or self._default_alert_thresholds()
        self.logger = self._setup_logging()
        self.resource_monitor = SystemResourceMonitor()

        # Load existing baselines
        self._load_baselines()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance monitoring"""
        logger = logging.getLogger('performance_detector')
        logger.setLevel(logging.INFO)

        # Console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _default_alert_thresholds(self) -> Dict[str, float]:
        """Default alert thresholds for performance metrics"""
        return {
            'execution_time_regression': 20.0,      # 20% slower than baseline
            'memory_usage_regression': 30.0,        # 30% more memory than baseline
            'placement_rate_regression': 5.0,       # 5% lower placement rate
            'efficiency_regression': 10.0,          # 10% lower efficiency
            'critical_execution_time': 300.0,       # 5 minutes absolute limit
            'critical_memory_mb': 1000.0,          # 1GB memory limit
            'critical_placement_rate': 90.0         # 90% minimum placement rate
        }

    def measure_performance(self, test_case: PlacementTestCase,
                          test_execution_func) -> Tuple[PlacementTestResult, PerformanceMetrics]:
        """
        Measure performance metrics during test execution

        Args:
            test_case: Test case being executed
            test_execution_func: Function that executes the test and returns PlacementTestResult

        Returns:
            Tuple of (test_result, performance_metrics)
        """
        self.logger.info(f"📊 Measuring performance for: {test_case.name}")

        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        start_time = time.time()

        try:
            # Execute the test
            test_result = test_execution_func()
            execution_time = time.time() - start_time

            # Stop monitoring and get resource metrics
            resource_metrics = self.resource_monitor.stop_monitoring()

            # Calculate performance metrics
            total_panels = sum(panel.quantity for panel in test_case.panels)
            panels_per_second = total_panels / execution_time if execution_time > 0 else 0

            performance_metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=resource_metrics.get('peak_memory_mb', 0.0),
                cpu_usage_percent=resource_metrics.get('avg_cpu_percent', 0.0),
                placement_rate=test_result.placement_rate,
                efficiency=test_result.average_efficiency,
                panels_per_second=panels_per_second,
                sheets_used=test_result.sheets_used
            )

            self.logger.info(f"📈 Performance: {execution_time:.2f}s, "
                           f"{resource_metrics.get('peak_memory_mb', 0):.1f}MB, "
                           f"{test_result.placement_rate:.1f}% placement")

            return test_result, performance_metrics

        except Exception as e:
            # Stop monitoring even if test fails
            self.resource_monitor.stop_monitoring()
            raise e

    def detect_regressions(self, test_name: str, algorithm: str,
                         current_metrics: PerformanceMetrics) -> List[RegressionAlert]:
        """
        Detect performance regressions by comparing current metrics to baseline

        Args:
            test_name: Name of the test case
            algorithm: Algorithm being tested
            current_metrics: Current performance metrics

        Returns:
            List of regression alerts
        """
        alerts = []
        baseline_key = f"{test_name}_{algorithm}"

        # Check if baseline exists
        if baseline_key not in self.baselines:
            self.logger.info(f"🔍 No baseline found for {baseline_key}, establishing new baseline")
            return alerts

        baseline = self.baselines[baseline_key]
        baseline_metrics = baseline.baseline_metrics

        # Check each performance metric for regression
        alerts.extend(self._check_execution_time_regression(test_name, algorithm,
                                                           baseline_metrics.execution_time,
                                                           current_metrics.execution_time))

        alerts.extend(self._check_memory_regression(test_name, algorithm,
                                                  baseline_metrics.memory_usage_mb,
                                                  current_metrics.memory_usage_mb))

        alerts.extend(self._check_placement_rate_regression(test_name, algorithm,
                                                           baseline_metrics.placement_rate,
                                                           current_metrics.placement_rate))

        alerts.extend(self._check_efficiency_regression(test_name, algorithm,
                                                       baseline_metrics.efficiency,
                                                       current_metrics.efficiency))

        # Check absolute thresholds (critical alerts)
        alerts.extend(self._check_critical_thresholds(test_name, algorithm, current_metrics))

        if alerts:
            self.logger.warning(f"🚨 {len(alerts)} performance regression(s) detected for {baseline_key}")
        else:
            self.logger.info(f"✅ No performance regressions detected for {baseline_key}")

        return alerts

    def establish_baseline(self, test_name: str, algorithm: str, panel_count: int,
                         performance_metrics: PerformanceMetrics):
        """
        Establish or update performance baseline for a test case

        Args:
            test_name: Name of the test case
            algorithm: Algorithm name
            panel_count: Number of panels in the test
            performance_metrics: Measured performance metrics
        """
        baseline_key = f"{test_name}_{algorithm}"

        # Calculate threshold metrics (alerts if exceeded)
        threshold_metrics = PerformanceMetrics(
            execution_time=performance_metrics.execution_time * (1 + self.alert_thresholds['execution_time_regression'] / 100),
            memory_usage_mb=performance_metrics.memory_usage_mb * (1 + self.alert_thresholds['memory_usage_regression'] / 100),
            cpu_usage_percent=performance_metrics.cpu_usage_percent * 1.5,  # 50% higher CPU usage
            placement_rate=performance_metrics.placement_rate * (1 - self.alert_thresholds['placement_rate_regression'] / 100),
            efficiency=performance_metrics.efficiency * (1 - self.alert_thresholds['efficiency_regression'] / 100),
            panels_per_second=performance_metrics.panels_per_second * 0.8,  # 20% slower processing
            sheets_used=int(performance_metrics.sheets_used * 1.2)  # 20% more sheets
        )

        benchmark = PerformanceBenchmark(
            test_name=test_name,
            algorithm=algorithm,
            panel_count=panel_count,
            baseline_metrics=performance_metrics,
            threshold_metrics=threshold_metrics,
            timestamp=datetime.now()
        )

        self.baselines[baseline_key] = benchmark
        self.logger.info(f"📊 Established baseline for {baseline_key}: "
                        f"{performance_metrics.execution_time:.2f}s, "
                        f"{performance_metrics.memory_usage_mb:.1f}MB, "
                        f"{performance_metrics.placement_rate:.1f}%")

    def save_baselines(self):
        """Save current baselines to file"""
        try:
            baseline_data = {}
            for key, benchmark in self.baselines.items():
                baseline_data[key] = {
                    'test_name': benchmark.test_name,
                    'algorithm': benchmark.algorithm,
                    'panel_count': benchmark.panel_count,
                    'baseline_metrics': benchmark.baseline_metrics._asdict(),
                    'threshold_metrics': benchmark.threshold_metrics._asdict(),
                    'timestamp': benchmark.timestamp.isoformat()
                }

            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved {len(self.baselines)} baselines to {self.baseline_file}")

        except Exception as e:
            self.logger.error(f"Failed to save baselines: {e}")

    def _load_baselines(self):
        """Load baselines from file"""
        if not self.baseline_file.exists():
            self.logger.info("📂 No existing baseline file found")
            return

        try:
            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)

            for key, data in baseline_data.items():
                baseline_metrics = PerformanceMetrics(**data['baseline_metrics'])
                threshold_metrics = PerformanceMetrics(**data['threshold_metrics'])
                timestamp = datetime.fromisoformat(data['timestamp'])

                benchmark = PerformanceBenchmark(
                    test_name=data['test_name'],
                    algorithm=data['algorithm'],
                    panel_count=data['panel_count'],
                    baseline_metrics=baseline_metrics,
                    threshold_metrics=threshold_metrics,
                    timestamp=timestamp
                )

                self.baselines[key] = benchmark

            self.logger.info(f"📂 Loaded {len(self.baselines)} baselines from {self.baseline_file}")

        except Exception as e:
            self.logger.error(f"Failed to load baselines: {e}")

    def _check_execution_time_regression(self, test_name: str, algorithm: str,
                                       baseline_time: float, current_time: float) -> List[RegressionAlert]:
        """Check for execution time regression"""
        alerts = []

        if baseline_time <= 0:
            return alerts

        regression_percent = ((current_time - baseline_time) / baseline_time) * 100
        threshold = self.alert_thresholds['execution_time_regression']

        if regression_percent > threshold:
            severity = self._determine_severity(regression_percent, threshold)

            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="execution_time",
                severity=severity,
                baseline_value=baseline_time,
                current_value=current_time,
                regression_percent=regression_percent,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Execution time increased by {regression_percent:.1f}% "
                          f"({baseline_time:.2f}s → {current_time:.2f}s)"
            ))

        return alerts

    def _check_memory_regression(self, test_name: str, algorithm: str,
                               baseline_memory: float, current_memory: float) -> List[RegressionAlert]:
        """Check for memory usage regression"""
        alerts = []

        if baseline_memory <= 0:
            return alerts

        regression_percent = ((current_memory - baseline_memory) / baseline_memory) * 100
        threshold = self.alert_thresholds['memory_usage_regression']

        if regression_percent > threshold:
            severity = self._determine_severity(regression_percent, threshold)

            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="memory_usage",
                severity=severity,
                baseline_value=baseline_memory,
                current_value=current_memory,
                regression_percent=regression_percent,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Memory usage increased by {regression_percent:.1f}% "
                          f"({baseline_memory:.1f}MB → {current_memory:.1f}MB)"
            ))

        return alerts

    def _check_placement_rate_regression(self, test_name: str, algorithm: str,
                                        baseline_rate: float, current_rate: float) -> List[RegressionAlert]:
        """Check for placement rate regression"""
        alerts = []

        if baseline_rate <= 0:
            return alerts

        regression_percent = ((baseline_rate - current_rate) / baseline_rate) * 100
        threshold = self.alert_thresholds['placement_rate_regression']

        if regression_percent > threshold:
            severity = "CRITICAL" if current_rate < 95.0 else self._determine_severity(regression_percent, threshold)

            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="placement_rate",
                severity=severity,
                baseline_value=baseline_rate,
                current_value=current_rate,
                regression_percent=regression_percent,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Placement rate decreased by {regression_percent:.1f}% "
                          f"({baseline_rate:.1f}% → {current_rate:.1f}%)"
            ))

        return alerts

    def _check_efficiency_regression(self, test_name: str, algorithm: str,
                                   baseline_efficiency: float, current_efficiency: float) -> List[RegressionAlert]:
        """Check for material efficiency regression"""
        alerts = []

        if baseline_efficiency <= 0:
            return alerts

        regression_percent = ((baseline_efficiency - current_efficiency) / baseline_efficiency) * 100
        threshold = self.alert_thresholds['efficiency_regression']

        if regression_percent > threshold:
            severity = self._determine_severity(regression_percent, threshold)

            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="efficiency",
                severity=severity,
                baseline_value=baseline_efficiency,
                current_value=current_efficiency,
                regression_percent=regression_percent,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Material efficiency decreased by {regression_percent:.1f}% "
                          f"({baseline_efficiency:.1%} → {current_efficiency:.1%})"
            ))

        return alerts

    def _check_critical_thresholds(self, test_name: str, algorithm: str,
                                 current_metrics: PerformanceMetrics) -> List[RegressionAlert]:
        """Check for critical threshold violations"""
        alerts = []

        # Critical execution time
        if current_metrics.execution_time > self.alert_thresholds['critical_execution_time']:
            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="execution_time_critical",
                severity="CRITICAL",
                baseline_value=self.alert_thresholds['critical_execution_time'],
                current_value=current_metrics.execution_time,
                regression_percent=0.0,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Execution time exceeded critical threshold: "
                          f"{current_metrics.execution_time:.2f}s > "
                          f"{self.alert_thresholds['critical_execution_time']:.0f}s"
            ))

        # Critical memory usage
        if current_metrics.memory_usage_mb > self.alert_thresholds['critical_memory_mb']:
            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="memory_usage_critical",
                severity="CRITICAL",
                baseline_value=self.alert_thresholds['critical_memory_mb'],
                current_value=current_metrics.memory_usage_mb,
                regression_percent=0.0,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Memory usage exceeded critical threshold: "
                          f"{current_metrics.memory_usage_mb:.1f}MB > "
                          f"{self.alert_thresholds['critical_memory_mb']:.0f}MB"
            ))

        # Critical placement rate
        if current_metrics.placement_rate < self.alert_thresholds['critical_placement_rate']:
            alerts.append(RegressionAlert(
                test_name=test_name,
                algorithm=algorithm,
                metric_name="placement_rate_critical",
                severity="CRITICAL",
                baseline_value=self.alert_thresholds['critical_placement_rate'],
                current_value=current_metrics.placement_rate,
                regression_percent=0.0,
                threshold_exceeded=True,
                timestamp=datetime.now(),
                description=f"Placement rate below critical threshold: "
                          f"{current_metrics.placement_rate:.1f}% < "
                          f"{self.alert_thresholds['critical_placement_rate']:.0f}%"
            ))

        return alerts

    def _determine_severity(self, regression_percent: float, threshold: float) -> str:
        """Determine alert severity based on regression percentage"""
        if regression_percent > threshold * 3:
            return "CRITICAL"
        elif regression_percent > threshold * 2:
            return "HIGH"
        elif regression_percent > threshold * 1.5:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_performance_report(self, alerts: List[RegressionAlert] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        alerts = alerts or []

        # Categorize alerts by severity
        alert_summary = {
            'CRITICAL': [a for a in alerts if a.severity == 'CRITICAL'],
            'HIGH': [a for a in alerts if a.severity == 'HIGH'],
            'MEDIUM': [a for a in alerts if a.severity == 'MEDIUM'],
            'LOW': [a for a in alerts if a.severity == 'LOW']
        }

        # Performance trends
        baseline_summary = {}
        for key, benchmark in self.baselines.items():
            baseline_summary[key] = {
                'test_name': benchmark.test_name,
                'algorithm': benchmark.algorithm,
                'panel_count': benchmark.panel_count,
                'execution_time': benchmark.baseline_metrics.execution_time,
                'memory_usage_mb': benchmark.baseline_metrics.memory_usage_mb,
                'placement_rate': benchmark.baseline_metrics.placement_rate,
                'efficiency': benchmark.baseline_metrics.efficiency,
                'established': benchmark.timestamp.isoformat()
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'alert_summary': {
                'total_alerts': len(alerts),
                'critical_count': len(alert_summary['CRITICAL']),
                'high_count': len(alert_summary['HIGH']),
                'medium_count': len(alert_summary['MEDIUM']),
                'low_count': len(alert_summary['LOW'])
            },
            'alerts_by_severity': {
                severity: [asdict(alert) for alert in alert_list]
                for severity, alert_list in alert_summary.items()
            },
            'baseline_summary': baseline_summary,
            'recommendations': self._generate_performance_recommendations(alerts)
        }

    def _generate_performance_recommendations(self, alerts: List[RegressionAlert]) -> List[str]:
        """Generate actionable performance recommendations"""
        recommendations = []

        critical_alerts = [a for a in alerts if a.severity == 'CRITICAL']
        if critical_alerts:
            recommendations.append("🚨 CRITICAL: Immediate action required - performance has severely degraded")

        # Execution time issues
        time_alerts = [a for a in alerts if 'execution_time' in a.metric_name]
        if time_alerts:
            recommendations.append("⏱️ Optimize algorithm efficiency - execution time has increased significantly")

        # Memory issues
        memory_alerts = [a for a in alerts if 'memory' in a.metric_name]
        if memory_alerts:
            recommendations.append("💾 Investigate memory leaks or optimize data structures - memory usage has increased")

        # Placement rate issues
        placement_alerts = [a for a in alerts if 'placement_rate' in a.metric_name]
        if placement_alerts:
            recommendations.append("🎯 Review placement algorithms - placement rate has decreased")

        # Efficiency issues
        efficiency_alerts = [a for a in alerts if 'efficiency' in a.metric_name]
        if efficiency_alerts:
            recommendations.append("📊 Optimize material utilization - efficiency has decreased")

        if not alerts:
            recommendations.append("✅ Performance is stable - no action required")

        return recommendations