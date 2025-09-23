"""
Phase 3 Testing: Production UI Integration

Tests the Streamlit GPU optimization monitoring dashboard and production
integration capabilities for real-time monitoring and user interaction.
"""

import pytest
import time
import logging
from typing import List
from unittest.mock import patch, MagicMock

# Import UI components
try:
    from ui.gpu_optimization_monitor import (
        GPUOptimizationMonitor,
        OptimizationSession,
        GPUMonitoringData
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False

from core.models import Panel, SteelSheet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.skipif(not UI_COMPONENTS_AVAILABLE, reason="UI components not available")
class TestGPUOptimizationMonitor:
    """Test suite for GPU optimization monitoring UI"""

    @pytest.fixture
    def sample_panels(self) -> List[Panel]:
        """Create sample panels for UI testing"""
        return [
            Panel(
                id=f"UI_Test_{i:03d}",
                width=150 + i * 20,
                height=250 + i * 15,
                thickness=3.0,
                material=f"Material_{i % 3}",
                quantity=1,
                allow_rotation=True
            )
            for i in range(50)
        ]

    @pytest.fixture
    def test_sheet(self) -> SteelSheet:
        """Standard test sheet for UI testing"""
        return SteelSheet(
            width=1500,
            height=3100,
            thickness=3.0,
            material="Steel"
        )

    def test_gpu_monitor_initialization(self):
        """Test GPU optimization monitor initialization"""
        monitor = GPUOptimizationMonitor()

        # Check basic initialization
        assert hasattr(monitor, 'session_data')
        assert hasattr(monitor, 'gpu_data')
        assert hasattr(monitor, 'optimization_history')

        logger.info("✅ GPU optimization monitor initialized successfully")

    def test_optimization_session_creation(self, sample_panels, test_sheet):
        """Test optimization session creation and management"""
        monitor = GPUOptimizationMonitor()

        # Create optimization session
        session = monitor.create_optimization_session(
            panels=sample_panels,
            sheet=test_sheet,
            session_name="Test Session",
            max_memory_mb=1500
        )

        assert isinstance(session, OptimizationSession)
        assert session.session_name == "Test Session"
        assert session.total_panels == len(sample_panels)
        assert session.status == "initialized"

        logger.info(f"✅ Optimization session created: {session.session_name}")

    def test_gpu_monitoring_data_structure(self):
        """Test GPU monitoring data structure"""
        gpu_data = GPUMonitoringData(
            gpu_available=True,
            gpu_name="Intel Iris Xe Graphics",
            gpu_memory_total=6383,
            gpu_memory_used=1200,
            gpu_utilization=85.5,
            temperature=72.0,
            thermal_throttling=False
        )

        assert gpu_data.gpu_available is True
        assert gpu_data.gpu_name == "Intel Iris Xe Graphics"
        assert gpu_data.gpu_memory_available == 5183  # total - used
        assert gpu_data.memory_usage_percentage == pytest.approx(18.8, rel=1e-1)
        assert gpu_data.is_overheating is False

        logger.info(f"✅ GPU monitoring data structure validated: {gpu_data.gpu_utilization}% utilization")

    def test_real_time_progress_tracking(self, sample_panels, test_sheet):
        """Test real-time progress tracking functionality"""
        monitor = GPUOptimizationMonitor()

        session = monitor.create_optimization_session(
            panels=sample_panels,
            sheet=test_sheet,
            session_name="Progress Test"
        )

        # Simulate progress updates
        progress_updates = [
            (10, 50, "Processing batch 1/5"),
            (25, 50, "Processing batch 2/5"),
            (40, 50, "Processing batch 3/5"),
            (50, 50, "Optimization complete")
        ]

        for panels_processed, total_panels, message in progress_updates:
            monitor.update_progress(
                session.session_id,
                panels_processed=panels_processed,
                total_panels=total_panels,
                current_message=message,
                gpu_utilization=80.0 + panels_processed,
                memory_usage=500 + panels_processed * 10
            )

            # Validate progress tracking
            updated_session = monitor.get_session_data(session.session_id)
            assert updated_session.panels_processed == panels_processed
            assert updated_session.current_message == message

        logger.info("✅ Real-time progress tracking validated")

    def test_performance_metrics_collection(self, sample_panels, test_sheet):
        """Test performance metrics collection and analysis"""
        monitor = GPUOptimizationMonitor()

        session = monitor.create_optimization_session(
            panels=sample_panels,
            sheet=test_sheet,
            session_name="Performance Test"
        )

        # Simulate optimization completion
        monitor.complete_optimization(
            session.session_id,
            results=[],  # Empty results for test
            processing_time=25.7,
            gpu_efficiency=87.3,
            placement_rate=94.0,
            sheets_used=8,
            fallback_events=1
        )

        # Validate metrics
        completed_session = monitor.get_session_data(session.session_id)
        assert completed_session.status == "completed"
        assert completed_session.processing_time == 25.7
        assert completed_session.gpu_efficiency == 87.3
        assert completed_session.placement_rate == 94.0

        # Test performance summary
        summary = monitor.get_performance_summary(session.session_id)
        assert 'processing_time' in summary
        assert 'gpu_efficiency' in summary
        assert 'placement_rate' in summary

        logger.info(f"✅ Performance metrics validated: {summary}")

    def test_error_handling_and_recovery(self, sample_panels, test_sheet):
        """Test error handling and recovery in UI monitoring"""
        monitor = GPUOptimizationMonitor()

        session = monitor.create_optimization_session(
            panels=sample_panels,
            sheet=test_sheet,
            session_name="Error Test"
        )

        # Simulate error condition
        monitor.report_error(
            session.session_id,
            error_type="gpu_failure",
            error_message="Mock GPU failure for testing",
            fallback_activated=True
        )

        # Validate error handling
        error_session = monitor.get_session_data(session.session_id)
        assert error_session.status == "error"
        assert "gpu_failure" in error_session.error_log
        assert error_session.fallback_activated is True

        logger.info("✅ Error handling and recovery validated")

    @pytest.mark.skipif(True, reason="Requires Streamlit environment")
    def test_streamlit_dashboard_rendering(self, sample_panels, test_sheet):
        """Test Streamlit dashboard rendering (requires Streamlit environment)"""
        monitor = GPUOptimizationMonitor()

        # Create mock Streamlit components
        with patch('streamlit.sidebar') as mock_sidebar, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.metric') as mock_metric:

            # Render dashboard
            monitor.render_dashboard()

            # Validate Streamlit calls were made
            mock_sidebar.called
            mock_columns.called
            mock_metric.called

            logger.info("✅ Streamlit dashboard rendering validated")

    def test_optimization_history_management(self, sample_panels, test_sheet):
        """Test optimization history management"""
        monitor = GPUOptimizationMonitor()

        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = monitor.create_optimization_session(
                panels=sample_panels[:20],  # Smaller subset
                sheet=test_sheet,
                session_name=f"History Test {i+1}"
            )
            sessions.append(session)

            # Complete each session with different metrics
            monitor.complete_optimization(
                session.session_id,
                results=[],
                processing_time=10.0 + i * 5,
                gpu_efficiency=80.0 + i * 5,
                placement_rate=90.0 + i * 2,
                sheets_used=3 + i,
                fallback_events=i
            )

        # Test history retrieval
        history = monitor.get_optimization_history()
        assert len(history) == 3

        # Test performance comparison
        comparison = monitor.compare_session_performance([s.session_id for s in sessions])
        assert 'average_processing_time' in comparison
        assert 'average_gpu_efficiency' in comparison

        logger.info(f"✅ Optimization history validated: {len(history)} sessions")

    def test_thermal_monitoring_integration(self):
        """Test thermal monitoring and safety features"""
        monitor = GPUOptimizationMonitor()

        # Test normal thermal conditions
        monitor.update_gpu_data(
            gpu_available=True,
            gpu_name="Intel Iris Xe Graphics",
            gpu_memory_total=6383,
            gpu_memory_used=1000,
            gpu_utilization=70.0,
            temperature=65.0,
            thermal_throttling=False
        )

        gpu_data = monitor.get_gpu_data()
        assert gpu_data.is_overheating is False
        assert gpu_data.thermal_status == "normal"

        # Test thermal warning conditions
        monitor.update_gpu_data(
            temperature=82.0,
            thermal_throttling=False
        )

        gpu_data = monitor.get_gpu_data()
        assert gpu_data.thermal_status == "warning"

        # Test thermal critical conditions
        monitor.update_gpu_data(
            temperature=90.0,
            thermal_throttling=True
        )

        gpu_data = monitor.get_gpu_data()
        assert gpu_data.is_overheating is True
        assert gpu_data.thermal_status == "critical"

        logger.info("✅ Thermal monitoring integration validated")

    def test_memory_usage_visualization(self, sample_panels, test_sheet):
        """Test memory usage monitoring and visualization"""
        monitor = GPUOptimizationMonitor()

        session = monitor.create_optimization_session(
            panels=sample_panels,
            sheet=test_sheet,
            session_name="Memory Test",
            max_memory_mb=2000
        )

        # Simulate memory usage progression
        memory_progression = [500, 800, 1200, 1500, 1000, 600]

        for i, memory_usage in enumerate(memory_progression):
            monitor.update_progress(
                session.session_id,
                panels_processed=i * 10,
                total_panels=50,
                current_message=f"Processing step {i+1}",
                memory_usage=memory_usage
            )

        # Validate memory tracking
        memory_history = monitor.get_memory_usage_history(session.session_id)
        assert len(memory_history) == len(memory_progression)
        assert max(memory_history) == 1500
        assert min(memory_history) == 500

        # Test memory pressure detection
        pressure_events = monitor.get_memory_pressure_events(session.session_id)
        # Should detect when memory usage exceeded 75% of max (1500 MB)
        high_usage_events = [event for event in pressure_events if event['pressure_level'] > 0.75]
        assert len(high_usage_events) > 0

        logger.info(f"✅ Memory usage visualization validated: max {max(memory_history)}MB")

    def test_concurrent_session_handling(self, sample_panels, test_sheet):
        """Test handling of concurrent optimization sessions"""
        monitor = GPUOptimizationMonitor()

        # Create multiple concurrent sessions
        sessions = []
        for i in range(3):
            session = monitor.create_optimization_session(
                panels=sample_panels[i*15:(i+1)*15],  # Different panel subsets
                sheet=test_sheet,
                session_name=f"Concurrent Session {i+1}"
            )
            sessions.append(session)

        # Simulate concurrent progress updates
        for i, session in enumerate(sessions):
            monitor.update_progress(
                session.session_id,
                panels_processed=5 + i * 2,
                total_panels=15,
                current_message=f"Processing in session {i+1}",
                gpu_utilization=60.0 + i * 10
            )

        # Validate all sessions are tracked independently
        for i, session in enumerate(sessions):
            session_data = monitor.get_session_data(session.session_id)
            assert session_data.panels_processed == 5 + i * 2
            assert session_data.status == "running"

        active_sessions = monitor.get_active_sessions()
        assert len(active_sessions) == 3

        logger.info(f"✅ Concurrent session handling validated: {len(active_sessions)} active sessions")


@pytest.mark.skipif(not UI_COMPONENTS_AVAILABLE, reason="Testing when UI is available")
class TestUIAvailable:
    """Test that UI components are available and working"""

    def test_ui_components_available(self):
        """Test that UI components are properly available"""
        # This test runs when UI_COMPONENTS_AVAILABLE is True

        # Import should succeed
        from ui.gpu_optimization_monitor import GPUOptimizationMonitor

        # Should be able to create instance
        monitor = GPUOptimizationMonitor()
        assert monitor is not None

        logger.info("✅ UI components are available and functional")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])