"""
Simple test to check UI component imports
"""

import pytest
import logging

logger = logging.getLogger(__name__)

def test_simple_ui_import():
    """Test simple UI import"""
    try:
        from ui.gpu_optimization_monitor import GPUOptimizationMonitor

        # Try to create instance
        monitor = GPUOptimizationMonitor()
        assert monitor is not None

        logger.info("✅ UI import successful")

    except ImportError as e:
        logger.error(f"❌ UI import failed: {e}")
        pytest.fail(f"Could not import UI components: {e}")
    except Exception as e:
        logger.error(f"❌ UI instantiation failed: {e}")
        pytest.fail(f"Could not create UI instance: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])