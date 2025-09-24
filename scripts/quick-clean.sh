#!/bin/bash
# Quick cleanup script for Steel Cutting Optimization System
# 鋼板切断最適化システム用クイッククリーンアップスクリプト

set -e  # Exit on error

echo "🧹 Steel Cutting Optimization System - Quick Cleanup"
echo "=================================================="

# Function to safely remove patterns
safe_remove() {
    local pattern="$1"
    local description="$2"

    echo "🔍 Cleaning $description..."

    # Use find with -print0 and xargs -0 for safety with spaces in filenames
    if command -v find >/dev/null 2>&1; then
        find . -name "$pattern" -type d -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
        find . -name "$pattern" -type f -print0 2>/dev/null | xargs -0 rm -f 2>/dev/null || true
    else
        echo "  ⚠️  find command not available, skipping $description"
    fi
}

# Clean Python cache files
safe_remove "__pycache__" "Python cache directories"
safe_remove "*.pyc" "Python compiled files"
safe_remove "*.pyo" "Python optimized files"

# Clean type checking cache
safe_remove ".mypy_cache" "MyPy cache"
safe_remove "dmypy.json" "MyPy daemon files"
safe_remove ".dmypy.json" "MyPy daemon configuration"

# Clean testing cache
safe_remove ".pytest_cache" "Pytest cache"
safe_remove ".coverage" "Coverage files"
safe_remove "htmlcov" "Coverage HTML reports"
safe_remove ".hypothesis" "Hypothesis cache"

# Clean OS and editor files
safe_remove ".DS_Store" "macOS metadata files"
safe_remove "Thumbs.db" "Windows thumbnail files"
safe_remove "*.swp" "Vim swap files"
safe_remove "*.swo" "Vim backup files"
safe_remove "*~" "Editor backup files"
safe_remove "nul" "Windows null files"

# Clean temporary files
safe_remove "temp" "Temporary directories"
safe_remove "tmp" "Temporary directories"
safe_remove "temporary" "Temporary directories"

echo ""
echo "✅ Quick cleanup completed!"
echo ""
echo "💡 For detailed cleanup with dry-run option, use:"
echo "   python scripts/cleanup.py --dry-run --verbose"
echo ""
echo "🎯 Project is now clean and ready for development!"