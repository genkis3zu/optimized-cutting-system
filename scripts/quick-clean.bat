@echo off
REM Quick cleanup batch script for Steel Cutting Optimization System
REM 鋼板切断最適化システム用クイッククリーンアップスクリプト

echo 🧹 Steel Cutting Optimization System - Quick Cleanup
echo ==================================================

echo 🔍 Cleaning Python cache directories...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1

echo 🔍 Cleaning Python compiled files...
for /r %%f in (*.pyc *.pyo) do @if exist "%%f" del /q "%%f" >nul 2>&1

echo 🔍 Cleaning MyPy cache...
if exist ".mypy_cache" rd /s /q ".mypy_cache" >nul 2>&1
if exist "dmypy.json" del /q "dmypy.json" >nul 2>&1
if exist ".dmypy.json" del /q ".dmypy.json" >nul 2>&1

echo 🔍 Cleaning test cache...
if exist ".pytest_cache" rd /s /q ".pytest_cache" >nul 2>&1
if exist ".coverage" del /q ".coverage" >nul 2>&1
if exist "htmlcov" rd /s /q "htmlcov" >nul 2>&1
if exist ".hypothesis" rd /s /q ".hypothesis" >nul 2>&1

echo 🔍 Cleaning OS and editor files...
for /r %%f in (.DS_Store Thumbs.db *.swp *.swo *~ nul) do @if exist "%%f" del /q "%%f" >nul 2>&1

echo 🔍 Cleaning temporary directories...
for /d %%d in (temp tmp temporary) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1

echo.
echo ✅ Quick cleanup completed!
echo.
echo 💡 For detailed cleanup with dry-run option, use:
echo    python scripts/cleanup.py --dry-run --verbose
echo.
echo 🎯 Project is now clean and ready for development!

pause