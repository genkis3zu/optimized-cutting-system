# Cleanup Scripts

This directory contains automated cleanup utilities for the Steel Cutting Optimization System.

## Available Scripts

### 1. Python Cleanup Tool (`cleanup.py`)

Comprehensive Python-based cleanup utility with dry-run support and detailed logging.

**Usage:**
```bash
# Dry run - see what would be cleaned without making changes
python scripts/cleanup.py --dry-run --verbose

# Actual cleanup
python scripts/cleanup.py

# Show detailed output
python scripts/cleanup.py --verbose
```

**Features:**
- Removes Python cache files (`__pycache__`, `*.pyc`, `*.pyo`)
- Cleans type checking cache (`.mypy_cache`, `dmypy.json`)
- Removes test cache (`.pytest_cache`, `.coverage`, `htmlcov`)
- Cleans IDE files (`.vscode`, `.idea`, swap files)
- Removes OS files (`.DS_Store`, `Thumbs.db`, `nul`)
- Cleans temporary files and directories
- Dry-run mode for safety
- Detailed logging and statistics

### 2. Quick Clean Shell Script (`quick-clean.sh`)

Fast shell script for Unix/Linux/macOS systems.

**Usage:**
```bash
./scripts/quick-clean.sh
```

### 3. Quick Clean Batch Script (`quick-clean.bat`)

Windows batch script equivalent.

**Usage:**
```cmd
scripts\quick-clean.bat
```

## What Gets Cleaned

| Category | Files/Directories |
|----------|-------------------|
| **Python Cache** | `__pycache__/`, `*.pyc`, `*.pyo` |
| **Type Cache** | `.mypy_cache/`, `dmypy.json` |
| **Test Cache** | `.pytest_cache/`, `.coverage`, `htmlcov/` |
| **IDE Files** | `.vscode/`, `.idea/`, `*.swp`, `*.swo` |
| **OS Files** | `.DS_Store`, `Thumbs.db`, `nul` |
| **Temp Files** | `temp/`, `tmp/`, `debug_*.py`, `test_*.py` |

## Safety Features

- **Dry-run mode**: Preview changes before execution
- **Error handling**: Graceful handling of permission issues
- **Path validation**: Only cleans within project directory
- **Logging**: Detailed output of all operations
- **Statistics**: Summary of cleanup results

## Automation

You can add these scripts to your development workflow:

**Pre-commit hook:**
```bash
python scripts/cleanup.py --dry-run
```

**CI/CD pipeline:**
```bash
python scripts/cleanup.py
```

**Daily development:**
```bash
./scripts/quick-clean.sh
```

## Troubleshooting

If you encounter permission errors:
1. Ensure you have write permissions to the project directory
2. Check if any processes are using the files to be cleaned
3. Use `--verbose` flag to see detailed error messages

For Windows users, if the batch script doesn't work:
- Run Command Prompt as Administrator
- Use the Python script instead: `python scripts/cleanup.py`