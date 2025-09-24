#!/usr/bin/env python3
"""
Automated cleanup script for Steel Cutting Optimization System
鋼板切断最適化システム用自動クリーンアップスクリプト

Usage:
    python scripts/cleanup.py [--dry-run] [--verbose]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

class ProjectCleaner:
    """
    Project cleanup utility with logging and safety features
    ログ機能と安全機能付きプロジェクトクリーンアップユーティリティ
    """

    def __init__(self, project_root: Path, dry_run: bool = False, verbose: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.verbose = verbose

        # Patterns to clean up
        self.cleanup_patterns = {
            'python_cache': ['__pycache__', '*.pyc', '*.pyo'],
            'type_cache': ['.mypy_cache', '.dmypy.json', 'dmypy.json'],
            'test_cache': ['.pytest_cache', '.coverage', 'htmlcov/', '.hypothesis/'],
            'ide_files': ['.vscode/', '.idea/', '*.swp', '*.swo', '*~'],
            'os_files': ['.DS_Store', 'Thumbs.db', 'nul'],
            'temp_files': ['temp/', 'tmp/', 'temporary/', 'debug_*.py', 'test_*.py', 'scratch_*.py']
        }

    def find_cleanup_targets(self) -> List[Tuple[str, Path]]:
        """Find files and directories to clean up"""
        targets = []

        for category, patterns in self.cleanup_patterns.items():
            for pattern in patterns:
                if pattern.endswith('/'):
                    # Directory pattern
                    for path in self.project_root.rglob(pattern[:-1]):
                        if path.is_dir():
                            targets.append((category, path))
                elif '*' in pattern:
                    # Glob pattern
                    for path in self.project_root.rglob(pattern):
                        targets.append((category, path))
                else:
                    # Exact match
                    for path in self.project_root.rglob(pattern):
                        targets.append((category, path))

        return targets

    def clean_target(self, path: Path) -> bool:
        """Clean a single target (file or directory)"""
        try:
            if path.is_dir():
                if not self.dry_run:
                    shutil.rmtree(path)
                if self.verbose:
                    print(f"  Removed directory: {path.relative_to(self.project_root)}")
            else:
                if not self.dry_run:
                    path.unlink()
                if self.verbose:
                    print(f"  Removed file: {path.relative_to(self.project_root)}")
            return True
        except Exception as e:
            print(f"  Error removing {path}: {e}")
            return False

    def cleanup(self) -> dict:
        """Perform cleanup and return statistics"""
        targets = self.find_cleanup_targets()

        if not targets:
            print("[OK] No cleanup targets found - project is clean!")
            return {'total': 0, 'cleaned': 0, 'failed': 0, 'categories': {}}

        stats = {'total': len(targets), 'cleaned': 0, 'failed': 0, 'categories': {}}

        # Group targets by category
        by_category = {}
        for category, path in targets:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(path)

        print(f"[CLEAN] Found {len(targets)} cleanup targets")
        if self.dry_run:
            print("[DRY-RUN] DRY RUN - No files will be removed")

        # Clean up by category
        for category, paths in by_category.items():
            print(f"\n[CATEGORY] {category.upper()} ({len(paths)} items)")

            cleaned = 0
            failed = 0

            for path in paths:
                if self.clean_target(path):
                    cleaned += 1
                else:
                    failed += 1

            stats['cleaned'] += cleaned
            stats['failed'] += failed
            stats['categories'][category] = {'cleaned': cleaned, 'failed': failed}

            if not self.dry_run:
                print(f"  [OK] Cleaned {cleaned} items, [ERROR] Failed {failed} items")

        return stats

    def print_summary(self, stats: dict):
        """Print cleanup summary"""
        print(f"\n" + "="*50)
        print(f"[SUMMARY] CLEANUP SUMMARY")
        print(f"="*50)
        print(f"Total targets: {stats['total']}")
        print(f"Successfully cleaned: {stats['cleaned']}")
        print(f"Failed: {stats['failed']}")

        if stats['categories']:
            print(f"\nBy category:")
            for category, cat_stats in stats['categories'].items():
                print(f"  {category}: {cat_stats['cleaned']} cleaned, {cat_stats['failed']} failed")


def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(
        description="Steel Cutting Optimization System - Project Cleanup Tool"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be cleaned without actually removing files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    # Find project root
    current_dir = Path.cwd()
    project_root = None

    # Look for key project files to identify root
    key_files = ['app.py', 'CLAUDE.md', 'requirements.txt']

    for parent in [current_dir] + list(current_dir.parents):
        if any((parent / key_file).exists() for key_file in key_files):
            project_root = parent
            break

    if not project_root:
        print("[ERROR] Could not find project root. Please run from within the project directory.")
        sys.exit(1)

    print("Steel Cutting Optimization System - Cleanup Tool")
    print(f"Project root: {project_root}")

    # Initialize cleaner and run cleanup
    cleaner = ProjectCleaner(project_root, dry_run=args.dry_run, verbose=args.verbose)
    stats = cleaner.cleanup()
    cleaner.print_summary(stats)

    # Exit with appropriate code
    if stats['failed'] > 0:
        print("\nSome cleanup operations failed. Check permissions and try again.")
        sys.exit(1)
    else:
        print("\nProject cleanup completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()