#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database initialization script
データベース初期化スクリプト
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.database_models import MaterialDatabase


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('database_init.log')
        ]
    )


def main():
    """Initialize database with sample data"""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("Steel Cutting Optimization System - Database Initialization")
    print("=" * 60)

    # Create database
    db_path = "data/materials.db"
    db = MaterialDatabase(db_path)
    logger.info(f"Database initialized at: {db_path}")

    # Load sample data
    results = db.initialize_from_sample_data()

    print(f"\nDatabase initialization completed:")
    print(f"  Materials loaded: {results['materials_loaded']}")
    print(f"  PI codes loaded: {results['pi_codes_loaded']}")

    # Verify data
    print(f"\nVerifying database contents:")

    # Check materials
    material_types = db.get_material_types()
    print(f"  Material types: {len(material_types)}")
    for mat_type in material_types[:10]:  # Show first 10
        print(f"    - {mat_type}")
    if len(material_types) > 10:
        print(f"    ... and {len(material_types) - 10} more")

    # Check standard sheets
    standard_sheets = db.get_standard_sheets()
    print(f"\n  Standard sheets (1500x3100): {len(standard_sheets)}")
    for sheet in standard_sheets[:5]:  # Show first 5
        print(f"    - {sheet.material_type} {sheet.thickness}t ({sheet.material_code})")
    if len(standard_sheets) > 5:
        print(f"    ... and {len(standard_sheets) - 5} more")

    # Test PI code lookup
    test_pi_codes = ["18131000", "18131001", "18131002"]
    print(f"\n  Testing PI code lookup:")
    for pi_code in test_pi_codes:
        pi_data = db.get_pi_code(pi_code)
        if pi_data:
            print(f"    - {pi_code}: W+{pi_data.width_expansion}, H+{pi_data.height_expansion}")
        else:
            print(f"    - {pi_code}: Not found")

    print(f"\nDatabase is ready for use!")
    print(f"Database file: {Path(db_path).absolute()}")


if __name__ == "__main__":
    main()