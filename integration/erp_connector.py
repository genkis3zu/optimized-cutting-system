"""
ERP/MES System Connector for Steel Cutting Operations
鋼板切断作業用ERP/MESシステムコネクタ

Provides integration with external ERP and MES systems
外部ERPおよびMESシステムとの統合を提供
"""

import asyncio
import logging
import json
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import csv
from pathlib import Path

from core.models import Panel, SteelSheet, PlacementResult
from cutting.instruction import WorkInstruction
from cutting.quality import QualityRecord


class ERPSystemType(Enum):
    """Supported ERP system types"""
    SAP = "sap"
    ORACLE = "oracle"
    GENERIC_REST = "generic_rest"
    GENERIC_SOAP = "generic_soap"
    CSV_IMPORT = "csv_import"
    DATABASE = "database"


class DataFormat(Enum):
    """Data exchange formats"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    EDI = "edi"


class IntegrationStatus(Enum):
    """Integration status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    SYNCING = "syncing"


@dataclass
class ERPConfiguration:
    """ERP system configuration"""
    system_type: ERPSystemType
    endpoint_url: str
    authentication: Dict[str, str]
    data_format: DataFormat
    sync_interval: int = 300  # seconds
    timeout: int = 30         # seconds
    retry_attempts: int = 3
    batch_size: int = 100


@dataclass
class MaterialInventory:
    """Material inventory record"""
    material_code: str
    material_name: str
    thickness: float
    width: float
    height: float
    quantity_available: int
    quantity_reserved: int
    unit_cost: float
    warehouse_location: str
    last_updated: datetime


@dataclass
class WorkOrder:
    """Work order from ERP system"""
    work_order_id: str
    customer_id: str
    project_id: str
    priority: int
    due_date: datetime
    panels_required: List[Panel]
    material_requirements: List[str]
    special_instructions: str
    status: str


@dataclass
class ProductionReport:
    """Production report for ERP system"""
    work_order_id: str
    sheet_id: int
    material_used: str
    panels_produced: List[Dict[str, Any]]
    waste_generated: float
    cutting_time: float
    quality_status: str
    operator: str
    completion_time: datetime


class ERPConnector:
    """
    ERP/MES system connector with multi-protocol support
    マルチプロトコル対応ERP/MESシステムコネクタ
    """

    def __init__(self, config: ERPConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = IntegrationStatus.DISCONNECTED

        # Local database for caching and offline operation
        self.db_path = "data/erp_cache.db"
        self._initialize_local_db()

        # Session management
        self.session = None
        self.last_sync = None

    async def connect(self) -> bool:
        """
        Establish connection to ERP system
        ERPシステムとの接続を確立
        """
        try:
            self.logger.info(f"Connecting to {self.config.system_type.value} ERP system")

            if self.config.system_type in [ERPSystemType.GENERIC_REST, ERPSystemType.SAP]:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )

                # Test connection
                await self._test_connection()

            elif self.config.system_type == ERPSystemType.DATABASE:
                # Database connection test
                self._test_database_connection()

            elif self.config.system_type == ERPSystemType.CSV_IMPORT:
                # File system access test
                self._test_file_access()

            self.status = IntegrationStatus.CONNECTED
            self.logger.info("ERP connection established successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to ERP system: {str(e)}")
            self.status = IntegrationStatus.ERROR
            return False

    async def disconnect(self):
        """Disconnect from ERP system"""
        if self.session:
            await self.session.close()
            self.session = None

        self.status = IntegrationStatus.DISCONNECTED
        self.logger.info("Disconnected from ERP system")

    async def fetch_work_orders(self, date_from: datetime, date_to: datetime) -> List[WorkOrder]:
        """
        Fetch work orders from ERP system
        ERPシステムから作業指示を取得
        """
        try:
            self.logger.info(f"Fetching work orders from {date_from} to {date_to}")

            if self.config.system_type == ERPSystemType.GENERIC_REST:
                return await self._fetch_work_orders_rest(date_from, date_to)
            elif self.config.system_type == ERPSystemType.DATABASE:
                return self._fetch_work_orders_database(date_from, date_to)
            elif self.config.system_type == ERPSystemType.CSV_IMPORT:
                return self._fetch_work_orders_csv()
            else:
                raise NotImplementedError(f"Work order fetch not implemented for {self.config.system_type}")

        except Exception as e:
            self.logger.error(f"Failed to fetch work orders: {str(e)}")
            # Return cached work orders as fallback
            return self._get_cached_work_orders(date_from, date_to)

    async def fetch_material_inventory(self) -> List[MaterialInventory]:
        """
        Fetch material inventory from ERP system
        ERPシステムから材料在庫を取得
        """
        try:
            self.logger.info("Fetching material inventory")

            if self.config.system_type == ERPSystemType.GENERIC_REST:
                return await self._fetch_inventory_rest()
            elif self.config.system_type == ERPSystemType.DATABASE:
                return self._fetch_inventory_database()
            elif self.config.system_type == ERPSystemType.CSV_IMPORT:
                return self._fetch_inventory_csv()
            else:
                raise NotImplementedError(f"Inventory fetch not implemented for {self.config.system_type}")

        except Exception as e:
            self.logger.error(f"Failed to fetch inventory: {str(e)}")
            # Return cached inventory as fallback
            return self._get_cached_inventory()

    async def update_work_order_status(self, work_order_id: str, status: str, notes: str = "") -> bool:
        """
        Update work order status in ERP system
        ERPシステムの作業指示ステータスを更新
        """
        try:
            self.logger.info(f"Updating work order {work_order_id} status to {status}")

            if self.config.system_type == ERPSystemType.GENERIC_REST:
                return await self._update_status_rest(work_order_id, status, notes)
            elif self.config.system_type == ERPSystemType.DATABASE:
                return self._update_status_database(work_order_id, status, notes)
            else:
                # Cache update for batch sync later
                self._cache_status_update(work_order_id, status, notes)
                return True

        except Exception as e:
            self.logger.error(f"Failed to update work order status: {str(e)}")
            # Cache update for retry later
            self._cache_status_update(work_order_id, status, notes)
            return False

    async def submit_production_report(self, report: ProductionReport) -> bool:
        """
        Submit production report to ERP system
        ERPシステムに生産レポートを送信
        """
        try:
            self.logger.info(f"Submitting production report for work order {report.work_order_id}")

            if self.config.system_type == ERPSystemType.GENERIC_REST:
                return await self._submit_report_rest(report)
            elif self.config.system_type == ERPSystemType.DATABASE:
                return self._submit_report_database(report)
            else:
                # Cache report for batch sync later
                self._cache_production_report(report)
                return True

        except Exception as e:
            self.logger.error(f"Failed to submit production report: {str(e)}")
            # Cache report for retry later
            self._cache_production_report(report)
            return False

    async def sync_quality_records(self, records: List[QualityRecord]) -> bool:
        """
        Sync quality records with ERP system
        ERPシステムと品質記録を同期
        """
        try:
            self.logger.info(f"Syncing {len(records)} quality records")

            batch_size = self.config.batch_size
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]

                if self.config.system_type == ERPSystemType.GENERIC_REST:
                    await self._sync_quality_rest(batch)
                elif self.config.system_type == ERPSystemType.DATABASE:
                    self._sync_quality_database(batch)
                else:
                    # Cache for later sync
                    self._cache_quality_records(batch)

            return True

        except Exception as e:
            self.logger.error(f"Failed to sync quality records: {str(e)}")
            # Cache all records for retry
            self._cache_quality_records(records)
            return False

    async def reserve_materials(self, material_requirements: List[Tuple[str, float]]) -> bool:
        """
        Reserve materials in ERP inventory
        ERP在庫で材料を予約
        """
        try:
            self.logger.info(f"Reserving {len(material_requirements)} materials")

            for material_code, quantity in material_requirements:
                if self.config.system_type == ERPSystemType.GENERIC_REST:
                    await self._reserve_material_rest(material_code, quantity)
                elif self.config.system_type == ERPSystemType.DATABASE:
                    self._reserve_material_database(material_code, quantity)
                else:
                    # Log reservation for manual processing
                    self._log_material_reservation(material_code, quantity)

            return True

        except Exception as e:
            self.logger.error(f"Failed to reserve materials: {str(e)}")
            return False

    # Private implementation methods for different ERP systems

    async def _test_connection(self):
        """Test REST API connection"""
        if not self.session:
            raise Exception("No session available")

        auth_headers = self._build_auth_headers()
        test_url = f"{self.config.endpoint_url}/health"

        async with self.session.get(test_url, headers=auth_headers) as response:
            if response.status != 200:
                raise Exception(f"Connection test failed: {response.status}")

    def _test_database_connection(self):
        """Test database connection"""
        db_config = self.config.authentication
        # Implementation depends on database type
        pass

    def _test_file_access(self):
        """Test file system access for CSV import"""
        import_path = Path(self.config.endpoint_url)
        if not import_path.exists():
            raise Exception(f"Import path does not exist: {import_path}")

    async def _fetch_work_orders_rest(self, date_from: datetime, date_to: datetime) -> List[WorkOrder]:
        """Fetch work orders via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/work_orders"

        params = {
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat()
        }

        async with self.session.get(url, headers=auth_headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [self._parse_work_order(wo_data) for wo_data in data.get("work_orders", [])]
            else:
                raise Exception(f"Failed to fetch work orders: {response.status}")

    def _fetch_work_orders_database(self, date_from: datetime, date_to: datetime) -> List[WorkOrder]:
        """Fetch work orders from database"""
        # Implementation depends on database schema
        return []

    def _fetch_work_orders_csv(self) -> List[WorkOrder]:
        """Fetch work orders from CSV files"""
        work_orders = []
        import_path = Path(self.config.endpoint_url)

        for csv_file in import_path.glob("work_orders_*.csv"):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    work_order = self._parse_work_order_csv(row)
                    if work_order:
                        work_orders.append(work_order)

        return work_orders

    async def _fetch_inventory_rest(self) -> List[MaterialInventory]:
        """Fetch inventory via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/inventory"

        async with self.session.get(url, headers=auth_headers) as response:
            if response.status == 200:
                data = await response.json()
                return [self._parse_inventory_item(item) for item in data.get("inventory", [])]
            else:
                raise Exception(f"Failed to fetch inventory: {response.status}")

    def _fetch_inventory_database(self) -> List[MaterialInventory]:
        """Fetch inventory from database"""
        # Implementation depends on database schema
        return []

    def _fetch_inventory_csv(self) -> List[MaterialInventory]:
        """Fetch inventory from CSV files"""
        inventory = []
        import_path = Path(self.config.endpoint_url)

        inventory_file = import_path / "material_inventory.csv"
        if inventory_file.exists():
            with open(inventory_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item = self._parse_inventory_csv(row)
                    if item:
                        inventory.append(item)

        return inventory

    async def _update_status_rest(self, work_order_id: str, status: str, notes: str) -> bool:
        """Update work order status via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/work_orders/{work_order_id}/status"

        data = {
            "status": status,
            "notes": notes,
            "updated_at": datetime.now().isoformat()
        }

        async with self.session.patch(url, headers=auth_headers, json=data) as response:
            return response.status == 200

    def _update_status_database(self, work_order_id: str, status: str, notes: str) -> bool:
        """Update work order status in database"""
        # Implementation depends on database schema
        return True

    async def _submit_report_rest(self, report: ProductionReport) -> bool:
        """Submit production report via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/production_reports"

        data = asdict(report)
        data["completion_time"] = report.completion_time.isoformat()

        async with self.session.post(url, headers=auth_headers, json=data) as response:
            return response.status in [200, 201]

    def _submit_report_database(self, report: ProductionReport) -> bool:
        """Submit production report to database"""
        # Implementation depends on database schema
        return True

    async def _sync_quality_rest(self, records: List[QualityRecord]):
        """Sync quality records via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/quality_records"

        data = {
            "records": [
                {
                    "checkpoint_id": r.checkpoint_id,
                    "inspector": r.inspector,
                    "timestamp": r.timestamp.isoformat(),
                    "measured_value": r.measured_value,
                    "pass_status": r.pass_status,
                    "notes": r.notes
                }
                for r in records
            ]
        }

        async with self.session.post(url, headers=auth_headers, json=data) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Failed to sync quality records: {response.status}")

    def _sync_quality_database(self, records: List[QualityRecord]):
        """Sync quality records to database"""
        # Implementation depends on database schema
        pass

    async def _reserve_material_rest(self, material_code: str, quantity: float):
        """Reserve material via REST API"""
        auth_headers = self._build_auth_headers()
        url = f"{self.config.endpoint_url}/inventory/{material_code}/reserve"

        data = {
            "quantity": quantity,
            "reserved_by": "cutting_optimization_system",
            "reserved_at": datetime.now().isoformat()
        }

        async with self.session.post(url, headers=auth_headers, json=data) as response:
            if response.status not in [200, 201]:
                raise Exception(f"Failed to reserve material: {response.status}")

    def _reserve_material_database(self, material_code: str, quantity: float):
        """Reserve material in database"""
        # Implementation depends on database schema
        pass

    # Utility methods

    def _build_auth_headers(self) -> Dict[str, str]:
        """Build authentication headers"""
        headers = {"Content-Type": "application/json"}

        auth = self.config.authentication
        if "api_key" in auth:
            headers["Authorization"] = f"Bearer {auth['api_key']}"
        elif "username" in auth and "password" in auth:
            import base64
            credentials = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    def _parse_work_order(self, data: Dict[str, Any]) -> WorkOrder:
        """Parse work order from ERP data"""
        # Convert ERP format to internal format
        panels = []
        for panel_data in data.get("panels", []):
            panel = Panel(
                id=panel_data["id"],
                width=panel_data["width"],
                height=panel_data["height"],
                quantity=panel_data["quantity"],
                material=panel_data["material"],
                thickness=panel_data["thickness"]
            )
            panels.append(panel)

        return WorkOrder(
            work_order_id=data["id"],
            customer_id=data.get("customer_id", ""),
            project_id=data.get("project_id", ""),
            priority=data.get("priority", 1),
            due_date=datetime.fromisoformat(data["due_date"]),
            panels_required=panels,
            material_requirements=data.get("materials", []),
            special_instructions=data.get("instructions", ""),
            status=data.get("status", "pending")
        )

    def _parse_work_order_csv(self, row: Dict[str, str]) -> Optional[WorkOrder]:
        """Parse work order from CSV row"""
        try:
            # Simplified CSV parsing
            return WorkOrder(
                work_order_id=row["work_order_id"],
                customer_id=row.get("customer_id", ""),
                project_id=row.get("project_id", ""),
                priority=int(row.get("priority", 1)),
                due_date=datetime.fromisoformat(row["due_date"]),
                panels_required=[],  # Would need separate panels file
                material_requirements=row.get("materials", "").split(","),
                special_instructions=row.get("instructions", ""),
                status=row.get("status", "pending")
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse work order CSV row: {e}")
            return None

    def _parse_inventory_item(self, data: Dict[str, Any]) -> MaterialInventory:
        """Parse inventory item from ERP data"""
        return MaterialInventory(
            material_code=data["material_code"],
            material_name=data["material_name"],
            thickness=data["thickness"],
            width=data["width"],
            height=data["height"],
            quantity_available=data["quantity_available"],
            quantity_reserved=data.get("quantity_reserved", 0),
            unit_cost=data["unit_cost"],
            warehouse_location=data.get("warehouse_location", ""),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )

    def _parse_inventory_csv(self, row: Dict[str, str]) -> Optional[MaterialInventory]:
        """Parse inventory item from CSV row"""
        try:
            return MaterialInventory(
                material_code=row["material_code"],
                material_name=row["material_name"],
                thickness=float(row["thickness"]),
                width=float(row["width"]),
                height=float(row["height"]),
                quantity_available=int(row["quantity_available"]),
                quantity_reserved=int(row.get("quantity_reserved", 0)),
                unit_cost=float(row["unit_cost"]),
                warehouse_location=row.get("warehouse_location", ""),
                last_updated=datetime.now()
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse inventory CSV row: {e}")
            return None

    # Local database operations for caching and offline operation

    def _initialize_local_db(self):
        """Initialize local SQLite database for caching"""
        Path("data").mkdir(exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id TEXT PRIMARY KEY,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory_cache (
                material_code TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _cache_status_update(self, work_order_id: str, status: str, notes: str):
        """Cache status update for later sync"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        update_data = {
            "work_order_id": work_order_id,
            "status": status,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }

        cursor.execute(
            "INSERT INTO pending_updates (update_type, data) VALUES (?, ?)",
            ("status_update", json.dumps(update_data))
        )

        conn.commit()
        conn.close()

    def _cache_production_report(self, report: ProductionReport):
        """Cache production report for later sync"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        report_data = asdict(report)
        report_data["completion_time"] = report.completion_time.isoformat()

        cursor.execute(
            "INSERT INTO pending_updates (update_type, data) VALUES (?, ?)",
            ("production_report", json.dumps(report_data))
        )

        conn.commit()
        conn.close()

    def _cache_quality_records(self, records: List[QualityRecord]):
        """Cache quality records for later sync"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for record in records:
            record_data = {
                "checkpoint_id": record.checkpoint_id,
                "inspector": record.inspector,
                "timestamp": record.timestamp.isoformat(),
                "measured_value": record.measured_value,
                "pass_status": record.pass_status,
                "notes": record.notes
            }

            cursor.execute(
                "INSERT INTO pending_updates (update_type, data) VALUES (?, ?)",
                ("quality_record", json.dumps(record_data))
            )

        conn.commit()
        conn.close()

    def _get_cached_work_orders(self, date_from: datetime, date_to: datetime) -> List[WorkOrder]:
        """Get cached work orders"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM work_orders")
        results = cursor.fetchall()

        work_orders = []
        for (data,) in results:
            try:
                wo_data = json.loads(data)
                work_order = self._parse_work_order(wo_data)
                work_orders.append(work_order)
            except Exception as e:
                self.logger.warning(f"Failed to parse cached work order: {e}")

        conn.close()
        return work_orders

    def _get_cached_inventory(self) -> List[MaterialInventory]:
        """Get cached inventory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT data FROM inventory_cache")
        results = cursor.fetchall()

        inventory = []
        for (data,) in results:
            try:
                inv_data = json.loads(data)
                item = self._parse_inventory_item(inv_data)
                inventory.append(item)
            except Exception as e:
                self.logger.warning(f"Failed to parse cached inventory: {e}")

        conn.close()
        return inventory

    def _log_material_reservation(self, material_code: str, quantity: float):
        """Log material reservation for manual processing"""
        self.logger.info(f"MATERIAL RESERVATION: {material_code} - {quantity} units")


def create_erp_connector(config: ERPConfiguration) -> ERPConnector:
    """Create ERP connector instance"""
    return ERPConnector(config)


# Configuration helpers
def create_sap_config(host: str, client: str, user: str, password: str) -> ERPConfiguration:
    """Create SAP ERP configuration"""
    return ERPConfiguration(
        system_type=ERPSystemType.SAP,
        endpoint_url=f"http://{host}:8000/sap/bc/rest",
        authentication={"username": user, "password": password, "client": client},
        data_format=DataFormat.JSON
    )


def create_rest_config(endpoint_url: str, api_key: str) -> ERPConfiguration:
    """Create generic REST API configuration"""
    return ERPConfiguration(
        system_type=ERPSystemType.GENERIC_REST,
        endpoint_url=endpoint_url,
        authentication={"api_key": api_key},
        data_format=DataFormat.JSON
    )


def create_csv_config(import_path: str) -> ERPConfiguration:
    """Create CSV import configuration"""
    return ERPConfiguration(
        system_type=ERPSystemType.CSV_IMPORT,
        endpoint_url=import_path,
        authentication={},
        data_format=DataFormat.CSV
    )