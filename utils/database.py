"""
Database utilities for order storage and analytics.
Uses SQLite for simplicity, can be easily swapped for PostgreSQL in production.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Thread-local storage for SQLite connections
_local = threading.local()


class Database:
    """
    Database manager for orders, interactions, and analytics.
    """

    def __init__(self, db_path: str = "orders.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        # Track whether we've created tables for this database file
        self._initialized: bool = False

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create a thread-local database connection for this db_path.

        We keep one connection per thread per database file to avoid
        cross-test interference (e.g., tests using temporary DB files).
        """
        conn: Optional[sqlite3.Connection] = getattr(_local, "connection", None)
        conn_path: Optional[str] = getattr(_local, "db_path", None)

        if conn is None or conn_path != self.db_path:
            # Close any existing connection for a different database file
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            _local.connection = conn
            _local.db_path = self.db_path

        return conn

    def _ensure_initialized(self) -> None:
        """
        Ensure the database schema exists.

        This makes endpoint usage and tests robust even if the FastAPI
        lifespan hook hasn't run yet.
        """
        if not self._initialized:
            self.initialize()

    def initialize(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Orders table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                caller TEXT,
                items TEXT,
                total REAL,
                status TEXT DEFAULT 'pending',
                payment_status TEXT DEFAULT 'pending',
                created_at TEXT,
                updated_at TEXT,
                conversation_history TEXT,
                metadata TEXT
            )
        """
        )

        # Interactions table (for call logs)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_sid TEXT,
                direction TEXT,
                text TEXT,
                timestamp TEXT,
                metadata TEXT
            )
        """
        )

        # Analytics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                event_data TEXT,
                timestamp TEXT
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_caller ON orders(caller)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_call_sid ON interactions(call_sid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type)")

        conn.commit()
        self._initialized = True
        logger.info(f"Database initialized: {self.db_path}")

    def save_order(self, order_data: Dict[str, Any]) -> str:
        """
        Save an order to the database.

        Args:
            order_data: Order information dictionary

        Returns:
            Order ID
        """
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        order_id = order_data.get("id")
        now = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO orders 
            (id, caller, items, total, status, payment_status, created_at, updated_at, conversation_history, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                order_id,
                order_data.get("caller", ""),
                json.dumps(order_data.get("items", [])),
                order_data.get("total", 0),
                order_data.get("status", "pending"),
                order_data.get("payment_status", "pending"),
                order_data.get("created_at", now),
                now,
                json.dumps(order_data.get("conversation_history", [])),
                json.dumps(order_data.get("metadata", {})),
            ),
        )

        conn.commit()
        logger.info(f"Order saved: {order_id}")

        # Log analytics event
        self._log_event("order_created", {"order_id": order_id, "total": order_data.get("total", 0)})

        return order_id

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an order by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order data dictionary or None
        """
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_order(row)
        return None

    def _row_to_order(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to order dictionary."""
        return {
            "id": row["id"],
            "caller": row["caller"],
            "items": json.loads(row["items"]),
            "total": row["total"],
            "status": row["status"],
            "payment_status": row["payment_status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "conversation_history": json.loads(row["conversation_history"]),
            "metadata": json.loads(row["metadata"]),
        }

    def get_orders_by_phone(self, phone: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get orders by phone number."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM orders WHERE caller = ? ORDER BY created_at DESC LIMIT ?",
            (phone, limit),
        )

        return [self._row_to_order(row) for row in cursor.fetchall()]

    def update_order_status(self, order_id: str, status: str) -> bool:
        """Update order status."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE orders SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), order_id),
        )
        conn.commit()

        self._log_event("order_status_changed", {"order_id": order_id, "status": status})

        return cursor.rowcount > 0

    def update_payment_status(self, order_id: str, payment_status: str) -> bool:
        """Update payment status."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE orders SET payment_status = ?, updated_at = ? WHERE id = ?",
            (payment_status, datetime.now().isoformat(), order_id),
        )
        conn.commit()

        self._log_event("payment_status_changed", {"order_id": order_id, "status": payment_status})

        return cursor.rowcount > 0

    def save_interaction(self, interaction: Dict[str, Any]):
        """Save a call interaction log."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO interactions (call_sid, direction, text, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                interaction.get("call_sid", ""),
                interaction.get("direction", ""),
                interaction.get("text", ""),
                interaction.get("timestamp", datetime.now().isoformat()),
                json.dumps(interaction.get("metadata", {})),
            ),
        )

        conn.commit()

    def get_call_transcript(self, call_sid: str) -> List[Dict[str, Any]]:
        """Get full transcript for a call."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM interactions WHERE call_sid = ? ORDER BY timestamp",
            (call_sid,),
        )

        return [
            {
                "direction": row["direction"],
                "text": row["text"],
                "timestamp": row["timestamp"],
            }
            for row in cursor.fetchall()
        ]

    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an analytics event."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO analytics (event_type, event_data, timestamp)
            VALUES (?, ?, ?)
        """,
            (event_type, json.dumps(event_data), datetime.now().isoformat()),
        )

        conn.commit()

    # Analytics methods

    def get_call_count(self, days: int = 30) -> int:
        """Get total number of calls in the last N days."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute(
            "SELECT COUNT(DISTINCT call_sid) FROM interactions WHERE timestamp > ?",
            (since,),
        )

        return cursor.fetchone()[0]

    def get_order_count(self, days: int = 30) -> int:
        """Get total number of orders in the last N days."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM orders WHERE created_at > ?", (since,))

        return cursor.fetchone()[0]

    def get_popular_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular ordered items."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT items FROM orders")

        item_counts = {}
        for row in cursor.fetchall():
            items = json.loads(row[0])
            for item in items:
                name = item.get("name", "Unknown")
                item_counts[name] = item_counts.get(name, 0) + 1

        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

        return [{"name": name, "count": count} for name, count in sorted_items[:limit]]

    def get_average_order_value(self, days: int = 30) -> float:
        """Get average order value in the last N days."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("SELECT AVG(total) FROM orders WHERE created_at > ?", (since,))

        result = cursor.fetchone()[0]
        return round(result, 2) if result else 0.0

    def get_daily_revenue(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily revenue for the last N days."""
        self._ensure_initialized()
        conn = self._get_connection()
        cursor = conn.cursor()

        results = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            cursor.execute("SELECT SUM(total) FROM orders WHERE DATE(created_at) = ?", (date_str,))

            total = cursor.fetchone()[0] or 0
            results.append({"date": date_str, "revenue": round(total, 2)})

        return results

    def close(self):
        """Close database connection."""
        if hasattr(_local, "connection") and _local.connection:
            try:
                _local.connection.close()
            except Exception:
                pass
            _local.connection = None
        if hasattr(_local, "db_path"):
            _local.db_path = None
        self._initialized = False
