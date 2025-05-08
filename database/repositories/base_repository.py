"""
Base Repository Pattern Implementation

This module provides a generic base repository class that implements common database operations
using the Repository Pattern. The BaseRepository class serves as a foundation for all domain-specific
repositories in the application, providing consistent CRUD operations and query capabilities.

Key Features:
- Generic type support for type-safe repository implementations
- Common CRUD operations (Create, Read, Update, Delete)
- Advanced query capabilities (find, find_one, count)
- SQLite-specific optimizations
- Transaction management through DatabaseManager
- Type hints for better IDE support and code safety
"""

from typing import TypeVar, Generic, Optional, List, Dict, Any
from ..manager import DatabaseManager
import logging
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class providing common database operations."""
    
    def __init__(self, db_manager: DatabaseManager, table_name: str):
        """
        Initialize the base repository.
        
        Args:
            db_manager: DatabaseManager instance
            table_name: Name of the table to manage
            
        Raises:
            ValueError: If table_name is None or empty
        """
        if not table_name:
            raise ValueError("table_name is required")
            
        self.db_manager = db_manager
        self.table_name = table_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @contextmanager
    def get_connection(self):
        """
        Get a database connection with automatic cleanup.
        
        Yields:
            sqlite3.Connection: A database connection
        """
        with self.db_manager.get_connection() as conn:
            yield conn
            
    def create(self, data: Dict[str, Any]) -> int:
        """
        Create a new record in the table.
        
        Args:
            data: Dictionary of column names and values
            
        Returns:
            The ID of the created record
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"""
            INSERT INTO {self.table_name} ({columns})
            VALUES ({placeholders})
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid
        
    def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """
        Get a record by its ID.
        
        Args:
            id: The record ID
            
        Returns:
            The record as a dictionary, or None if not found
        """
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE id = ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
        
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all records from the table.
        
        Returns:
            List of records as dictionaries
        """
        query = f"SELECT * FROM {self.table_name}"
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def update(self, id: int, data: Dict[str, Any]) -> bool:
        """
        Update a record by its ID.
        
        Args:
            id: The record ID
            data: Dictionary of column names and values to update
            
        Returns:
            True if the record was updated, False otherwise
        """
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"""
            UPDATE {self.table_name}
            SET {set_clause}
            WHERE id = ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(data.values()) + (id,))
            return cursor.rowcount > 0
        
    def delete(self, id: int) -> bool:
        """
        Delete a record by its ID.
        
        Args:
            id: The record ID
            
        Returns:
            True if the record was deleted, False otherwise
        """
        query = f"""
            DELETE FROM {self.table_name}
            WHERE id = ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            return cursor.rowcount > 0
        
    def find(self, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find records matching the given conditions.
        
        Args:
            conditions: Dictionary of column names and values to match
            
        Returns:
            List of matching records as dictionaries
        """
        if not conditions:
            return self.get_all()
            
        where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(conditions.values()))
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def find_one(self, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single record matching the given conditions.
        
        Args:
            conditions: Dictionary of column names and values to match
            
        Returns:
            The first matching record as a dictionary, or None if not found
        """
        if not conditions:
            return None
            
        where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
            LIMIT 1
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(conditions.values()))
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
        
    def count(self, conditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records matching the given conditions.
        
        Args:
            conditions: Optional dictionary of column names and values to match
            
        Returns:
            Number of matching records
        """
        if not conditions:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                return cursor.fetchone()[0]
                
        where_clause = ' AND '.join([f"{k} = ?" for k in conditions.keys()])
        query = f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE {where_clause}
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(conditions.values()))
            return cursor.fetchone()[0]

    def get_size_bytes(self) -> int:
        """
        Get the size of the table in bytes.
        
        Returns:
            Size of the table in bytes
        """
        query = f"""
            SELECT SUM(pgsize) 
            FROM dbstat 
            WHERE name = ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (self.table_name,))
            return cursor.fetchone()[0] or 0

    def cleanup_old_entries(self, days: int) -> int:
        """
        Remove entries older than the specified number of days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of entries removed
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        query = f"""
            DELETE FROM {self.table_name}
            WHERE created_at < ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (cutoff_date,))
            return cursor.rowcount

    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the table.
        
        Returns:
            Total number of entries
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchone()[0] 
        cutoff_date = datetime.now() - timedelta(days=days)
        query = f"""
            DELETE FROM {self.table_name}
            WHERE created_at < ?
        """
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (cutoff_date,))
            return cursor.rowcount

    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the table.
        
        Returns:
            Total number of entries
        """
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchone()[0] 