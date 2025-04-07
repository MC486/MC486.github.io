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

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class providing common database operations."""
    
    def __init__(self, db_manager: DatabaseManager, table_name: Optional[str] = None):
        """
        Initialize the base repository.
        
        Args:
            db_manager: DatabaseManager instance
            table_name: Optional name of the table to manage
        """
        self.db = db_manager
        self.table_name = table_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        
        self.db.execute(query, tuple(data.values()))
        return self.db.get_scalar("SELECT last_insert_rowid()")
        
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
        return self.db.get_one(query, (id,))
        
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all records from the table.
        
        Returns:
            List of records as dictionaries
        """
        query = f"SELECT * FROM {self.table_name}"
        return self.db.execute_query(query)
        
    def update(self, id: int, data: Dict[str, Any]) -> bool:
        """
        Update a record by its ID.
        
        Args:
            id: The record ID
            data: Dictionary of column names and new values
            
        Returns:
            True if the record was updated, False otherwise
        """
        if not data:
            return False
            
        set_clause = ', '.join(f"{k} = ?" for k in data.keys())
        query = f"""
            UPDATE {self.table_name}
            SET {set_clause}
            WHERE id = ?
        """
        
        params = list(data.values()) + [id]
        self.db.execute(query, tuple(params))
        return True
        
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
        
        self.db.execute(query, (id,))
        return True
        
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
            
        where_clause = ' AND '.join(f"{k} = ?" for k in conditions.keys())
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
        """
        
        return self.db.execute_query(query, tuple(conditions.values()))
        
    def find_one(self, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single record matching the given conditions.
        
        Args:
            conditions: Dictionary of column names and values to match
            
        Returns:
            The matching record as a dictionary, or None if not found
        """
        if not conditions:
            return None
            
        where_clause = ' AND '.join(f"{k} = ?" for k in conditions.keys())
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
            LIMIT 1
        """
        
        return self.db.get_one(query, tuple(conditions.values()))
        
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
            return self.db.get_scalar(query) or 0
            
        where_clause = ' AND '.join(f"{k} = ?" for k in conditions.keys())
        query = f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE {where_clause}
        """
        
        return self.db.get_scalar(query, tuple(conditions.values())) or 0

    def get_size_bytes(self) -> int:
        """
        Get the size of the table in bytes.

        Returns:
            Size in bytes
        """
        if not self.table_name:
            return 0

        try:
            result = self.db.execute_query("""
                SELECT pgsize as total_size
                FROM dbstat
                WHERE name = ?
            """, (self.table_name,))
            return result[0]['total_size'] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting table size: {e}")
            return 0

    def cleanup_old_entries(self, days: int) -> int:
        """
        Remove entries older than the specified number of days.

        Args:
            days: Number of days of inactivity before removal

        Returns:
            Number of entries removed
        """
        if not self.table_name:
            return 0

        try:
            # Check if created_at column exists
            table_info = self.db.execute_query(f"PRAGMA table_info({self.table_name})")
            has_created_at = any(col['name'] == 'created_at' for col in table_info)

            if not has_created_at:
                self.logger.warning(f"Table {self.table_name} does not have created_at column")
                return 0

            cutoff_date = datetime.now() - timedelta(days=days)
            result = self.db.execute_query(f"""
                DELETE FROM {self.table_name}
                WHERE created_at < ?
                RETURNING COUNT(*) as deleted_count
            """, (cutoff_date,))

            return result[0]['deleted_count'] if result else 0
        except Exception as e:
            self.logger.error(f"Error cleaning up old entries: {e}")
            return 0

    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the table.

        Returns:
            The number of entries
        """
        if not self.table_name:
            return 0

        try:
            result = self.db.execute_query(f"SELECT COUNT(*) as count FROM {self.table_name}")
            return result[0]['count'] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting entry count: {e}")
            return 0 