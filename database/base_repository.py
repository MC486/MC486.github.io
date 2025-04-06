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

Usage:
    class WordRepository(BaseRepository[Word]):
        def __init__(self, db_manager: DatabaseManager):
            super().__init__(db_manager, 'words')
            
        def get_by_word(self, word: str) -> Optional[Word]:
            return self.find_one({'word': word})

The BaseRepository class is designed to be extended by domain-specific repositories,
providing a consistent interface for database operations while allowing for custom
query methods specific to each domain.
"""

from typing import TypeVar, Generic, Optional, List, Dict, Any
from .manager import DatabaseManager

T = TypeVar('T')

class BaseRepository(Generic[T]):
    """Base repository class providing common database operations."""
    
    def __init__(self, db_manager: DatabaseManager, table_name: str):
        """
        Initialize the base repository.
        
        Args:
            db_manager: DatabaseManager instance
            table_name: Name of the table this repository manages
        """
        self.db = db_manager
        self.table_name = table_name
        
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