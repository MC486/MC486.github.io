from typing import Optional, List, Dict, Any
from .base_repository import BaseRepository
from ..manager import DatabaseManager
import logging

class CategoryRepository(BaseRepository):
    """Repository for managing word categories."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the category repository."""
        super().__init__(db_manager)
        self.table_name = 'categories'
        
        # Create categories table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create words table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER NOT NULL,
                word TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories(id),
                UNIQUE(category_id, word)
            )
        """)
        
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a category by name.
        
        Args:
            name: The name of the category
            
        Returns:
            Category data or None if not found
        """
        result = self.db.execute_query("""
            SELECT *
            FROM categories
            WHERE name = ?
        """, (name,))
        return result[0] if result else None
        
    def get_categories_with_word_count(self) -> List[Dict[str, Any]]:
        """
        Get all categories with their word counts.
        
        Returns:
            List of dictionaries containing category data and word counts
        """
        return self.db.execute_query("""
            SELECT 
                c.*,
                COUNT(w.id) as word_count
            FROM categories c
            LEFT JOIN words w ON c.id = w.category_id
            GROUP BY c.id
            ORDER BY c.name
        """)
        
    def get_popular_categories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most popular categories based on word count.
        
        Args:
            limit: Maximum number of categories to return
            
        Returns:
            List of dictionaries containing category data and word counts
        """
        return self.db.execute_query("""
            SELECT 
                c.*,
                COUNT(w.id) as word_count
            FROM categories c
            LEFT JOIN words w ON c.id = w.category_id
            GROUP BY c.id
            ORDER BY word_count DESC
            LIMIT ?
        """, (limit,))
        
    def get_category_stats(self, category_id: int) -> Dict[str, Any]:
        """
        Get statistics for a category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            Dictionary containing category statistics
        """
        result = self.db.execute_query("""
            SELECT 
                c.*,
                COUNT(w.id) as word_count,
                MIN(LENGTH(w.word)) as min_word_length,
                MAX(LENGTH(w.word)) as max_word_length,
                AVG(LENGTH(w.word)) as avg_word_length
            FROM categories c
            LEFT JOIN words w ON c.id = w.category_id
            WHERE c.id = ?
            GROUP BY c.id
        """, (category_id,))
        
        return result[0] if result else {}
        
    def update_category_words(self, category_id: int, words: List[str]) -> None:
        """
        Update the words in a category.
        
        Args:
            category_id: ID of the category
            words: List of words to add/update
        """
        # First, delete existing words
        self.db.execute_query("""
            DELETE FROM words
            WHERE category_id = ?
        """, (category_id,))
        
        # Then insert new words
        for word in words:
            self.db.execute_query("""
                INSERT INTO words (category_id, word)
                VALUES (?, ?)
            """, (category_id, word))
            
    def delete_category(self, category_id: int) -> bool:
        """
        Delete a category and its words.
        
        Args:
            category_id: ID of the category to delete
            
        Returns:
            True if successful
        """
        try:
            # First delete words
            self.db.execute_query("""
                DELETE FROM words
                WHERE category_id = ?
            """, (category_id,))
            
            # Then delete category
            self.db.execute_query("""
                DELETE FROM categories
                WHERE id = ?
            """, (category_id,))
            
            return True
        except Exception as e:
            logging.error(f"Error deleting category {category_id}: {e}")
            return False
            
    def merge_categories(self, source_id: int, target_id: int) -> bool:
        """
        Merge two categories.
        
        Args:
            source_id: ID of the source category
            target_id: ID of the target category
            
        Returns:
            True if successful
        """
        try:
            # Move words from source to target
            self.db.execute_query("""
                UPDATE words
                SET category_id = ?
                WHERE category_id = ?
            """, (target_id, source_id))
            
            # Delete source category
            self.delete_category(source_id)
            
            return True
        except Exception as e:
            logging.error(f"Error merging categories {source_id} into {target_id}: {e}")
            return False
            
    def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Remove categories and their words that haven't been updated in the specified number of days.
        
        Args:
            days: Number of days after which to remove entries
            
        Returns:
            Number of categories removed
        """
        try:
            # First get the count of categories to be deleted
            result = self.db.execute_query("""
                SELECT COUNT(*) as count
                FROM categories
                WHERE updated_at < datetime('now', ?)
            """, (f"-{days} days",))
            count = result[0]['count'] if result else 0
            
            # Delete words for old categories
            self.db.execute_query("""
                DELETE FROM words
                WHERE category_id IN (
                    SELECT id FROM categories
                    WHERE updated_at < datetime('now', ?)
                )
            """, (f"-{days} days",))
            
            # Delete old categories
            self.db.execute_query("""
                DELETE FROM categories
                WHERE updated_at < datetime('now', ?)
            """, (f"-{days} days",))
            
            return count
        except Exception as e:
            logging.error(f"Error cleaning up old entries: {e}")
            return 0
            
    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the categories table.
        
        Returns:
            The number of entries
        """
        query = "SELECT COUNT(*) FROM categories"
        return self.db.get_scalar(query) or 0
        
    def get_size_bytes(self) -> int:
        """
        Get the size of the categories and words tables in bytes.
        
        Returns:
            Size in bytes
        """
        try:
            result = self.db.execute_query("""
                SELECT SUM(pgsize) as total_size
                FROM dbstat
                WHERE name IN ('categories', 'words')
            """)
            return result[0]['total_size'] if result else 0
        except Exception as e:
            logging.error(f"Error getting table size: {e}")
            return 0 