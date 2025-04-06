from typing import Optional, List, Dict, Any
from ..base_repository import BaseRepository
from ..manager import DatabaseManager

class CategoryRepository(BaseRepository):
    """Repository for managing word categories."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the category repository."""
        super().__init__(db_manager, 'categories')
        
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a category by its name.
        
        Args:
            name: The category name
            
        Returns:
            The category record if found, None otherwise
        """
        return self.find_one({'name': name})
        
    def get_categories_with_word_count(self) -> List[Dict[str, Any]]:
        """
        Get all categories with their word counts.
        
        Returns:
            List of category records with word counts
        """
        query = """
            SELECT c.*, COUNT(w.id) as word_count
            FROM categories c
            LEFT JOIN words w ON c.id = w.category_id
            GROUP BY c.id
            ORDER BY word_count DESC
        """
        return self.db.execute_query(query)
        
    def get_popular_categories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the categories with the most words.
        
        Args:
            limit: Maximum number of categories to return
            
        Returns:
            List of popular category records
        """
        query = """
            SELECT c.*, COUNT(w.id) as word_count
            FROM categories c
            LEFT JOIN words w ON c.id = w.category_id
            GROUP BY c.id
            ORDER BY word_count DESC
            LIMIT ?
        """
        return self.db.execute_query(query, (limit,))
        
    def get_category_stats(self) -> Dict[str, Any]:
        """
        Get statistics about categories.
        
        Returns:
            Dictionary containing category statistics
        """
        stats = {
            'total_categories': self.count(),
            'categories_with_words': self.db.get_scalar("""
                SELECT COUNT(DISTINCT category_id)
                FROM words
                WHERE category_id IS NOT NULL
            """),
            'words_per_category': self.db.execute_query("""
                SELECT c.name, COUNT(w.id) as word_count
                FROM categories c
                LEFT JOIN words w ON c.id = w.category_id
                GROUP BY c.id
                ORDER BY word_count DESC
            """),
            'avg_words_per_category': self.db.get_scalar("""
                SELECT AVG(word_count)
                FROM (
                    SELECT COUNT(*) as word_count
                    FROM words
                    WHERE category_id IS NOT NULL
                    GROUP BY category_id
                )
            """)
        }
        return stats
        
    def update_category_words(self, category_id: int, word_ids: List[int]) -> None:
        """
        Update the category for multiple words at once.
        
        Args:
            category_id: The category ID to assign
            word_ids: List of word IDs to update
        """
        if not word_ids:
            return
            
        query = """
            UPDATE words
            SET category_id = ?
            WHERE id = ?
        """
        
        params = [(category_id, word_id) for word_id in word_ids]
        self.db.execute_many(query, params)
        
    def delete_category(self, category_id: int) -> bool:
        """
        Delete a category and update its words to have no category.
        
        Args:
            category_id: The category ID to delete
            
        Returns:
            True if the category was deleted
        """
        # First, remove the category from all words
        self.db.execute("""
            UPDATE words
            SET category_id = NULL
            WHERE category_id = ?
        """, (category_id,))
        
        # Then delete the category
        return self.delete(category_id)
        
    def merge_categories(self, source_id: int, target_id: int) -> bool:
        """
        Merge one category into another.
        
        Args:
            source_id: The category ID to merge from
            target_id: The category ID to merge into
            
        Returns:
            True if the merge was successful
        """
        # Update all words from source category to target category
        self.db.execute("""
            UPDATE words
            SET category_id = ?
            WHERE category_id = ?
        """, (target_id, source_id))
        
        # Delete the source category
        return self.delete(source_id) 