from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from .base_repository import BaseRepository
from ..manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class CategoryRepository(BaseRepository):
    """Repository for managing word categories."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the repository.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__(db_manager, 'categories')
        
        # Define custom SQLite functions
        self._is_palindrome = lambda word: word == word[::-1]
        self._scrabble_score = lambda word: sum({
            'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4,
            'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3,
            'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8,
            'y': 4, 'z': 10
        }.get(c, 0) for c in word.lower())
        self._vowel_count = lambda word: sum(1 for c in word.lower() if c in {'a', 'e', 'i', 'o', 'u'})
        self._consonant_count = lambda word: sum(1 for c in word.lower() if c.isalpha() and c not in {'a', 'e', 'i', 'o', 'u'})
        
    def _register_custom_functions(self, conn):
        """Register custom SQLite functions with the connection."""
        conn.create_function('is_palindrome', 1, self._is_palindrome)
        conn.create_function('scrabble_score', 1, self._scrabble_score)
        conn.create_function('vowel_count', 1, self._vowel_count)
        conn.create_function('consonant_count', 1, self._consonant_count)
        
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a category by name.
        
        Args:
            name: The name of the category
            
        Returns:
            Category data or None if not found
        """
        try:
            result = self.db_manager.execute_query("""
                SELECT *
                FROM categories
                WHERE name = ?
            """, (name,))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting category by name {name}: {str(e)}")
            return None
        
    def get_category_by_id(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a category by ID.
        
        Args:
            category_id: The ID of the category
            
        Returns:
            Category data or None if not found
        """
        try:
            result = self.db_manager.execute_query("""
                SELECT *
                FROM categories
                WHERE id = ?
            """, (category_id,))
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting category by ID {category_id}: {str(e)}")
            return None
        
    def get_category_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a category by name.
        
        Args:
            name: The name of the category
            
        Returns:
            Category data or None if not found
        """
        return self.get_by_name(name)
        
    def get_categories_with_word_count(self) -> List[Dict[str, Any]]:
        """
        Get all categories with their word counts.
        
        Returns:
            List of dictionaries containing category data and word counts
        """
        try:
            return self.db_manager.execute_query("""
                SELECT 
                    c.*,
                    COUNT(w.id) as word_count
                FROM categories c
                LEFT JOIN words w ON c.id = w.category_id
                GROUP BY c.id
                ORDER BY c.name
            """)
        except Exception as e:
            logger.error(f"Error getting categories with word count: {str(e)}")
            return []
        
    def get_popular_categories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most popular categories based on word count.
        
        Args:
            limit: Maximum number of categories to return
            
        Returns:
            List of dictionaries containing category data and word counts
        """
        try:
            return self.db_manager.execute_query("""
                SELECT 
                    c.id,
                    c.name,
                    c.description,
                    c.created_at,
                    c.updated_at,
                    COUNT(w.id) as word_count,
                    SUM(w.frequency) as total_frequency
                FROM categories c
                LEFT JOIN words w ON c.id = w.category_id
                GROUP BY c.id
                ORDER BY total_frequency DESC, word_count DESC, name
                LIMIT ?
            """, (limit,))
        except Exception as e:
            logger.error(f"Error getting popular categories: {str(e)}")
            return []
        
    def get_category_stats(self, category_id: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a category.
        
        Args:
            category_id: The ID of the category
            
        Returns:
            Dictionary of statistics or None if category not found
        """
        try:
            # First check if the category exists
            category = self.get_category_by_id(category_id)
            if not category:
                return None
                
            # Get word statistics
            result = self.db_manager.execute_query("""
                SELECT 
                    COUNT(*) as word_count,
                    SUM(frequency) as total_frequency,
                    AVG(frequency) as average_frequency,
                    SUM(CASE WHEN allowed = 1 THEN 1 ELSE 0 END) as allowed_word_count,
                    SUM(CASE WHEN allowed = 0 THEN 1 ELSE 0 END) as disallowed_word_count,
                    MIN(LENGTH(word)) as min_word_length,
                    MAX(LENGTH(word)) as max_word_length,
                    AVG(LENGTH(word)) as average_word_length
                FROM words
                WHERE category_id = ?
            """, (category_id,))
            
            if not result:
                return None
                
            stats = result[0]
            return {
                'word_count': stats['word_count'],
                'total_frequency': stats['total_frequency'],
                'average_frequency': stats['average_frequency'],
                'allowed_word_count': stats['allowed_word_count'],
                'disallowed_word_count': stats['disallowed_word_count'],
                'min_word_length': stats['min_word_length'],
                'max_word_length': stats['max_word_length'],
                'average_word_length': stats['average_word_length']
            }
        except Exception as e:
            logger.error(f"Error getting category stats: {str(e)}")
            return None
        
    def update_category_words(self, category_id: int, word_ids: List[int]) -> None:
        """
        Update the category for multiple words.
        
        Args:
            category_id: ID of the category
            word_ids: List of word IDs to update
        """
        try:
            # Update words to new category
            for word_id in word_ids:
                self.db_manager.execute_query("""
                    UPDATE words
                    SET category_id = ?
                    WHERE id = ?
                """, (category_id, word_id))
        except Exception as e:
            logger.error(f"Error updating category words: {str(e)}")
            
    def delete_category(self, category_id: int) -> bool:
        """
        Delete a category and all its words.
        
        Args:
            category_id: The ID of the category to delete
            
        Returns:
            True if the category was deleted, False if it didn't exist
        """
        try:
            # First check if the category exists
            category = self.get_category_by_id(category_id)
            if not category:
                return False
                
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Delete all words in the category
                cursor.execute("""
                    DELETE FROM words
                    WHERE category_id = ?
                """, (category_id,))
                
                # Delete the category
                cursor.execute("""
                    DELETE FROM categories
                    WHERE id = ?
                """, (category_id,))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting category {category_id}: {str(e)}")
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
            # Update words to target category
            self.db_manager.execute_query("""
                UPDATE words
                SET category_id = ?
                WHERE category_id = ?
            """, (target_id, source_id))
            
            # Delete source category
            self.db_manager.execute_query("""
                DELETE FROM categories
                WHERE id = ?
            """, (source_id,))
            
            return True
        except Exception as e:
            logger.error(f"Error merging categories {source_id} into {target_id}: {str(e)}")
            return False
            
    def cleanup_old_entries(self, days: int) -> int:
        """Delete categories where all associated words are older than the specified number of days.

        Args:
            days (int): Number of days to consider as old.

        Returns:
            int: Number of categories deleted.
        """
        try:
            old_date = datetime.now() - timedelta(days=days)
            old_date_str = old_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # First, get categories where all words are older than the specified date
            query = """
                SELECT DISTINCT c.id
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN datetime(w.updated_at) <= datetime(?) THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND datetime(w2.updated_at) > datetime(?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.updated_at IS NULL
                )
            """
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (old_date_str, old_date_str))
                category_ids = [row[0] for row in cursor.fetchall()]
                
                if not category_ids:
                    return 0
                    
                # Delete words first (due to foreign key constraint)
                word_query = "DELETE FROM words WHERE category_id IN ({})".format(
                    ','.join(['?'] * len(category_ids))
                )
                cursor.execute(word_query, category_ids)
                
                # Then delete categories
                category_query = "DELETE FROM categories WHERE id IN ({})".format(
                    ','.join(['?'] * len(category_ids))
                )
                cursor.execute(category_query, category_ids)
                
                conn.commit()
                return len(category_ids)
                
        except Exception as e:
            logger.error(f"Error cleaning up old entries: {e}")
            return 0
            
    def get_entry_count(self) -> int:
        """
        Get the total number of categories.
        
        Returns:
            Number of categories
        """
        try:
            result = self.db_manager.execute_query("""
                SELECT COUNT(*) as count
                FROM categories
            """)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting category count: {str(e)}")
            return 0
            
    def get_size_bytes(self) -> int:
        """
        Get the approximate size of the categories table in bytes.
        
        Returns:
            Size in bytes
        """
        try:
            result = self.db_manager.execute_query("""
                SELECT SUM(length(name) + length(description)) as size
                FROM categories
            """)
            return result[0]['size'] if result else 0
        except Exception as e:
            logger.error(f"Error getting category size: {str(e)}")
            return 0
            
    def bulk_create_categories(self, categories: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Create multiple categories at once.
        
        Args:
            categories: List of dictionaries containing name and description
            
        Returns:
            List of created category dictionaries
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                created_categories = []
                for category in categories:
                    cursor.execute("""
                        INSERT INTO categories (name, description)
                        VALUES (?, ?)
                    """, (category['name'], category['description']))
                    category_id = cursor.lastrowid
                    created_categories.append({
                        'id': category_id,
                        'name': category['name'],
                        'description': category['description'],
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                conn.commit()
                return created_categories
        except Exception as e:
            logger.error(f"Error bulk creating categories: {str(e)}")
            return []
            
    def add_category(self, name: str, description: Optional[str] = None) -> int:
        """
        Add a new category.
        
        Args:
            name: Name of the category
            description: Optional description
            
        Returns:
            ID of the created category
        """
        try:
            data = {
                'name': name,
                'description': description or '',
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            return self.create(data)
        except Exception as e:
            logger.error(f"Error adding category {name}: {str(e)}")
            return 0
            
    def get_categories_by_word_count_range(self, min_count: int, max_count: int) -> List[Dict[str, Any]]:
        """
        Get categories with word counts in a specific range.
        
        Args:
            min_count: Minimum word count
            max_count: Maximum word count
            
        Returns:
            List of category records with word counts
        """
        try:
            return self.db_manager.execute_query("""
                SELECT 
                    c.*,
                    COUNT(w.id) as word_count
                FROM categories c
                LEFT JOIN words w ON c.id = w.category_id
                GROUP BY c.id
                HAVING word_count >= ? AND word_count <= ?
                ORDER BY word_count DESC
            """, (min_count, max_count))
        except Exception as e:
            logger.error(f"Error getting categories by word count range: {str(e)}")
            return []
            
    def get_words_in_category(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get all words in a category.
        
        Args:
            category_id: The ID of the category
            
        Returns:
            List of word dictionaries
        """
        try:
            result = self.db_manager.execute_query("""
                SELECT 
                    id,
                    word,
                    category_id,
                    frequency,
                    allowed,
                    created_at,
                    updated_at
                FROM words
                WHERE category_id = ?
                ORDER BY word
            """, (category_id,))
            
            return [{
                'id': row['id'],
                'word': row['word'],
                'category_id': row['category_id'],
                'frequency': row['frequency'],
                'allowed': bool(row['allowed']),
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            } for row in result]
        except Exception as e:
            logger.error(f"Error getting words in category: {str(e)}")
            return []
            
    def create_category(self, name: str, description: str = "") -> Optional[Dict[str, Any]]:
        """
        Create a new category.
        
        Args:
            name: Name of the category
            description: Description of the category
            
        Returns:
            Created category record or None if failed
        """
        try:
            data = {
                'name': name,
                'description': description,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            category_id = self.create(data)
            if category_id:
                return self.get_by_id(category_id)
            return None
        except Exception as e:
            logger.error(f"Error creating category {name}: {str(e)}")
            return None
            
    def update_category(self, category_id: int, name: str = None, description: str = None) -> Optional[Dict[str, Any]]:
        """
        Update a category.
        
        Args:
            category_id: ID of the category to update
            name: New name (optional)
            description: New description (optional)
            
        Returns:
            Updated category record or None if failed
        """
        try:
            data = {}
            if name is not None:
                data['name'] = name
            if description is not None:
                data['description'] = description
            data['updated_at'] = datetime.now()
            
            if data:
                self.update(category_id, data)
                return self.get_by_id(category_id)
            return None
        except Exception as e:
            logger.error(f"Error updating category {category_id}: {str(e)}")
            return None
            
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories.
        
        Returns:
            List of dictionaries containing category data:
                - id: Category ID
                - name: Category name
                - description: Category description
                - created_at: Creation timestamp
                - updated_at: Last update timestamp
                - word_count: Number of words in the category
        """
        try:
            return self.db_manager.execute_query("""
                SELECT 
                    c.*,
                    COUNT(w.id) as word_count
                FROM categories c
                LEFT JOIN words w ON c.id = w.category_id
                GROUP BY c.id, c.name, c.description, c.created_at, c.updated_at
                ORDER BY c.name
            """)
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []

    def get_categories_by_frequency_range(self, min_freq: int, max_freq: int) -> List[Dict[str, Any]]:
        """
        Get categories with words in a frequency range.
        
        Args:
            min_freq: Minimum word frequency
            max_freq: Maximum word frequency
            
        Returns:
            List of categories with words in the frequency range
        """
        try:
            return self.db_manager.execute_query("""
                SELECT DISTINCT c.*
                FROM categories c
                JOIN words w ON c.id = w.category_id
                WHERE w.frequency BETWEEN ? AND ?
                ORDER BY c.name
            """, (min_freq, max_freq))
        except Exception as e:
            logger.error(f"Error getting categories by frequency range: {str(e)}")
            return []

    def get_categories_by_allowed_status(self, allowed: bool) -> List[Dict]:
        """Get categories where all words have the specified allowed status.

        Args:
            allowed (bool): True to get categories with all allowed words,
                           False to get categories with all disallowed words.

        Returns:
            List[Dict]: List of category dictionaries.
        """
        try:
            query = """
                SELECT DISTINCT c.*
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN w.allowed = ? THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND w2.allowed != ?
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.allowed IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w4
                    WHERE w4.category_id = c.id
                    AND w4.allowed = ?
                )
            """
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (1 if allowed else 0, 1 if allowed else 0, 1 if not allowed else 0))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting categories by allowed status: {e}")
            return []

    def get_categories_by_search_term(self, term: str) -> List[Dict[str, Any]]:
        """
        Search categories by name or description.
        
        Args:
            term: Search term
            
        Returns:
            List of matching categories
        """
        try:
            search_term = f"%{term}%"
            return self.db_manager.execute_query("""
                SELECT *
                FROM categories
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY name
            """, (search_term, search_term))
        except Exception as e:
            logger.error(f"Error searching categories: {str(e)}")
            return []

    def get_categories_by_word_length(self, min_length: int, max_length: int) -> List[Dict[str, Any]]:
        """
        Get categories with words of specific lengths.
        
        Args:
            min_length: Minimum word length
            max_length: Maximum word length
            
        Returns:
            List of categories with words in the length range
        """
        try:
            return self.db_manager.execute_query("""
                SELECT DISTINCT c.*
                FROM categories c
                JOIN words w ON c.id = w.category_id
                WHERE LENGTH(w.word) BETWEEN ? AND ?
                ORDER BY c.name
            """, (min_length, max_length))
        except Exception as e:
            logger.error(f"Error getting categories by word length: {str(e)}")
            return []

    def get_categories_by_word_contains(self, substring: str) -> List[Dict[str, Any]]:
        """
        Get categories with words containing a substring.
        
        Args:
            substring: Substring to search for
            
        Returns:
            List of categories with matching words
        """
        try:
            search_term = f"%{substring}%"
            return self.db_manager.execute_query("""
                SELECT DISTINCT c.*
                FROM categories c
                JOIN words w ON c.id = w.category_id
                WHERE w.word LIKE ?
                ORDER BY c.name
            """, (search_term,))
        except Exception as e:
            logger.error(f"Error getting categories by word contains: {str(e)}")
            return []

    def get_categories_by_word_regex(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Get categories with words matching a regex pattern.
        
        Args:
            pattern: Regex pattern to match
            
        Returns:
            List of categories with matching words
        """
        try:
            # SQLite doesn't have built-in regex support, so we'll use LIKE with wildcards
            # This is a simplified version that only handles basic patterns
            if pattern.startswith('^'):
                pattern = pattern[1:] + '%'
            elif pattern.endswith('$'):
                pattern = '%' + pattern[:-1]
            else:
                pattern = '%' + pattern + '%'
                
            return self.db_manager.execute_query("""
                SELECT DISTINCT c.*
                FROM categories c
                JOIN words w ON c.id = w.category_id
                WHERE w.word LIKE ?
                ORDER BY c.name
            """, (pattern,))
        except Exception as e:
            logger.error(f"Error getting categories by word regex: {str(e)}")
            return []

    def get_categories_by_word_palindrome(self, is_palindrome: bool) -> List[Dict]:
        """Get categories where all words are palindromes or non-palindromes.

        Args:
            is_palindrome (bool): True to get categories with all palindrome words,
                                False to get categories with all non-palindrome words.

        Returns:
            List[Dict]: List of category dictionaries.
        """
        try:
            query = """
                SELECT DISTINCT c.*
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN is_palindrome(w.word) = ? THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND is_palindrome(w2.word) != ?
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.word IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w4
                    WHERE w4.category_id = c.id
                    AND is_palindrome(w4.word) = ?
                )
            """
            with self.db_manager.get_connection() as conn:
                self._register_custom_functions(conn)
                cursor = conn.cursor()
                cursor.execute(query, (1 if is_palindrome else 0, 1 if is_palindrome else 0, 1 if not is_palindrome else 0))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting categories by palindrome status: {e}")
            return []

    def get_categories_by_word_scrabble_score(self, min_score: int, max_score: int) -> List[Dict]:
        """Get categories where all words have Scrabble scores within the specified range.

        Args:
            min_score (int): Minimum Scrabble score (inclusive).
            max_score (int): Maximum Scrabble score (inclusive).

        Returns:
            List[Dict]: List of category dictionaries.
        """
        try:
            query = """
                SELECT DISTINCT c.*
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN scrabble_score(w.word) BETWEEN ? AND ? THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND (scrabble_score(w2.word) < ? OR scrabble_score(w2.word) > ?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.word IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w4
                    WHERE w4.category_id = c.id
                    AND scrabble_score(w4.word) NOT BETWEEN ? AND ?
                )
            """
            with self.db_manager.get_connection() as conn:
                self._register_custom_functions(conn)
                cursor = conn.cursor()
                cursor.execute(query, (min_score, max_score, min_score, max_score, min_score, max_score))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting categories by word Scrabble score: {e}")
            return []

    def get_categories_by_word_vowel_count(self, min_count: int, max_count: int) -> List[Dict]:
        """Get categories where all words have vowel counts within the specified range.

        Args:
            min_count (int): Minimum vowel count (inclusive).
            max_count (int): Maximum vowel count (inclusive).

        Returns:
            List[Dict]: List of category dictionaries.
        """
        try:
            query = """
                SELECT DISTINCT c.*
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN vowel_count(w.word) BETWEEN ? AND ? THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND (vowel_count(w2.word) < ? OR vowel_count(w2.word) > ?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.word IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w4
                    WHERE w4.category_id = c.id
                    AND vowel_count(w4.word) NOT BETWEEN ? AND ?
                )
            """
            with self.db_manager.get_connection() as conn:
                self._register_custom_functions(conn)
                cursor = conn.cursor()
                cursor.execute(query, (min_count, max_count, min_count, max_count, min_count, max_count))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting categories by word vowel count: {e}")
            return []

    def get_categories_by_word_consonant_count(self, min_count: int, max_count: int) -> List[Dict]:
        """Get categories where all words have consonant counts within the specified range.

        Args:
            min_count (int): Minimum consonant count (inclusive).
            max_count (int): Maximum consonant count (inclusive).

        Returns:
            List[Dict]: List of category dictionaries.
        """
        try:
            query = """
                SELECT DISTINCT c.*
                FROM categories c
                INNER JOIN words w ON w.category_id = c.id
                GROUP BY c.id
                HAVING COUNT(*) = SUM(CASE WHEN consonant_count(w.word) BETWEEN ? AND ? THEN 1 ELSE 0 END)
                AND COUNT(*) > 0
                AND NOT EXISTS (
                    SELECT 1 FROM words w2
                    WHERE w2.category_id = c.id
                    AND (consonant_count(w2.word) < ? OR consonant_count(w2.word) > ?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w3
                    WHERE w3.category_id = c.id
                    AND w3.word IS NULL
                )
                AND NOT EXISTS (
                    SELECT 1 FROM words w4
                    WHERE w4.category_id = c.id
                    AND consonant_count(w4.word) NOT BETWEEN ? AND ?
                )
            """
            with self.db_manager.get_connection() as conn:
                self._register_custom_functions(conn)
                cursor = conn.cursor()
                cursor.execute(query, (min_count, max_count, min_count, max_count, min_count, max_count))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting categories by word consonant count: {e}")
            return []