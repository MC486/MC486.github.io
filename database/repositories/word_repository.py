from typing import Optional, List, Dict, Any
from datetime import datetime
from ..manager import DatabaseManager
from .base_repository import BaseRepository
import logging

logger = logging.getLogger(__name__)

class WordRepository(BaseRepository):
    """Repository for managing word usage data."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the word repository."""
        super().__init__(db_manager, "words")
        
    def record_word_usage(self, word_id: int) -> None:
        """Record word usage and increment frequency."""
        self.db_manager.execute_query("""
            UPDATE words 
            SET frequency = frequency + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (word_id,))
        
    def get_word_frequency(self, word: str) -> int:
        """
        Get the frequency of a word.
        
        Args:
            word: The word to check
            
        Returns:
            The frequency of the word
        """
        result = self.db_manager.execute_query("""
            SELECT frequency
            FROM words
            WHERE word = ?
        """, (word,))
        return result[0]['frequency'] if result else 0
        
    def get_player_stats(self) -> Dict[str, Any]:
        """
        Get word usage statistics globally.
        
        Returns:
            Dictionary containing:
                - total_words: Total number of words
                - unique_words: Number of unique words
                - most_used: Most frequently used word
                - average_frequency: Average word frequency
        """
        result = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_words,
                COUNT(DISTINCT word) as unique_words,
                MAX(frequency) as max_frequency,
                AVG(frequency) as avg_frequency,
                (SELECT word FROM words 
                 ORDER BY frequency DESC LIMIT 1) as most_used
            FROM words
        """)
        
        return result[0] if result else {
            'total_words': 0,
            'unique_words': 0,
            'max_frequency': 0,
            'avg_frequency': 0,
            'most_used': None
        }
        
    def get_entry_count(self) -> int:
        """
        Get the total number of words.
        
        Returns:
            The number of words
        """
        query = "SELECT COUNT(*) FROM words"
        return self.db_manager.get_scalar(query) or 0

    def add_word(self, word: str, category_id: int) -> int:
        """
        Add a word to a category.
        
        Args:
            word: The word to add
            category_id: ID of the category
            
        Returns:
            ID of the created word
        """
        data = {
            'word': word,
            'category_id': category_id,
            'frequency': 0,
            'allowed': True
        }
        return self.create(data)
        
    def get_words_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get all words in a category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            List of word records
        """
        return self.db_manager.execute_query("""
            SELECT *
            FROM words
            WHERE category_id = ?
            ORDER BY word
        """, (category_id,))
        
    def get_word_count_by_category(self, category_id: int) -> int:
        """
        Get the number of words in a category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            Number of words in the category
        """
        result = self.db_manager.execute_query("""
            SELECT COUNT(*) as count
            FROM words
            WHERE category_id = ?
        """, (category_id,))
        return result[0]['count'] if result else 0
        
    def get_word_stats_by_category(self, category_id: int) -> Dict[str, Any]:
        """
        Get word statistics for a category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            Dictionary containing word statistics
        """
        result = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_words,
                MIN(LENGTH(word)) as min_word_length,
                MAX(LENGTH(word)) as max_word_length,
                AVG(LENGTH(word)) as avg_word_length,
                SUM(frequency) as total_usage,
                AVG(frequency) as avg_usage
            FROM words
            WHERE category_id = ?
        """, (category_id,))
        return result[0] if result else {}
        
    def get_word_stats(self) -> dict:
        """Get word statistics."""
        result = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_words,
                SUM(frequency) as total_usage,
                AVG(frequency) as avg_usage,
                MAX(frequency) as max_usage,
                COUNT(CASE WHEN allowed = 1 THEN 1 END) as allowed_words,
                MIN(LENGTH(word)) as min_word_length,
                MAX(LENGTH(word)) as max_word_length,
                AVG(LENGTH(word)) as avg_word_length
            FROM words
        """)
        return result[0] if result else None
        
    def get_by_word(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Get a word's record by its exact spelling.
        
        Args:
            word: The word to find
            
        Returns:
            The word record if found, None otherwise
        """
        try:
            result = self.db_manager.execute_query(
                "SELECT * FROM words WHERE word = ?",
                (word,)
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting word {word}: {str(e)}")
            return None
        
    def get_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Get all words in a category.
        
        Args:
            category_id: The category ID
            
        Returns:
            List of word records in the category
        """
        return self.find({'category_id': category_id})
        
    def get_by_frequency_range(self, min_freq: int, max_freq: int) -> List[Dict[str, Any]]:
        """
        Get words within a frequency range.
        
        Args:
            min_freq: Minimum frequency
            max_freq: Maximum frequency
            
        Returns:
            List of word records in the frequency range
        """
        query = """
            SELECT * FROM words
            WHERE frequency >= ? AND frequency <= ?
            ORDER BY frequency DESC
        """
        return self.db_manager.execute_query(query, (min_freq, max_freq))
        
    def get_top_words(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequent words.
        
        Args:
            limit: Maximum number of words to return
            
        Returns:
            List of the most frequent word records
        """
        query = """
            SELECT * FROM words
            ORDER BY frequency DESC
            LIMIT ?
        """
        return self.db_manager.execute_query(query, (limit,))
        
    def increment_frequency(self, word_id: int) -> None:
        """Increment word frequency."""
        self.db_manager.execute_query("""
            UPDATE words 
            SET frequency = frequency + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (word_id,))
        
    def search_words(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for words matching a pattern."""
        return self.db_manager.execute_query("""
            SELECT * FROM words
            WHERE word LIKE ?
            ORDER BY frequency DESC
        """, (pattern,))
        
    def get_all_words(self) -> List[Dict[str, Any]]:
        """Get all words."""
        return self.db_manager.execute_query("""
            SELECT * FROM words
            ORDER BY word
        """)
        
    def get_words_by_length(self, length: int) -> List[Dict[str, Any]]:
        """
        Get words of a specific length.
        
        Args:
            length: Word length
            
        Returns:
            List of word records
        """
        return self.db_manager.execute_query("""
            SELECT * FROM words
            WHERE LENGTH(word) = ?
            ORDER BY frequency DESC
        """, (length,))
        
    def bulk_update_frequency(self, frequency_updates: Dict[int, int]) -> None:
        """
        Update frequencies for multiple words.
        
        Args:
            frequency_updates: Dictionary of word_id to frequency
        """
        for word_id, frequency in frequency_updates.items():
            self.db_manager.execute_query("""
                UPDATE words 
                SET frequency = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (frequency, word_id))
        
    def get_words_without_category(self) -> List[Dict[str, Any]]:
        """Get words without a category."""
        return self.db_manager.execute_query("""
            SELECT * FROM words
            WHERE category_id IS NULL
            ORDER BY word
        """)
        
    def get_valid_words(self) -> List[Dict[str, Any]]:
        """Get all valid words."""
        return self.db_manager.execute_query("""
            SELECT * FROM words
            WHERE allowed = 1
            ORDER BY word
        """)
        
    def get_rare_words(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the least frequent words.
        
        Args:
            limit: Maximum number of words to return
            
        Returns:
            List of the least frequent word records
        """
        return self.db_manager.execute_query("""
            SELECT * FROM words
            ORDER BY frequency ASC
            LIMIT ?
        """, (limit,))
        
    def get_word_by_pattern(self, pattern: str) -> Optional[Dict[str, Any]]:
        """
        Get a word matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Word record if found, None otherwise
        """
        result = self.db_manager.execute_query("""
            SELECT * FROM words
            WHERE word LIKE ? AND allowed = 1
            ORDER BY frequency DESC
            LIMIT 1
        """, (pattern,))
        return result[0] if result else None

    def get_word_usage(self) -> List[Dict[str, Any]]:
        """
        Get word usage data for all words.
        
        Returns:
            List of dictionaries containing word usage data:
                - word_id: ID of the word
                - word: The word itself
                - frequency: Number of times the word has been used
                - last_used: Timestamp of last usage
                - avg_score: Average score when used
        """
        return self.db_manager.execute_query("""
            SELECT 
                w.id as word_id,
                w.word,
                w.frequency,
                MAX(wu.used_at) as last_used,
                AVG(wu.score) as avg_score
            FROM words w
            LEFT JOIN word_usage wu ON w.id = wu.word_id
            GROUP BY w.id, w.word, w.frequency
            ORDER BY w.frequency DESC, w.word
        """)
