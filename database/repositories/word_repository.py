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
        super().__init__(db_manager)
        self.table_name = 'word_usage'
        
        # Create word_usage table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS word_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                word TEXT NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
    def record_word_usage(self, word: str, name: str) -> None:
        """
        Record a word usage.
        
        Args:
            word: The word used
            name: The name of the player
        """
        self.db.execute_query("""
            INSERT INTO word_usage (word, name, frequency)
            VALUES (?, ?, 1)
            ON CONFLICT(word, name) DO UPDATE SET
                frequency = frequency + 1,
                last_used = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
        """)
        
    def get_word_frequency(self, word: str, name: str) -> int:
        """
        Get the frequency of a word for a player.
        
        Args:
            word: The word to check
            name: The name of the player
            
        Returns:
            The frequency of the word
        """
        result = self.db.execute_query("""
            SELECT frequency
            FROM word_usage
            WHERE word = ? AND name = ?
        """, (word, name))
        return result[0]['frequency'] if result else 0
        
    def get_player_stats(self, name: str) -> Dict[str, Any]:
        """
        Get word usage statistics for a player.
        
        Args:
            name: The name of the player
            
        Returns:
            Dictionary containing:
                - total_words: Total number of words used
                - unique_words: Number of unique words used
                - most_used: Most frequently used word
                - average_frequency: Average word frequency
        """
        result = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_words,
                COUNT(DISTINCT word) as unique_words,
                MAX(frequency) as max_frequency,
                AVG(frequency) as avg_frequency,
                (SELECT word FROM word_usage w2 
                 WHERE w2.name = w1.name 
                 ORDER BY frequency DESC LIMIT 1) as most_used
            FROM word_usage w1
            WHERE name = ?
            GROUP BY name
        """, (name,))
        
        return result[0] if result else {
            'total_words': 0,
            'unique_words': 0,
            'max_frequency': 0,
            'avg_frequency': 0,
            'most_used': None
        }
        
    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the word_usage table.
        
        Returns:
            The number of entries
        """
        query = "SELECT COUNT(*) FROM word_usage"
        return self.db.get_scalar(query) or 0

    def create_table(self) -> None:
        """Create the word_usage table if it doesn't exist."""
        try:
            # Drop the existing table if it exists
            self.db.execute("DROP TABLE IF EXISTS word_usage")
            
            # Create the table with the updated schema
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS word_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL UNIQUE,
                    num_played INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_word_usage_word ON word_usage(word)
            """)
            
            # Create trigger for updating the updated_at timestamp
            self.db.execute("""
                CREATE TRIGGER IF NOT EXISTS update_word_usage_timestamp
                AFTER UPDATE ON word_usage
                BEGIN
                    UPDATE word_usage SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = NEW.id;
                END
            """)
            
            logger.debug("Word usage table created successfully")
        except Exception as e:
            logger.error(f"Error creating word usage table: {str(e)}")
            raise
            
    def add_word(self, word: str) -> None:
        """
        Add a word to the repository or increment its usage count if it exists.
        
        Args:
            word: The word to add/update
        """
        try:
            # Try to increment usage count if word exists
            result = self.db.execute("""
                UPDATE word_usage 
                SET num_played = num_played + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE word = ?
            """, (word,))
            
            # If word doesn't exist, insert it
            if result.rowcount == 0:
                self.db.execute("""
                    INSERT INTO word_usage (word, num_played)
                    VALUES (?, 1)
                """, (word,))
                
        except Exception as e:
            logger.error(f"Error adding/updating word {word}: {str(e)}")
            raise
        
    def get_word_stats(self) -> Dict[str, Any]:
        """
        Get statistics about word usage.
        
        Returns:
            Dictionary containing word usage statistics
        """
        try:
            result = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total_words,
                    SUM(num_played) as total_plays,
                    AVG(num_played) as avg_plays_per_word,
                    MAX(num_played) as max_plays
                FROM word_usage
            """)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Error getting word stats: {str(e)}")
            return {}
        
    def get_by_word(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Get a word's usage record by its exact spelling.
        
        Args:
            word: The word to find
            
        Returns:
            The word record if found, None otherwise
        """
        try:
            result = self.db.execute_query(
                "SELECT * FROM word_usage WHERE word = ?",
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
        return self.db.execute_query(query, (min_freq, max_freq))
        
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
        return self.db.execute_query(query, (limit,))
        
    def increment_frequency(self, word: str) -> bool:
        """
        Increment the frequency count of a word.
        
        Args:
            word: The word to update
            
        Returns:
            True if the word was found and updated
        """
        query = """
            UPDATE words
            SET frequency = frequency + 1
            WHERE word = ?
        """
        self.db.execute(query, (word,))
        return True
        
    def search_words(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for words matching a pattern.
        
        Args:
            pattern: SQL LIKE pattern (e.g., 'test%' for words starting with 'test')
            
        Returns:
            List of matching word records
        """
        query = """
            SELECT * FROM word_usage
            WHERE word LIKE ?
            ORDER BY num_played DESC
        """
        return self.db.execute_query(query, (pattern,))
        
    def get_all_words(self) -> List[Dict[str, Any]]:
        """
        Get all words from the repository.
        
        Returns:
            List of all word records
        """
        try:
            return self.db.execute_query(
                "SELECT * FROM word_usage WHERE allowed = 1"
            )
        except Exception as e:
            logger.error(f"Error getting all words: {str(e)}")
            return []
        
    def get_words_by_length(self, length: int) -> List[Dict[str, Any]]:
        """
        Get words of a specific length.
        
        Args:
            length: The length of words to find
            
        Returns:
            List of word records with the specified length
        """
        query = """
            SELECT * FROM words
            WHERE LENGTH(word) = ?
            ORDER BY frequency DESC
        """
        return self.db.execute_query(query, (length,))
        
    def bulk_update_frequency(self, word_frequencies: Dict[str, int]) -> None:
        """
        Update frequencies for multiple words at once.
        
        Args:
            word_frequencies: Dictionary mapping words to their new frequencies
        """
        if not word_frequencies:
            return
            
        query = """
            UPDATE words
            SET frequency = ?
            WHERE word = ?
        """
        
        params = [(freq, word) for word, freq in word_frequencies.items()]
        self.db.execute_many(query, params)
        
    def get_words_without_category(self) -> List[Dict[str, Any]]:
        """
        Get words that don't have a category assigned.
        
        Returns:
            List of uncategorized word records
        """
        query = """
            SELECT * FROM words
            WHERE category_id IS NULL
            ORDER BY frequency DESC
        """
        return self.db.execute_query(query)
        
    def get_word_usage(self) -> List[Dict[str, Any]]:
        """
        Get usage data for all words.
        
        Returns:
            List of dictionaries containing word usage data
        """
        try:
            return self.db.execute_query("""
                SELECT word, num_played as frequency, created_at, updated_at
                FROM word_usage
                ORDER BY num_played DESC
            """)
        except Exception as e:
            logger.error(f"Error getting word usage data: {str(e)}")
            return []

    def get_all_words(self) -> List[Dict[str, Any]]:
        """
        Get all words in the repository.
        
        Returns:
            List of dictionaries containing word data
        """
        try:
            return self.db.execute_query("""
                SELECT word, num_played as frequency, created_at, updated_at
                FROM word_usage
                ORDER BY word ASC
            """)
        except Exception as e:
            logger.error(f"Error getting all words: {str(e)}")
            return [] 