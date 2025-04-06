from typing import Optional, List, Dict, Any
from ..base_repository import BaseRepository
from ..manager import DatabaseManager

class WordRepository(BaseRepository):
    """Repository for managing words in the dictionary."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the word repository."""
        super().__init__(db_manager, 'words')
        
    def get_by_word(self, word: str) -> Optional[Dict[str, Any]]:
        """
        Get a word by its exact spelling.
        
        Args:
            word: The word to find
            
        Returns:
            The word record if found, None otherwise
        """
        return self.find_one({'word': word})
        
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
        
    def get_word_stats(self) -> Dict[str, Any]:
        """
        Get statistics about words in the dictionary.
        
        Returns:
            Dictionary containing word statistics
        """
        stats = {
            'total_words': self.count(),
            'avg_frequency': self.db.get_scalar("""
                SELECT AVG(frequency) FROM words
            """),
            'max_frequency': self.db.get_scalar("""
                SELECT MAX(frequency) FROM words
            """),
            'min_frequency': self.db.get_scalar("""
                SELECT MIN(frequency) FROM words
            """),
            'words_by_category': self.db.execute_query("""
                SELECT category_id, COUNT(*) as count
                FROM words
                GROUP BY category_id
            """)
        }
        return stats
        
    def search_words(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for words matching a pattern.
        
        Args:
            pattern: SQL LIKE pattern (e.g., 'test%' for words starting with 'test')
            
        Returns:
            List of matching word records
        """
        query = """
            SELECT * FROM words
            WHERE word LIKE ?
            ORDER BY frequency DESC
        """
        return self.db.execute_query(query, (pattern,))
        
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