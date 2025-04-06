from typing import Dict, List, Optional
from datetime import datetime
from ..manager import DatabaseManager

class NaiveBayesRepository:
    """Repository for managing Naive Bayes model data."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def record_word_probability(self, word: str, probability: float, 
                              pattern_type: Optional[str] = None) -> None:
        """
        Record a word probability and its pattern type.
        
        Args:
            word: The word
            probability: Probability value
            pattern_type: Optional pattern type (e.g., 'prefix', 'suffix')
        """
        self.db.execute_query("""
            INSERT INTO naive_bayes_words (word, probability, pattern_type, visit_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(word, pattern_type) DO UPDATE SET
                probability = ?,
                visit_count = visit_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (word, probability, pattern_type, probability))
        
    def get_word_probability(self, word: str, pattern_type: Optional[str] = None) -> float:
        """
        Get the probability for a word and pattern type.
        
        Args:
            word: The word
            pattern_type: Optional pattern type
            
        Returns:
            float: Probability value
        """
        result = self.db.execute_query("""
            SELECT probability FROM naive_bayes_words
            WHERE word = ? AND (pattern_type = ? OR pattern_type IS NULL)
        """, (word, pattern_type))
        
        return result[0]['probability'] if result else 0.0
        
    def get_pattern_probabilities(self, pattern_type: str) -> Dict[str, float]:
        """
        Get probabilities for all words with a specific pattern type.
        
        Args:
            pattern_type: Pattern type to filter by
            
        Returns:
            Dict[str, float]: Dictionary of word probabilities
        """
        results = self.db.execute_query("""
            SELECT word, probability
            FROM naive_bayes_words
            WHERE pattern_type = ?
        """, (pattern_type,))
        
        return {row['word']: row['probability'] for row in results}
        
    def get_total_observations(self) -> int:
        """
        Get total number of observations.
        
        Returns:
            int: Total observations
        """
        result = self.db.execute_query("""
            SELECT SUM(visit_count) as total
            FROM naive_bayes_words
        """)
        
        return result[0]['total'] if result else 0
        
    def get_word_stats(self, word: str) -> Dict:
        """
        Get statistics for a word.
        
        Args:
            word: The word
            
        Returns:
            Dict containing:
                - total_probability: Combined probability
                - pattern_probabilities: Dict of pattern probabilities
                - visit_count: Number of times seen
        """
        results = self.db.execute_query("""
            SELECT probability, pattern_type, visit_count
            FROM naive_bayes_words
            WHERE word = ?
        """, (word,))
        
        if not results:
            return {
                'total_probability': 0.0,
                'pattern_probabilities': {},
                'visit_count': 0
            }
            
        pattern_probs = {}
        total_prob = 0.0
        total_visits = 0
        
        for row in results:
            if row['pattern_type']:
                pattern_probs[row['pattern_type']] = row['probability']
            else:
                total_prob = row['probability']
            total_visits += row['visit_count']
            
        return {
            'total_probability': total_prob,
            'pattern_probabilities': pattern_probs,
            'visit_count': total_visits
        }
        
    def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Remove entries that haven't been updated in the specified number of days.
        
        Args:
            days: Number of days after which to remove entries
            
        Returns:
            int: Number of entries removed
        """
        result = self.db.execute_query("""
            DELETE FROM naive_bayes_words
            WHERE updated_at < datetime('now', ?)
            SELECT changes()
        """, (f"-{days} days",))
        
        return result[0]['changes()'] if result else 0
        
    def get_learning_stats(self) -> Dict:
        """
        Get overall statistics about the Naive Bayes model.
        
        Returns:
            Dict containing:
                - total_words: Total unique words
                - total_patterns: Total pattern types
                - average_probability: Average probability
                - most_common_pattern: Most frequent pattern type
        """
        result = self.db.execute_query("""
            WITH pattern_stats AS (
                SELECT 
                    pattern_type,
                    COUNT(*) as count
                FROM naive_bayes_words
                WHERE pattern_type IS NOT NULL
                GROUP BY pattern_type
            )
            SELECT 
                COUNT(DISTINCT word) as total_words,
                COUNT(DISTINCT pattern_type) as total_patterns,
                AVG(probability) as average_probability,
                (SELECT pattern_type FROM pattern_stats ORDER BY count DESC LIMIT 1) as most_common_pattern
            FROM naive_bayes_words
        """)
        
        return result[0] if result else {
            'total_words': 0,
            'total_patterns': 0,
            'average_probability': 0.0,
            'most_common_pattern': None
        } 