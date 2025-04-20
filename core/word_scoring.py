# core/word_scoring.py
# Scores words based on rarity and length, with progressive penalty for repeated use.

from typing import Dict, List, Optional
import logging
from collections import Counter
from core.letter_pool import WEIGHTED_ALPHABET
from core.validation.word_validator import WordValidator
from database.repositories.word_repository import WordRepository
from database.manager import DatabaseManager

logger = logging.getLogger(__name__)

# Build a basic frequency map from the weighted alphabet
_letter_frequencies = Counter(WEIGHTED_ALPHABET)
max_freq = max(_letter_frequencies.values())  # Highest possible frequency
letter_score_map = {
    letter: max(1, (max_freq - count + 1))  # Inverse frequency scoring
    for letter, count in _letter_frequencies.items()
}

class WordScorer:
    """Handles word scoring and statistics."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the word scorer."""
        self.db_manager = db_manager
        self.word_repo = WordRepository(self.db_manager)
        
    def score_word(self, word: str) -> Dict[str, any]:
        """Score a word based on various factors.
        
        Args:
            word: The word to score
            
        Returns:
            Dictionary containing score and statistics
        """
        try:
            # Get word usage stats
            word_stats = self.word_repo.get_by_word(word)
            if not word_stats:
                return {
                    'score': 0,
                    'num_played': 0,
                    'is_valid': False
                }
                
            # Calculate score based on word length and usage
            base_score = len(word)
            usage_penalty = min(word_stats['num_played'] * 0.1, 2.0)  # Max 2 point penalty
            score = max(base_score - usage_penalty, 1)
            
            return {
                'score': score,
                'num_played': word_stats['num_played'],
                'is_valid': word_stats['allowed']
            }
        except Exception as e:
            logger.error(f"Error scoring word {word}: {str(e)}")
            return {
                'score': 0,
                'num_played': 0,
                'is_valid': False
            }

def score_word(word: str, word_validator: WordValidator, category: Optional[str] = None) -> int:
    """
    Calculate the score for a word based on various factors.
    
    Args:
        word: The word to score
        word_validator: Validator instance to check word validity
        category: Optional category to check against (currently unused)
        
    Returns:
        Score for the word, or 0 if invalid
    """
    if not word_validator.validate_word(word):
        return 0
        
    # Base score is the length of the word
    score = len(word)
    
    # Bonus points for using less common letters
    letter_scores = {
        'e': 1, 'a': 1, 'i': 1, 'o': 1, 'n': 1, 'r': 1, 't': 1, 'l': 1, 's': 1,
        'd': 2, 'g': 2, 'b': 3, 'c': 3, 'm': 3, 'p': 3,
        'f': 4, 'h': 4, 'v': 4, 'w': 4, 'y': 4,
        'k': 5,
        'j': 8, 'x': 8,
        'q': 10, 'z': 10
    }
    
    for letter in word.lower():
        if letter in letter_scores:
            score += letter_scores[letter]
            
    # Bonus for word length
    if len(word) >= 7:
        score *= 2
    elif len(word) >= 5:
        score = int(score * 1.5)
        
    return score

def get_word_stats(word: str, word_repo: WordRepository) -> Dict:
    """
    Get statistics for a word.
    
    Args:
        word: The word to get stats for
        word_repo: Repository for word usage data
        
    Returns:
        Dictionary containing word statistics
    """
    return word_repo.get_word_stats(word)