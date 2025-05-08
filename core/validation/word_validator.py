from typing import Set, List, Optional
import nltk
from .trie import Trie
from .trie_utils import TrieUtils
import os
import logging
from database.repositories.word_repository import WordRepository
from database.manager import DatabaseManager

logger = logging.getLogger(__name__)

class WordValidator:
    """Validates words using NLTK and tracks word usage in the database."""
    
    def __init__(self, word_repo: WordRepository, use_nltk: bool = True, custom_dictionary_path: Optional[str] = None):
        """Initialize the WordValidator.
        
        Args:
            word_repo: Repository for word usage data
            use_nltk: Whether to use NLTK for word validation
            custom_dictionary_path: Path to a custom dictionary file
        """
        self.word_repo = word_repo
        self.use_nltk = use_nltk
        self.custom_words = set()
        self.nltk_words = set()
        
        if use_nltk:
            # Ensure NLTK words corpus is downloaded
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words', quiet=True)
                
            # Load NLTK words
            try:
                from nltk.corpus import words
                nltk_words = words.words()
                self.nltk_words = {word.upper() for word in nltk_words if 3 <= len(word) <= 15}
                logger.debug(f"Loaded {len(self.nltk_words)} NLTK words")
            except Exception as e:
                logger.error(f"Error loading NLTK words: {e}")
                
        if custom_dictionary_path and os.path.exists(custom_dictionary_path):
            try:
                with open(custom_dictionary_path, 'r') as f:
                    self.custom_words = {line.strip().upper() for line in f if line.strip() and 3 <= len(line.strip()) <= 15}
                logger.debug(f"Loaded {len(self.custom_words)} custom words")
            except Exception as e:
                logger.error(f"Error loading custom dictionary: {e}")
        
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid using NLTK and/or custom dictionary.
        
        Args:
            word: The word to validate
            
        Returns:
            bool: True if the word is valid
        """
        if not word or not isinstance(word, str):
            return False
            
        word = word.upper()
        if not word.isalpha() or not 3 <= len(word) <= 15:
            return False
            
        is_valid = False
        
        if self.use_nltk:
            is_valid = word in self.nltk_words
            logger.debug(f"NLTK validation for {word}: {is_valid} (nltk_words: {len(self.nltk_words)})")
                
        if not is_valid and self.custom_words:
            is_valid = word in self.custom_words
            logger.debug(f"Custom validation for {word}: {is_valid} (custom_words: {len(self.custom_words)})")
            
        # Only record word usage if it's valid
        if is_valid:
            try:
                self.word_repo.add_word(word)
                logger.debug(f"Recorded word usage: {word} (valid: {is_valid})")
            except Exception as e:
                logger.error(f"Error recording word usage for {word}: {e}")
            
        return is_valid

    def get_word_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get word suggestions starting with the given prefix.
        
        Args:
            prefix: The prefix to search for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested words
        """
        if not prefix:
            return []
            
        prefix = prefix.upper()
        suggestions = []
        
        if self.use_nltk:
            for word in self.nltk_words:
                if word.startswith(prefix):
                    suggestions.append(word)
                    if len(suggestions) >= max_suggestions:
                        break
                
        if len(suggestions) < max_suggestions and self.custom_words:
            for word in self.custom_words:
                if word.startswith(prefix):
                    suggestions.append(word)
                    if len(suggestions) >= max_suggestions:
                        break
                        
        return suggestions

    def get_dictionary_stats(self) -> dict:
        """Get statistics about the dictionary."""
        stats = {
            'total_words': 0,
            'max_length': 0,
            'min_length': 0,
            'usage_stats': self.word_repo.get_word_stats()
        }
        
        if self.use_nltk:
            stats['total_words'] += len(self.nltk_words)
            if self.nltk_words:
                stats['max_length'] = max((len(word) for word in self.nltk_words))
                stats['min_length'] = min((len(word) for word in self.nltk_words))
                
        if self.custom_words:
            stats['total_words'] += len(self.custom_words)
            if self.custom_words:
                stats['max_length'] = max(stats['max_length'], max((len(word) for word in self.custom_words)))
                stats['min_length'] = min(stats['min_length'], min((len(word) for word in self.custom_words)))
            
        return stats

    def validate_word_with_letters(self, word: str, available_letters: List[str]) -> bool:
        """Check if a word is valid and can be formed using the available letters.
        
        Args:
            word: The word to validate
            available_letters: List of available letters (may contain duplicates)
            
        Returns:
            bool: True if the word is valid and can be formed using the available letters
        """
        if not word or not isinstance(word, str):
            return False
            
        word = word.upper()
        if not word.isalpha() or not 3 <= len(word) <= 15:
            return False
            
        # First check if the word is valid
        if not self.is_valid_word(word):
            return False
            
        # Then check if word can be formed using available letters
        word_letters = {}
        for letter in word:
            word_letters[letter] = word_letters.get(letter, 0) + 1
            
        available_letters_dict = {}
        for letter in available_letters:
            letter = letter.upper()
            available_letters_dict[letter] = available_letters_dict.get(letter, 0) + 1
            
        for letter, count in word_letters.items():
            if letter not in available_letters_dict or available_letters_dict[letter] < count:
                logger.debug(f"Letter {letter} not available or insufficient count")
                return False
                
        return True

    def validate_word(self, word: str) -> bool:
        """Alias for is_valid_word for backward compatibility.
        
        Args:
            word: The word to validate
            
        Returns:
            bool: True if the word is valid
        """
        return self.is_valid_word(word)