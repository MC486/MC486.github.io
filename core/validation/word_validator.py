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
    
    def __init__(self, word_repo: WordRepository):
        """Initialize the WordValidator.
        
        Args:
            word_repo: Repository for word usage data
        """
        self.word_repo = word_repo
        # Ensure NLTK words corpus is downloaded
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        
    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid using NLTK.
        
        Args:
            word: The word to validate
            
        Returns:
            bool: True if the word is valid
        """
        word = word.upper()
        if not word.isalpha() or not 1 <= len(word) <= 15:
            return False
            
        try:
            from nltk.corpus import words
            is_valid = word.lower() in words.words()
            
            # Only record word usage if it's valid
            if is_valid:
                try:
                    self.word_repo.add_word(word)
                    logger.debug(f"Recorded word usage: {word} (valid: {is_valid})")
                except Exception as e:
                    logger.error(f"Error recording word usage for {word}: {e}")
            
            return is_valid
        except Exception as e:
            logger.error(f"Error validating word {word}: {e}")
            return False

    def get_word_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get word suggestions starting with the given prefix.
        
        Args:
            prefix: The prefix to search for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested words
        """
        prefix = prefix.upper()
        suggestions = []
        
        try:
            from nltk.corpus import words
            for word in words.words():
                if word.upper().startswith(prefix):
                    suggestions.append(word.upper())
                    if len(suggestions) >= max_suggestions:
                        break
        except Exception as e:
            logger.error(f"Error getting word suggestions: {e}")
            
        return suggestions

    def get_dictionary_stats(self) -> dict:
        """Get statistics about the dictionary."""
        try:
            from nltk.corpus import words
            word_list = words.words()
            return {
                'total_words': len(word_list),
                'max_length': max((len(word) for word in word_list), default=0),
                'min_length': min((len(word) for word in word_list), default=0),
                'usage_stats': self.word_repo.get_word_stats()
            }
        except Exception as e:
            logger.error(f"Error getting dictionary stats: {e}")
            return {
                'total_words': 0,
                'max_length': 0,
                'min_length': 0,
                'usage_stats': {}
            }

    def validate_word_with_letters(self, word: str, available_letters: Set[str]) -> bool:
        """Check if a word is valid and can be formed using the available letters.
        
        Args:
            word: The word to validate
            available_letters: Set of available letters
            
        Returns:
            bool: True if the word is valid and can be formed using the available letters
        """
        word = word.upper()
        if not word.isalpha() or not 1 <= len(word) <= 15:
            return False
            
        # Check if word can be formed using available letters
        word_letters = {}
        for letter in word:
            word_letters[letter] = word_letters.get(letter, 0) + 1
            
        available_letters_dict = {}
        for letter in available_letters:
            available_letters_dict[letter.upper()] = available_letters_dict.get(letter.upper(), 0) + 1
            
        for letter, count in word_letters.items():
            if letter not in available_letters_dict or available_letters_dict[letter] < count:
                return False
                
        # Check if word is valid using NLTK
        try:
            from nltk.corpus import words
            return word.lower() in words.words()
        except Exception as e:
            logger.error(f"Error validating word {word}: {e}")
            return False

    def validate_word(self, word: str) -> bool:
        """Alias for is_valid_word for backward compatibility.
        
        Args:
            word: The word to validate
            
        Returns:
            bool: True if the word is valid
        """
        return self.is_valid_word(word)