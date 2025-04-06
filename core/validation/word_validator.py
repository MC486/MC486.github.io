from typing import Set, List, Optional
import nltk
from .trie import Trie
from .trie_utils import TrieUtils
import os
import logging

logger = logging.getLogger(__name__)

class WordValidator:
    """Validates words using a Trie-based dictionary with NLTK words."""
    
    def __init__(self, use_nltk: bool = True, custom_dictionary_path: Optional[str] = None):
        """Initialize the WordValidator.
        
        Args:
            use_nltk: Whether to use NLTK word list (default: True)
            custom_dictionary_path: Optional path to custom dictionary file
        """
        self.trie = Trie()
        self.cache_path = None
        
        # Load custom dictionary first
        try:
            custom_path = custom_dictionary_path or "data/custom_dictionary.txt"
            if os.path.exists(custom_path):
                self.load_custom_dictionary(custom_path)
        except Exception as e:
            logger.warning(f"Failed to load custom dictionary: {e}")
        
        if use_nltk:
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words')
            
            from nltk.corpus import words
            word_list = words.words()
            # Filter words to only include valid lengths and remove special characters
            valid_words = {word.upper() for word in word_list 
                         if word.isalpha() and 1 <= len(word) <= 15}
            
            # Add common plural forms
            valid_words.update(self._get_common_plurals(valid_words))
            
            self.trie = TrieUtils.build_trie_from_words(valid_words)

    def _get_common_plurals(self, base_words: Set[str]) -> Set[str]:
        """Generate common plural forms from base words.
        
        Args:
            base_words: Set of base words to generate plurals from
            
        Returns:
            Set of plural forms
        """
        plurals = set()
        for word in base_words:
            # Add 's' plural
            if not word.endswith('S'):
                plurals.add(word + 'S')
            # Add 'es' plural for words ending in s, x, z, ch, sh
            if word.endswith(('S', 'X', 'Z', 'CH', 'SH')):
                plurals.add(word + 'ES')
            # Add 'ies' plural for words ending in y
            if word.endswith('Y'):
                plurals.add(word[:-1] + 'IES')
        return plurals

    def load_custom_dictionary(self, file_path: str) -> None:
        """Load additional words from a custom dictionary file.
        
        Args:
            file_path: Path to the dictionary file
        """
        custom_words = TrieUtils.load_word_list(file_path)
        for word in custom_words:
            self.trie.insert(word)
        
        self.cache_path = file_path + ".cache"
        
        # Try to save cached version
        try:
            TrieUtils.save_trie(self.trie, self.cache_path)
        except:
            pass  # Ignore cache saving errors

    def is_valid_word(self, word: str) -> bool:
        """Check if a word is valid.
        
        Args:
            word: Word to validate
            
        Returns:
            bool: True if word is in dictionary
        """
        if not word:
            return False
        return self.trie.search(word.upper())

    def validate_word(self, word: str) -> bool:
        """Alias for is_valid_word for compatibility.
        
        Args:
            word: Word to validate
            
        Returns:
            bool: True if word is in dictionary
        """
        return self.is_valid_word(word)

    def get_valid_words(self, letters: List[str], min_length: int = 3) -> Set[str]:
        """Find all valid words that can be made from given letters.
        
        Args:
            letters: List of available letters
            min_length: Minimum word length
            
        Returns:
            Set of valid words
        """
        if not letters:
            return set()
            
        letters = [l.upper() for l in letters]
        valid_words = set()
        letter_counts = {}
        
        # Count available letters
        for letter in letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1

        def can_make_word(word: str) -> bool:
            """Check if word can be made from available letters."""
            word_counts = {}
            for char in word:
                word_counts[char] = word_counts.get(char, 0) + 1
                if word_counts[char] > letter_counts.get(char, 0):
                    return False
            return True

        # Get all words with valid prefixes
        for i in range(len(letters)):
            words = self.trie.get_words_with_prefix(letters[i])
            for word in words:
                if len(word) >= min_length and can_make_word(word):
                    valid_words.add(word)

        return valid_words

    def validate_word_with_letters(self, word: str, letters: List[str]) -> bool:
        """Check if word is valid and can be made from given letters.
        
        Args:
            word: Word to validate
            letters: List of available letters
            
        Returns:
            bool: True if word is valid and can be made from letters
        """
        if not word or not letters:
            return False
            
        word = word.upper()
        letters = [l.upper() for l in letters]
        
        if not self.is_valid_word(word):
            return False
            
        # Check if word can be made from letters
        letter_counts = {}
        for letter in letters:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            
        for char in word:
            if char not in letter_counts or letter_counts[char] == 0:
                return False
            letter_counts[char] -= 1
            
        return True

    def get_word_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get word suggestions starting with prefix.
        
        Args:
            prefix: Starting characters
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested words
        """
        if not prefix:
            return []
            
        return self.trie.get_words_with_prefix(prefix, max_suggestions)

    def get_dictionary_stats(self) -> dict:
        """Get statistics about the loaded dictionary.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            "total_words": self.trie.total_words,
            "max_word_length": self.trie.max_word_length,
            "memory_usage": TrieUtils.get_memory_usage(self.trie)
        }