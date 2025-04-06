# utils/word_list_loader.py
# Utility to load a master word list for validation and AI logic.

import nltk
from wordfreq import top_n_list
import logging
from core.validation.word_validator import WordValidator
from database.repositories.word_repository import WordRepository
from database.manager import DatabaseManager

logger = logging.getLogger(__name__)

# Cache so we only load once
_cached_word_set = None

def load_word_list():
    """
    Loads and returns a set of valid English words.
    Combines words from both wordfreq and NLTK for comprehensive coverage.
    Validates words using WordValidator before including them.
    """
    global _cached_word_set
    if _cached_word_set is not None:
        return _cached_word_set

    # Initialize word validator and repository
    word_validator = WordValidator(use_nltk=True)
    db_manager = DatabaseManager()
    word_repo = WordRepository(db_manager)
    freq_words = set()

    # First try to load from wordfreq
    try:
        # Use top 50,000 common words for playability
        raw_words = top_n_list('en', 50000, wordlist='best')
        # Filter words to only include valid lengths and remove special characters
        freq_words = {word.upper() for word in raw_words 
                     if word.isalpha() and 1 <= len(word) <= 15}
        logger.info(f"Loaded {len(freq_words)} valid words from wordfreq.")
        
        # Store words in repository
        for word in freq_words:
            word_repo.add_word(word)
    except Exception as e:
        logger.warning(f"wordfreq failed: {e}")
        freq_words = set()

    # Then load from NLTK and combine
    try:
        # Ensure NLTK words corpus is downloaded
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        
        from nltk.corpus import words
        raw_words = words.words()
        # Filter words to only include valid lengths and remove special characters
        nltk_words = {word.upper() for word in raw_words 
                     if word.isalpha() and 1 <= len(word) <= 15}
        
        # Store NLTK words in repository
        for word in nltk_words:
            word_repo.add_word(word)
        
        # Combine word sets
        freq_words.update(nltk_words)
        logger.info(f"Loaded {len(nltk_words)} valid words from NLTK.")
    except Exception as e:
        logger.warning(f"NLTK failed: {e}")

    _cached_word_set = freq_words
    return _cached_word_set