# utils/word_list_loader.py
# Utility to load a master word list for validation and AI logic.

import nltk
from wordfreq import top_n_list
import logging
from core.validation.word_validator import WordValidator

logger = logging.getLogger(__name__)

# Cache so we only load once
_cached_word_set = None

def load_word_list():
    """
    Loads and returns a set of valid English words.
    Uses top frequency words and fallbacks from nltk if needed.
    Validates words using WordValidator before including them.
    """
    global _cached_word_set
    if _cached_word_set is not None:
        return _cached_word_set # Return the cached word list if it's already loaded.

    # Initialize word validator
    word_validator = WordValidator(use_nltk=True)
    freq_words = set()

    try:
        # Use top 50,000 common words for playability
        raw_words = top_n_list('en', 50000, wordlist='best') # Load the top 50,000 English words from wordfreq.
        # Validate each word
        freq_words = {word for word in raw_words if word_validator.validate_word(word)}
        logger.info(f"Loaded {len(freq_words)} valid words from wordfreq.")
    except Exception as e:
        logger.warning(f"wordfreq failed, using nltk fallback: {e}") # Log a warning if wordfreq fails.
        try:
            nltk.download("words", quiet=True) # Download the nltk words corpus (if not already downloaded).
            from nltk.corpus import words
            raw_words = words.words() # Load the words from the nltk corpus.
            # Validate each word
            freq_words = {word for word in raw_words if word_validator.validate_word(word)}
            logger.info(f"Loaded {len(freq_words)} valid words from nltk corpus.")
        except Exception as err:
            logger.error(f"Unable to load fallback word list: {err}") # Log an error if nltk also fails.
            freq_words = set() # Create an empty set if both wordfreq and nltk fail.

    _cached_word_set = freq_words # Cache the loaded word list.
    return freq_words # Return the loaded word list.