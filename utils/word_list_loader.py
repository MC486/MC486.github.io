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
    Combines words from both wordfreq and NLTK for comprehensive coverage.
    Validates words using WordValidator before including them.
    """
    global _cached_word_set
    if _cached_word_set is not None:
        return _cached_word_set

    # Initialize word validator
    word_validator = WordValidator(use_nltk=True)
    freq_words = set()

    # First try to load from wordfreq
    try:
        # Use top 50,000 common words for playability
        raw_words = top_n_list('en', 50000, wordlist='best')
        # Filter words to only include valid lengths and remove special characters
        freq_words = {word.upper() for word in raw_words 
                     if word.isalpha() and 1 <= len(word) <= 15}
        logger.info(f"Loaded {len(freq_words)} valid words from wordfreq.")
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
        logger.info(f"Loaded {len(nltk_words)} valid words from nltk corpus.")
        
        # Combine both sets
        freq_words.update(nltk_words)
        logger.info(f"Combined word list contains {len(freq_words)} unique words.")
    except Exception as err:
        logger.error(f"Unable to load NLTK word list: {err}")

    # Validate all words using the validator
    validated_words = {word for word in freq_words if word_validator.validate_word(word)}
    logger.info(f"Final validated word list contains {len(validated_words)} words.")

    _cached_word_set = validated_words
    return validated_words