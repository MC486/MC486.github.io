# utils/word_list_loader.py
# Utility to load a master word list for validation and AI logic.

import nltk
from wordfreq import top_n_list
import logging

logger = logging.getLogger(__name__)

# Cache so we only load once
_cached_word_set = None

def load_word_list():
    """
    Loads and returns a set of valid English words.
    Uses top frequency words and fallbacks from nltk if needed.
    """
    global _cached_word_set
    if _cached_word_set is not None:
        return _cached_word_set # Return the cached word list if it's already loaded.

    try:
        # Use top 50,000 common words for playability
        freq_words = set(top_n_list('en', 50000, wordlist='best')) # Load the top 50,000 English words from wordfreq.
        logger.info(f"Loaded {len(freq_words)} words from wordfreq.") # Log the number of words loaded.
    except Exception as e:
        logger.warning(f"wordfreq failed, using nltk fallback: {e}") # Log a warning if wordfreq fails.
        try:
            nltk.download("words", quiet=True) # Download the nltk words corpus (if not already downloaded).
            from nltk.corpus import words
            freq_words = set(words.words()) # Load the words from the nltk corpus.
            logger.info(f"Loaded {len(freq_words)} words from nltk corpus.") # Log the number of words loaded from nltk.
        except Exception as err:
            logger.error(f"Unable to load fallback word list: {err}") # Log an error if nltk also fails.
            freq_words = set() # Create an empty set if both wordfreq and nltk fail.

    _cached_word_set = freq_words # Cache the loaded word list.
    return freq_words # Return the loaded word list.