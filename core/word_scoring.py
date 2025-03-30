# core/word_scoring.py
# Scores words based on rarity and length, with progressive penalty for repeated use.

import logging
from collections import Counter
from core.letter_pool import WEIGHTED_ALPHABET
from core.validation.word_validator import WordValidator

logger = logging.getLogger(__name__)

# Build a basic frequency map from the weighted alphabet
_letter_frequencies = Counter(WEIGHTED_ALPHABET)
max_freq = max(_letter_frequencies.values())  # Highest possible frequency
letter_score_map = {
    letter: max(1, (max_freq - count + 1))  # Inverse frequency scoring
    for letter, count in _letter_frequencies.items()
}

# Initialize word validator
_word_validator = WordValidator(use_nltk=True)

def score_word(word: str, repeat_count: int = 0) -> int:
    """
    Scores a word based on letter rarity and length. Applies a progressive penalty if it's repeated.
    Only scores valid words.

    :param word: The word to score.
    :param repeat_count: How many times this word has already been used.
    :return: Calculated score after fatigue penalty, or 0 if word is invalid.
    """
    word = word.upper()  # Normalize for consistency with letter pool
    
    # Validate word before scoring
    if not _word_validator.validate_word(word):
        logger.warning(f"Attempted to score invalid word: '{word}'")
        return 0
        
    base_score = sum(letter_score_map.get(letter, 1) for letter in word)
    
    # Ensure minimum score of 1 for any valid word
    base_score = max(1, base_score)

    # Apply diminishing returns if word is repeated
    if repeat_count > 0:
        fatigue_factor = 1 / (repeat_count + 1) # Calculate the penalty factor based on repeat count.
        final_score = max(1, int(base_score * fatigue_factor)) # Apply the penalty factor and ensure score is at least 1.
        logger.info(
            f"Word '{word}' scored {final_score} points after repeat fatigue (used {repeat_count + 1} times)."
        ) # Log the score after applying the repeat penalty.
    else:
        final_score = base_score # If the word is not repeated, the final score is the base score.
        logger.info(f"New word '{word}' scored {final_score} points.") # Log the score of a new word.

    return final_score # Return the calculated final score.