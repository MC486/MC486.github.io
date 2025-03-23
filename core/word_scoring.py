# core/word_scoring.py
# Scores words based on length, with progressive penalty for repeated use.

import logging

logger = logging.getLogger(__name__)

def score_word(word: str, repeat_count: int = 0) -> int:
    """
    Scores a word based on its length. Applies a progressive penalty if it's a repeated word.
    
    :param word: The word to score.
    :param repeat_count: How many times this word has already been used.
    :return: Calculated score after fatigue penalty.
    """
    base_score = len(word) # The base score is the length of the word.

    # Apply diminishing returns if word is repeated
    if repeat_count > 0:
        fatigue_factor = 1 / (repeat_count + 1) # Calculate the penalty factor; the more repeats, the smaller the factor.
        final_score = max(1, int(base_score * fatigue_factor)) # Apply the fatigue factor and ensure the score is at least 1.
        logger.info(f"Word '{word}' scored {final_score} points after repeat fatigue (used {repeat_count + 1} times).") # Log the score after applying the fatigue penalty.
    else:
        final_score = base_score # If the word is not repeated, the final score is the base score.
        logger.info(f"New word '{word}' scored {final_score} points.") # Log the score of a new word.

    return final_score # Return the calculated final score.