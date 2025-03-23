# core/letter_pool.py
# Handles generation of shared and boggle letter pools.

import random
import logging

logger = logging.getLogger(__name__)

# Common English letters for shared pool (higher frequency)
COMMON_LETTERS = list("ETAOINSHRDLU") # List of common letters, used for the shared letters.

# Full alphabet for boggle letters, weighted by usage frequency
WEIGHTED_ALPHABET = [ # List of letters, weighted by how often they are used in the English language.
    'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E',
    'A', 'A', 'A', 'A', 'A', 'A', 'A',
    'R', 'R', 'R', 'R', 'R',
    'I', 'I', 'I', 'I', 'I',
    'O', 'O', 'O', 'O', 'O',
    'T', 'T', 'T', 'T',
    'N', 'N', 'N', 'N',
    'S', 'S', 'S', 'S',
    'L', 'L', 'L',
    'C', 'C', 'C',
    'U', 'U',
    'D', 'D',
    'M', 'M',
    'P', 'P',
    'B',
    'G',
    'Y',
    'F',
    'W',
    'K',
    'V',
    'X',
    'Z',
    'J',
    'Q'
]

def generate_letter_pool(num_shared=4, num_boggle=6):
    """
    Generates a pool of shared letters and player-specific boggle letters.
    Ensures shared letters are distinct and include at least one vowel and one consonant.
    """
    while True: # Loop until a valid set of shared letters is generated.
        shared = random.sample(COMMON_LETTERS, num_shared) # Randomly select shared letters.
        if any(l in "AEIOU" for l in shared) and any(l not in "AEIOU" for l in shared): # Check if there is at least one vowel and one consonant.
            break # Exit the loop if the shared letters meet the criteria.

    boggle = [random.choice(WEIGHTED_ALPHABET) for _ in range(num_boggle)] # Randomly select boggle letters, respecting the weighted alphabet.
    logger.debug(f"Generated shared letters: {shared}") # Log the generated shared letters.
    logger.debug(f"Generated boggle letters: {boggle}") # Log the generated boggle letters.
    return shared, boggle # Return the shared and boggle letters.