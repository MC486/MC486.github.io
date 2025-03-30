# ai/markov_model.py

import random
import logging
from collections import defaultdict
from typing import List, Dict


class MarkovModel:
    """
    A simple first-order character-level Markov model.
    Trains on a list of words and predicts likely next characters based on frequency.
    """

    def __init__(self):
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.trained = False

    def train(self, words: List[str]):
        """
        Builds transition frequencies from a list of words.
        """
        for word in words:
            padded_word = '^' + word + '$'  # Add start and end markers
            for i in range(len(padded_word) - 1):
                current_char = padded_word[i]
                next_char = padded_word[i + 1]
                self.transitions[current_char][next_char] += 1

        self.trained = True
        logging.debug("Markov model trained on %d words.", len(words))

    def generate(self, max_length: int = 10) -> str:
        """
        Generates a word-like string using learned transitions.
        """
        if not self.trained:
            raise ValueError("Model must be trained before generating.")

        word = ''
        current_char = '^'
        while current_char != '$' and len(word) < max_length:
            next_chars = self.transitions[current_char]
            total = sum(next_chars.values())
            if not next_chars:
                break
            choices, weights = zip(*next_chars.items())
            current_char = random.choices(choices, weights=weights, k=1)[0]
            if current_char != '$':
                word += current_char

        return word
