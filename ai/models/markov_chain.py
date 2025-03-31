# ai/markov_chain.py

import random
import logging
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from core.validation.trie import Trie


class MarkovChain:
    """
    Markov Chain model for word generation.
    Uses transition probabilities between letters to generate words.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 word_analyzer: WordFrequencyAnalyzer,
                 trie: Trie,
                 order: int = 2):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.trie = trie
        self.order = order
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.start_probabilities: Dict[str, float] = {}
        
        self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> None:
        """Build transition probabilities matrix from word analyzer data"""
        words = self.word_analyzer.get_analyzed_words()
        total_words = len(words)
        
        # Build start probabilities
        for word in words:
            prefix = word[:self.order]
            self.start_probabilities[prefix] = self.start_probabilities.get(prefix, 0) + 1
            
        # Normalize start probabilities
        for prefix in self.start_probabilities:
            self.start_probabilities[prefix] /= total_words
            
        # Build transition probabilities
        for word in words:
            for i in range(len(word) - self.order):
                current = word[i:i+self.order]
                next_char = word[i+self.order]
                
                if current not in self.transitions:
                    self.transitions[current] = {}
                self.transitions[current][next_char] = \
                    self.transitions[current].get(next_char, 0) + 1
                    
        # Normalize transition probabilities
        for current in self.transitions:
            total = sum(self.transitions[current].values())
            for next_char in self.transitions[current]:
                self.transitions[current][next_char] /= total
                
    def generate_word(self, 
                     available_letters: List[str], 
                     prefix: str = "",
                     max_length: int = 15) -> Optional[str]:
        """
        Generate a word using Markov Chain and available letters.
        Uses Trie for validation and prefix guidance.
        Each letter can only be used once from the available pool.
        
        Args:
            available_letters: List of available letters
            prefix: Optional prefix to start with
            max_length: Maximum word length
            
        Returns:
            Generated word or None if no valid word found
        """
        min_length = 3  # Minimum word length requirement
        available_letters = [letter.upper() for letter in available_letters]  # Convert to list to handle duplicates
        prefix = prefix.upper()
        
        logging.info(f"Markov Chain generating word with letters: {available_letters}")
        if prefix:
            logging.info(f"Using prefix: {prefix}")
        
        # If prefix provided, validate it and remove used letters
        if prefix:
            if not self.trie.starts_with(prefix):
                logging.debug(f"Invalid prefix: {prefix}")
                return None
            # Remove letters used in prefix from available letters
            for letter in prefix:
                if letter in available_letters:
                    available_letters.remove(letter)
            
        # Try multiple times to generate a valid word
        for attempt in range(50):  # Increased from 10 to 50 attempts
            current_word = prefix
            remaining_letters = available_letters.copy()
            
            # Start with a common prefix if no prefix provided
            if not current_word:
                # Try all possible 2-letter combinations as starts
                valid_starts = []
                for i in range(len(remaining_letters)):
                    for j in range(i + 1, len(remaining_letters)):
                        start = remaining_letters[i] + remaining_letters[j]
                        if self.trie.starts_with(start):
                            valid_starts.append(start)
                            
                if not valid_starts:
                    logging.debug(f"Attempt {attempt + 1}: No valid starts found")
                    continue
                    
                # Choose start randomly from valid starts
                current_word = random.choice(valid_starts)
                logging.debug(f"Attempt {attempt + 1}: Chose start: {current_word}")
                # Remove letters used in start
                for letter in current_word:
                    if letter in remaining_letters:
                        remaining_letters.remove(letter)
            
            # Generate rest of the word
            while len(current_word) < max_length and remaining_letters:
                current = current_word[-self.order:] if len(current_word) >= self.order else current_word
                
                # Get valid next characters (only from remaining letters)
                valid_next = []
                for letter in remaining_letters:
                    next_word = current_word + letter
                    # Only consider letters that lead to valid words
                    if self.trie.starts_with(next_word):
                        # Check if this could lead to a complete word
                        if self.trie.search(next_word) or any(self.trie.starts_with(next_word + l) for l in remaining_letters):
                            valid_next.append(letter)
                
                if not valid_next:
                    # No valid next characters, check if current word is valid
                    if len(current_word) >= min_length and self.trie.search(current_word):
                        logging.info(f"Generated valid word: {current_word}")
                        return current_word
                    logging.debug(f"Attempt {attempt + 1}: No valid next characters for: {current_word}")
                    break
                    
                # Choose next character based on transition probabilities
                if current in self.transitions:
                    valid_transitions = {
                        char: prob for char, prob in self.transitions[current].items()
                        if char in valid_next
                    }
                    if valid_transitions:
                        next_char = random.choices(
                            list(valid_transitions.keys()),
                            weights=list(valid_transitions.values())
                        )[0]
                        current_word += next_char
                        if next_char in remaining_letters:
                            remaining_letters.remove(next_char)
                        logging.debug(f"Attempt {attempt + 1}: Added {next_char} -> {current_word}")
                        continue
                        
                # If no valid transitions, choose randomly from valid next characters
                next_char = random.choice(valid_next)
                current_word += next_char
                if next_char in remaining_letters:
                    remaining_letters.remove(next_char)
                logging.debug(f"Attempt {attempt + 1}: Randomly added {next_char} -> {current_word}")
                    
            # Check if we have a valid word of minimum length
            if len(current_word) >= min_length and self.trie.search(current_word):
                logging.info(f"Generated valid word: {current_word}")
                return current_word
                
        logging.warning("Failed to generate valid word after 50 attempts")
        return None
        
    def update(self, word: str, score: float) -> None:
        """Update model based on word success"""
        # Update transition probabilities based on successful words
        if score > 0:
            word = word.upper()
            for i in range(len(word) - self.order):
                current = word[i:i+self.order]
                next_char = word[i+self.order]
                
                if current not in self.transitions:
                    self.transitions[current] = {}
                    
                # Increase probability for successful transition
                total = sum(self.transitions[current].values())
                self.transitions[current][next_char] = \
                    self.transitions[current].get(next_char, 0) + score / total

    def train(self, words: List[str]) -> None:
        """
        Train the Markov Chain model on a list of words.
        
        Args:
            words (List[str]): List of words to train on
        """
        self.word_analyzer.analyze_word_list(words)
        self._build_transition_matrix()
        
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={"message": "Markov Chain model trained"},
            debug_data={"word_count": len(words)}
        ))
