# ai/markov_model.py

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
                     available_letters: Set[str], 
                     prefix: str = "",
                     max_length: int = 15) -> Optional[str]:
        """
        Generate a word using Markov Chain and available letters.
        Uses Trie for validation and prefix guidance.
        
        Args:
            available_letters: Set of available letters
            prefix: Optional prefix to start with
            max_length: Maximum word length
            
        Returns:
            Generated word or None if no valid word found
        """
        available_letters = set(letter.upper() for letter in available_letters)
        prefix = prefix.upper()
        
        # If prefix provided, validate it
        if prefix and not self.trie.starts_with(prefix):
            return None
            
        # Try multiple times to generate a valid word
        for _ in range(10):
            current_word = prefix
            letters_used = set(prefix)
            
            # Start with a common prefix if no prefix provided
            if not current_word:
                valid_starts = [
                    start for start in self.start_probabilities
                    if all(c in available_letters for c in start)
                    and self.trie.starts_with(start)
                ]
                if not valid_starts:
                    continue
                    
                # Choose start based on probabilities
                weights = [self.start_probabilities[start] for start in valid_starts]
                current_word = random.choices(valid_starts, weights=weights)[0]
                letters_used = set(current_word)
            
            # Generate rest of the word
            while len(current_word) < max_length:
                current = current_word[-self.order:] if len(current_word) >= self.order else current_word
                
                # Get valid next characters
                valid_next = [
                    char for char in available_letters - letters_used
                    if self.trie.starts_with(current_word + char)
                ]
                
                if not valid_next:
                    # No valid next characters, check if current word is valid
                    if len(current_word) >= 3 and self.trie.search(current_word):
                        return current_word
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
                        letters_used.add(next_char)
                        continue
                        
                # If no valid transitions, choose randomly from valid next characters
                next_char = random.choice(valid_next)
                current_word += next_char
                letters_used.add(next_char)
                
                # Check if current word is valid
                if len(current_word) >= 3 and self.trie.search(current_word):
                    return current_word
                    
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
