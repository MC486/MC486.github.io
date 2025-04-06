# ai/markov_chain.py

import random
import logging
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from core.validation.trie import Trie
from database.repositories.markov_repository import MarkovRepository


class MarkovChain:
    """
    Markov Chain model for word generation.
    Uses transition probabilities between letters to generate words.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 word_analyzer: WordFrequencyAnalyzer,
                 trie: Trie,
                 markov_repository: MarkovRepository,
                 order: int = 2):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.trie = trie
        self.order = order
        self.markov_repository = markov_repository
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
                
    def generate_word(self, available_letters: str, prefix: str = "") -> str:
        """Generate a word based on available letters and optional prefix"""
        available_letters = available_letters.upper()
        prefix = prefix.upper()
        
        # Get transitions from repository
        transitions = self.markov_repository.get_transitions()
        
        # If no transitions in repository, use local transitions
        if not transitions:
            transitions = self.transitions
        
        # Start with prefix if provided
        current = prefix[-self.order:] if prefix else ""
        word = prefix
        
        # Generate word
        while len(word) < 15:  # Maximum word length
            # Get possible next characters
            if current in transitions:
                next_chars = [c for c in transitions[current].keys() 
                            if c in available_letters and c not in word]
                if not next_chars:
                    break
                    
                # Choose next character based on probabilities
                probs = [transitions[current][c] for c in next_chars]
                next_char = random.choices(next_chars, weights=probs)[0]
            else:
                # If no transitions for current state, choose random available letter
                next_chars = [c for c in available_letters if c not in word]
                if not next_chars:
                    break
                next_char = random.choice(next_chars)
            
            word += next_char
            current = word[-self.order:]
            
            # Check if word is valid
            if self.trie.is_word(word):
                return word
        
        return word
        
    def update(self, word: str, score: float) -> None:
        """Update model based on word success"""
        # Update transition probabilities based on successful words
        if score > 0:
            word = word.upper()
            transitions = []
            for i in range(len(word) - self.order):
                current = word[i:i+self.order]
                next_char = word[i+self.order]
                transitions.append((current, next_char, int(score * 100)))
            
            # Bulk update transitions in repository
            self.markov_repository.bulk_update_transitions(transitions)
            
            # Update local transitions
            for current, next_char, count in transitions:
                if current not in self.transitions:
                    self.transitions[current] = {}
                self.transitions[current][next_char] = \
                    self.transitions[current].get(next_char, 0) + count

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

    def get_model_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the model"""
        # Get stats from repository
        repo_stats = self.markov_repository.get_learning_stats()
        
        # Combine with local stats
        stats = {
            "order": self.order,
            "total_states": len(self.transitions),
            "total_transitions": sum(len(trans) for trans in self.transitions.values()),
            "repository_stats": repo_stats
        }
        
        return stats

    def cleanup(self, days: int = 30) -> None:
        """Cleanup old entries from the repository"""
        self.markov_repository.cleanup_old_entries(days)

    def save(self) -> None:
        """Save the model to the repository"""
        self.markov_repository.bulk_record_transitions(self.transitions)

    def load(self) -> None:
        """Load the model from the repository"""
        transitions = self.markov_repository.get_transitions()
        if transitions:
            self.transitions = transitions

    def get_start_probability(self, start: str) -> float:
        """Get the probability of starting with a given state"""
        # Try to get from repository first
        prob = self.markov_repository.get_start_probability(start)
        if prob is not None:
            return prob
            
        # Fall back to local start probabilities
        return self.start_probabilities.get(start, 0.0)

    def get_state_stats(self, state: str) -> Dict[str, Any]:
        """Get statistics for a given state"""
        # Try to get from repository first
        stats = self.markov_repository.get_state_stats(state)
        if stats:
            return stats
            
        # Fall back to local transitions
        if state in self.transitions:
            return {
                "total_transitions": len(self.transitions[state]),
                "transitions": self.transitions[state]
            }
        return {"total_transitions": 0, "transitions": {}}
