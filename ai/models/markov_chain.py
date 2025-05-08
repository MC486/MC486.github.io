# ai/markov_chain.py

import logging
import random
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from core.validation.trie import Trie
from database.repositories.markov_repository import MarkovRepository

logger = logging.getLogger(__name__)

class MarkovChain:
    """
    Markov Chain model for word generation.
    Uses transition probabilities between letters to generate words.
    """
    def __init__(self, 
                 event_manager: Optional[GameEventManager] = None,
                 word_analyzer: Optional[WordFrequencyAnalyzer] = None,
                 trie: Optional[Trie] = None,
                 markov_repository: Optional[MarkovRepository] = None,
                 order: int = 2):
        """
        Initialize the Markov Chain model.
        
        Args:
            event_manager (GameEventManager): Event manager for game events
            word_analyzer (WordFrequencyAnalyzer): Word analyzer for frequency data
            trie (Trie): Trie data structure for word validation
            markov_repository (MarkovRepository): Repository for storing transitions
            order (int): Order of the Markov Chain (default: 2)
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
            
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.trie = trie
        self.markov_repository = markov_repository
        self.order = order
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.start_probabilities: Dict[str, float] = {}
        self.is_trained = False
        self.min_length = 3  # Minimum word length
        self.max_length = 15  # Maximum word length
        self.word_validator = word_analyzer.word_validator if word_analyzer else None
        
        if word_analyzer:
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
                
    def generate_word(self, available_letters: Set[str]) -> Optional[str]:
        """Generate a word using available letters."""
        if not self.transitions:
            logger.warning("Model not trained, returning None")
            return None
            
        # Convert available letters to uppercase
        available_letters = {letter.upper() for letter in available_letters}
        
        # Try to generate a word
        max_attempts = 10
        for _ in range(max_attempts):
            word = []
            current_state = self._choose_start_state()
            
            # Generate word
            for _ in range(self.max_length):
                if current_state not in self.transitions:
                    break
                    
                next_letter = self._choose_next_letter(current_state, available_letters)
                if not next_letter:
                    break
                    
                word.append(next_letter)
                current_state = self._update_state(current_state, next_letter)
                
            # Check if word is valid
            generated_word = ''.join(word)
            if len(generated_word) >= self.min_length and self.word_validator.is_valid_word(generated_word):
                return generated_word
                
        return None
        
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
        # Check if repository is available
        if not self.markov_repository:
            raise RuntimeError("No repository available")
            
        # Convert all words to uppercase
        words = [word.upper() for word in words]
        
        # Validate words
        if not words:
            raise ValueError("Cannot train on empty word list")
        if not all(word.isalpha() for word in words):
            raise ValueError("All words must contain only letters")
            
        # Train word analyzer if available
        if self.word_analyzer:
            self.word_analyzer.analyze_word_list(words)
            
        # Build transition matrix
        self._build_transition_matrix()
        
        # Record transitions in repository
        for word in words:
            # Record start transition
            prefix = word[:self.order]
            self.markov_repository.record_transition("START", prefix)
            
            # Record letter transitions
            for i in range(len(word) - self.order):
                current = word[i:i+self.order]
                next_char = word[i+self.order]
                self.markov_repository.record_transition(current, next_char)
        
        # Set trained flag
        self.is_trained = True
        
        # Emit event if event manager is available
        if self.event_manager:
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

    def get_state_probabilities(self, state: str) -> Dict[str, float]:
        """Get transition probabilities for a given state."""
        if not self.markov_repository:
            raise RuntimeError("No repository available")
        if not self.is_trained:
            raise RuntimeError("Model not trained")
            
        state = state.upper()
        return self.markov_repository.get_state_probabilities(state)

    def _choose_start_state(self) -> str:
        """Choose a start state based on start probabilities."""
        if not self.start_probabilities:
            return ""
            
        states = list(self.start_probabilities.keys())
        weights = list(self.start_probabilities.values())
        return random.choices(states, weights=weights)[0]
        
    def _choose_next_letter(self, current_state: str, available_letters: Set[str]) -> Optional[str]:
        """Choose the next letter based on transition probabilities and available letters."""
        if current_state not in self.transitions:
            return None
            
        # Filter transitions to only use available letters
        valid_transitions = {
            letter: prob for letter, prob in self.transitions[current_state].items()
            if letter in available_letters
        }
        
        if not valid_transitions:
            return None
            
        letters = list(valid_transitions.keys())
        weights = list(valid_transitions.values())
        return random.choices(letters, weights=weights)[0]
        
    def _update_state(self, current_state: str, next_letter: str) -> str:
        """Update the current state with the next letter."""
        if len(current_state) < self.order:
            return current_state + next_letter
        return current_state[1:] + next_letter

    def get_suggestion(self, available_letters: Set[str]) -> Tuple[str, float]:
        """
        Get word suggestion with confidence score.
        
        Args:
            available_letters: Set of available letters
            
        Returns:
            Tuple of (word, confidence_score)
        """
        word = self.generate_word(available_letters)
        if not word:
            return ("", 0.0)
            
        # Calculate confidence based on transition probabilities
        confidence = 0.0
        for i in range(len(word) - 1):
            prefix = word[i:i+self.order]
            next_char = word[i+1] if i+1 < len(word) else "END"
            transitions = self.markov_repository.get_transitions(prefix)
            if transitions and next_char in transitions:
                confidence += transitions[next_char]
                
        # Normalize confidence
        confidence = confidence / len(word) if word else 0.0
        
        return (word, confidence)
