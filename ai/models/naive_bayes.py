from typing import Dict, List, Set, Optional
import math
from collections import defaultdict
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ..repositories.naive_bayes_repository import NaiveBayesRepository

logger = logging.getLogger(__name__)

class NaiveBayes:
    """
    Naive Bayes classifier for word prediction in the game.
    Uses word frequency and pattern analysis to estimate probabilities.
    """
    def __init__(self, event_manager: GameEventManager, word_analyzer: WordFrequencyAnalyzer, db_manager):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.repository = NaiveBayesRepository(db_manager)
        self.word_probabilities: Dict[str, float] = defaultdict(float)
        self.pattern_probabilities: Dict[str, float] = defaultdict(float)
        self.total_observations = 0
        
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for model updates"""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
    
    def _handle_word_submission(self, event: GameEvent) -> None:
        """Update model based on word submissions"""
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        
        if score > 0:
            self._update_probabilities(word)
    
    def _handle_game_start(self, event: GameEvent) -> None:
        """Reset model state at game start"""
        self.word_probabilities.clear()
        self.pattern_probabilities.clear()
        self.total_observations = 0
    
    def _update_probabilities(self, word: str) -> None:
        """Update probabilities for a word and its patterns."""
        # Get word patterns
        patterns = self.word_analyzer.get_patterns(word)
        
        # Calculate base probability
        base_prob = self._calculate_base_probability(word)
        self.repository.record_word_probability(word, base_prob)
        
        # Update pattern probabilities
        for pattern_type, pattern in patterns.items():
            pattern_prob = self._calculate_pattern_probability(word, pattern)
            self.repository.record_word_probability(word, pattern_prob, pattern_type)
    
    def _calculate_base_probability(self, word: str) -> float:
        """Calculate base probability for a word."""
        # Get existing probability if available
        existing_prob = self.repository.get_word_probability(word)
        if existing_prob > 0:
            return existing_prob
            
        # Calculate new probability based on word characteristics
        length_factor = len(word) / 10.0  # Normalize by max expected length
        rarity_factor = self.word_analyzer.get_rarity_score(word)
        return length_factor * rarity_factor
        
    def _calculate_pattern_probability(self, word: str, pattern: str) -> float:
        """Calculate probability for a word pattern."""
        # Get existing probability if available
        existing_prob = self.repository.get_word_probability(word, pattern)
        if existing_prob > 0:
            return existing_prob
            
        # Calculate new probability based on pattern characteristics
        pattern_frequency = self.word_analyzer.get_pattern_frequency(pattern)
        return pattern_frequency
    
    def estimate_word_probability(self, word: str) -> float:
        """Estimate the probability of a word being valid."""
        # Get base probability
        base_prob = self.repository.get_word_probability(word)
        
        # Get pattern probabilities
        patterns = self.word_analyzer.get_patterns(word)
        pattern_probs = {}
        for pattern_type, pattern in patterns.items():
            pattern_probs[pattern_type] = self.repository.get_word_probability(word, pattern_type)
            
        # Combine probabilities
        if pattern_probs:
            pattern_prob = sum(pattern_probs.values()) / len(pattern_probs)
            return (base_prob + pattern_prob) / 2
        return base_prob

    def train(self, words: List[str], labels: List[bool]) -> None:
        """Train the model with labeled words."""
        for word, is_valid in zip(words, labels):
            if is_valid:
                self._update_probabilities(word)
                
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={"message": "Naive Bayes model trained"},
            debug_data={"word_count": len(words)}
        ))

    def get_learning_stats(self) -> Dict:
        """Get statistics about the model's learning."""
        return self.repository.get_learning_stats()
        
    def cleanup(self, days: int = 30) -> int:
        """Clean up old entries."""
        return self.repository.cleanup_old_entries(days)