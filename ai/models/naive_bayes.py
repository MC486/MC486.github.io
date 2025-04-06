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
from database.repositories.naive_bayes_repository import NaiveBayesRepository

logger = logging.getLogger(__name__)

class NaiveBayes:
    """
    Naive Bayes classifier for word prediction in the game.
    Uses word frequency and pattern analysis to estimate probabilities.
    """
    def __init__(self, event_manager: GameEventManager, word_analyzer: WordFrequencyAnalyzer, repo_manager):
        """Initialize the Naive Bayes model with event manager, word analyzer, and repository."""
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.repository = repo_manager.get_naive_bayes_repository() if repo_manager else None
        self.word_probabilities: Dict[str, float] = defaultdict(float)
        self.pattern_probabilities: Dict[str, float] = defaultdict(float)
        self.total_observations = 0
        
        # Load existing probabilities from repository
        if self.repository:
            self._load_from_repository()
        
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
            # Update probabilities
            self._update_probabilities(word)
            
            # Save state to repository periodically
            if self.total_observations % 10 == 0:  # Every 10 observations
                self.save_state()
                
            # Emit event with updated stats
            if self.repository:
                stats = self.repository.get_learning_stats()
                self.event_manager.emit(GameEvent(
                    type=EventType.MODEL_STATE_UPDATE,
                    data={
                        "message": "Naive Bayes model updated",
                        "word": word,
                        "score": score,
                        "total_observations": stats.get('total_observations', 0),
                        "unique_words": stats.get('unique_words', 0)
                    }
                ))
    
    def _handle_game_start(self, event: GameEvent) -> None:
        """Reset model state at game start"""
        # Clear local state
        self.word_probabilities.clear()
        self.pattern_probabilities.clear()
        self.total_observations = 0
        
        # Load fresh state from repository
        if self.repository:
            self._load_from_repository()
            
            # Emit event with initial stats
            stats = self.repository.get_learning_stats()
            self.event_manager.emit(GameEvent(
                type=EventType.MODEL_STATE_UPDATE,
                data={
                    "message": "Naive Bayes model initialized",
                    "total_observations": stats.get('total_observations', 0),
                    "unique_words": stats.get('unique_words', 0),
                    "average_probability": stats.get('average_probability', 0.0)
                }
            ))
    
    def _update_probabilities(self, word: str) -> None:
        """Update probabilities for a word and its patterns."""
        # Get word patterns
        patterns = self.word_analyzer.get_patterns(word)
        
        # Calculate and record base probability
        base_prob = self._calculate_base_probability(word)
        if self.repository:
            self.repository.record_word_probability(word, base_prob)
        self.word_probabilities[word] = base_prob
        
        # Update pattern probabilities
        for pattern_type, pattern in patterns.items():
            pattern_prob = self._calculate_pattern_probability(word, pattern, pattern_type)
            if self.repository:
                self.repository.record_word_probability(word, pattern_prob, pattern_type)
            self.pattern_probabilities[pattern] = pattern_prob
            
        # Update total observations
        self.total_observations += 1
    
    def _calculate_base_probability(self, word: str) -> float:
        """Calculate base probability for a word."""
        # Get existing probability from repository
        if self.repository:
            existing_prob = self.repository.get_word_probability(word)
            if existing_prob > 0:
                return existing_prob
            
        # Calculate new probability based on word characteristics
        length_factor = len(word) / 10.0  # Normalize by max expected length
        rarity_factor = self.word_analyzer.get_rarity_score(word)
        frequency_factor = self.word_analyzer.get_word_frequency(word) / self.word_analyzer.total_words
        
        # Combine factors with weights
        prob = (0.3 * length_factor + 0.4 * rarity_factor + 0.3 * frequency_factor)
        
        # Store in repository if available
        if self.repository:
            self.repository.record_word_probability(word, prob)
            
        return prob
        
    def _calculate_pattern_probability(self, word: str, pattern: str, pattern_type: str) -> float:
        """Calculate probability for a word pattern."""
        # Get existing probability from repository
        if self.repository:
            existing_prob = self.repository.get_word_probability(word, pattern_type)
            if existing_prob > 0:
                return existing_prob
            
        # Calculate new probability based on pattern characteristics
        pattern_frequency = self.word_analyzer.get_pattern_frequency(pattern)
        pattern_rarity = self.word_analyzer.get_pattern_rarity(pattern)
        pattern_success = self.word_analyzer.get_pattern_success_rate(pattern)
        
        # Combine factors with weights
        prob = (0.3 * pattern_frequency + 0.3 * pattern_rarity + 0.4 * pattern_success)
        
        # Store in repository if available
        if self.repository:
            self.repository.record_word_probability(word, prob, pattern_type)
            
        return prob
        
    def estimate_word_probability(self, word: str) -> float:
        """Estimate the probability of a word being valid."""
        # Get base probability from repository or calculate
        base_prob = self._calculate_base_probability(word)
        
        # Get pattern probabilities
        patterns = self.word_analyzer.get_patterns(word)
        pattern_probs = {}
        for pattern_type, pattern in patterns.items():
            pattern_probs[pattern_type] = self._calculate_pattern_probability(word, pattern, pattern_type)
            
        # Combine probabilities with weights
        if pattern_probs:
            # Weight patterns by their predictive power
            weighted_probs = []
            for pattern_type, prob in pattern_probs.items():
                weight = self.word_analyzer.get_pattern_weight(pattern_type)
                weighted_probs.append(prob * weight)
            
            if weighted_probs:
                pattern_prob = sum(weighted_probs) / sum(self.word_analyzer.get_pattern_weight(pt) 
                                                       for pt in pattern_probs.keys())
                return 0.4 * base_prob + 0.6 * pattern_prob
                
        return base_prob

    def train(self, words: List[str], labels: List[bool]) -> None:
        """Train the model with labeled words."""
        # Process words in batches for efficiency
        batch_size = 100
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Update probabilities for valid words
            for word, is_valid in zip(batch_words, batch_labels):
                if is_valid:
                    self._update_probabilities(word)
                    
            # Save batch to repository
            if self.repository:
                for word, is_valid in zip(batch_words, batch_labels):
                    if is_valid:
                        # Get patterns for word
                        patterns = self.word_analyzer.get_patterns(word)
                        
                        # Record word probability
                        prob = self.word_probabilities.get(word, 0.0)
                        self.repository.record_word_probability(word, prob)
                        
                        # Record pattern probabilities
                        for pattern_type, pattern in patterns.items():
                            prob = self.pattern_probabilities.get(pattern, 0.0)
                            self.repository.record_word_probability(word, prob, pattern_type)
                            
        # Update learning stats
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={
                "message": "Naive Bayes model trained",
                "total_words": len(words),
                "valid_words": sum(labels),
                "total_observations": self.total_observations
            }
        ))

    def get_learning_stats(self) -> Dict:
        """Get statistics about the model's learning."""
        return self.repository.get_learning_stats()
        
    def cleanup(self, days: int = 30) -> int:
        """Clean up old entries."""
        return self.repository.cleanup_old_entries(days)

    def _load_from_repository(self) -> None:
        """Load existing probabilities from repository."""
        # Load word probabilities
        for word in self.word_analyzer.words:
            prob = self.repository.get_word_probability(word)
            if prob > 0:
                self.word_probabilities[word] = prob
                
        # Load pattern probabilities
        for word in self.word_analyzer.words:
            patterns = self.word_analyzer.get_patterns(word)
            for pattern_type, pattern in patterns.items():
                prob = self.repository.get_word_probability(word, pattern_type)
                if prob > 0:
                    self.pattern_probabilities[pattern] = prob
                    
        # Update total observations
        stats = self.repository.get_learning_stats()
        self.total_observations = stats.get('total_observations', 0)

    def save_state(self) -> None:
        """Save current model state to repository."""
        if self.repository:
            # Save word probabilities
            for word, prob in self.word_probabilities.items():
                self.repository.record_word_probability(word, prob)
                
            # Save pattern probabilities
            for word in self.word_analyzer.words:
                patterns = self.word_analyzer.get_patterns(word)
                for pattern_type, pattern in patterns.items():
                    if pattern in self.pattern_probabilities:
                        prob = self.pattern_probabilities[pattern]
                        self.repository.record_word_probability(word, prob, pattern_type)
                        
    def load_state(self) -> None:
        """Load model state from repository."""
        if self.repository:
            self._load_from_repository()

    def get_word_stats(self, word: str) -> Dict:
        """Get statistics for a word from repository."""
        if self.repository:
            return self.repository.get_word_stats(word)
        return {
            "probability": self.word_probabilities.get(word, 0.0),
            "visit_count": 0,
            "patterns": {}
        }
        
    def get_pattern_stats(self, pattern_type: str) -> Dict:
        """Get statistics for a pattern type from repository."""
        if self.repository:
            return self.repository.get_pattern_stats(pattern_type)
        return {
            "total_words": 0,
            "average_probability": 0.0,
            "success_rate": 0.0
        }
        
    def get_model_stats(self) -> Dict:
        """Get comprehensive statistics about the model."""
        stats = {
            "total_observations": self.total_observations,
            "unique_words": len(self.word_probabilities),
            "unique_patterns": len(self.pattern_probabilities),
            "average_word_probability": sum(self.word_probabilities.values()) / len(self.word_probabilities) if self.word_probabilities else 0.0,
            "average_pattern_probability": sum(self.pattern_probabilities.values()) / len(self.pattern_probabilities) if self.pattern_probabilities else 0.0
        }
        
        if self.repository:
            repo_stats = self.repository.get_learning_stats()
            stats.update({
                "repository_stats": repo_stats,
                "total_stored_words": repo_stats.get('unique_words', 0),
                "average_stored_probability": repo_stats.get('average_probability', 0.0)
            })
            
        return stats
        
    def reset(self) -> None:
        """Reset the model state."""
        self.word_probabilities.clear()
        self.pattern_probabilities.clear()
        self.total_observations = 0
        
        if self.repository:
            self.repository.reset()