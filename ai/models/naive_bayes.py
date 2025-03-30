from typing import Dict, List, Set, Optional
import math
from collections import defaultdict
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer

class NaiveBayes:
    """
    Naive Bayes classifier for word prediction in the game.
    Uses word frequency and pattern analysis to estimate probabilities.
    """
    def __init__(self, event_manager: GameEventManager, word_analyzer: WordFrequencyAnalyzer):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
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
        """Update word and pattern probabilities based on new observation"""
        self.total_observations += 1
        self.word_probabilities[word] += 1
        
        # Update pattern probabilities (e.g., prefixes, suffixes)
        if len(word) >= 3:
            prefix = word[:3]
            suffix = word[-3:]
            self.pattern_probabilities[f"prefix_{prefix}"] += 1
            self.pattern_probabilities[f"suffix_{suffix}"] += 1
    
    def estimate_word_probability(self, word: str) -> float:
        """
        Estimate probability of a word being valid and valuable.
        Combines word frequency and pattern matching.
        """
        if self.total_observations == 0:
            return self.word_analyzer.get_word_score(word)
        
        # Calculate word probability
        word_prob = self.word_probabilities[word] / self.total_observations
        
        # Calculate pattern probability
        pattern_prob = 0.0
        if len(word) >= 3:
            prefix = word[:3]
            suffix = word[-3:]
            prefix_prob = self.pattern_probabilities[f"prefix_{prefix}"] / self.total_observations
            suffix_prob = self.pattern_probabilities[f"suffix_{suffix}"] / self.total_observations
            pattern_prob = (prefix_prob + suffix_prob) / 2
        
        # Combine probabilities with smoothing
        combined_prob = (word_prob + pattern_prob + self.word_analyzer.get_word_score(word)) / 3
        return max(0.01, combined_prob)  # Ensure non-zero probability