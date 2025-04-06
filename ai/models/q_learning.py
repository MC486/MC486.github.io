from typing import Dict, List, Set, Tuple, Optional
import random
from collections import defaultdict
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
import logging

logger = logging.getLogger(__name__)

class QLearningAgent:
    """
    Q-Learning agent for word selection strategy.
    Uses reinforcement learning to optimize word choices based on game state.
    """
    def __init__(self, event_manager: GameEventManager, word_analyzer: WordFrequencyAnalyzer,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: state -> action -> value mapping
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track current state and last action
        self.current_state: Optional[str] = None
        self.last_action: Optional[str] = None
        
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for model updates"""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)
    
    def _get_state_representation(self, available_letters: Set[str]) -> str:
        """Convert current game state to string representation"""
        return ''.join(sorted(available_letters))
    
    def _handle_word_submission(self, event: GameEvent) -> None:
        """Update Q-values based on word submission results"""
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        
        if self.current_state and self.last_action:
            # Calculate reward based on word score
            reward = score / 10.0  # Normalize score to reasonable range
            
            # Update Q-value
            self._update_q_value(self.current_state, self.last_action, reward)
    
    def _handle_game_start(self, event: GameEvent) -> None:
        """Reset agent state at game start"""
        self.current_state = None
        self.last_action = None
    
    def _handle_turn_start(self, event: GameEvent) -> None:
        """Update state at turn start"""
        available_letters = event.data.get("available_letters", set())
        self.current_state = self._get_state_representation(available_letters)
    
    def _update_q_value(self, state: str, action: str, reward: float) -> None:
        """Update Q-value for state-action pair"""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[self.current_state].values()) if self.current_state else 0
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def select_action(self, available_letters: Set[str], valid_words: Set[str]) -> str:
        """Select next word based on Q-values and exploration strategy"""
        state = self._get_state_representation(available_letters)
        
        # Exploration: random selection
        if random.random() < self.exploration_rate:
            valid_candidates = [word for word in valid_words 
                              if set(word).issubset(available_letters)]
            return random.choice(valid_candidates) if valid_candidates else ""
        
        # Exploitation: select best Q-value
        valid_actions = {word: self.q_table[state][word] for word in valid_words
                        if set(word).issubset(available_letters)}
        
        if not valid_actions:
            return ""
        
        # Select action with highest Q-value
        best_action = max(valid_actions.items(), key=lambda x: x[1])[0]
        self.last_action = best_action
        return best_action