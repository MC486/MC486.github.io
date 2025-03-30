from typing import Dict, List, Tuple, Set, Optional
import random
import numpy as np
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer

class QLearningAgent:
    """
    Q-Learning implementation for adaptive word selection strategy.
    Learns optimal word selection policies through reinforcement learning.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 word_analyzer: WordFrequencyAnalyzer,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state -> {action -> value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # State tracking
        self.current_state: Optional[str] = None
        self.last_action: Optional[str] = None
        self.total_reward = 0.0
        
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for Q-Learning updates"""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)

    def _get_state_key(self, available_letters: Set[str], turn_number: int) -> str:
        """
        Convert current game state to a hashable key.
        
        Args:
            available_letters: Set of available letters
            turn_number: Current turn number
            
        Returns:
            String representation of state
        """
        letters_key = ''.join(sorted(available_letters))
        return f"{letters_key}_{turn_number}"

    def _get_valid_actions(self, available_letters: Set[str], valid_words: Set[str]) -> List[str]:
        """
        Get list of valid words that can be formed with available letters.
        
        Args:
            available_letters: Set of available letters
            valid_words: Set of valid words
            
        Returns:
            List of valid possible words
        """
        return [word for word in valid_words 
                if set(word.upper()).issubset(available_letters)]

    def select_action(self, 
                     available_letters: Set[str], 
                     valid_words: Set[str],
                     turn_number: int) -> str:
        """
        Select word using epsilon-greedy policy.
        
        Args:
            available_letters: Set of available letters
            valid_words: Set of valid words
            turn_number: Current turn number
            
        Returns:
            Selected word
        """
        state = self._get_state_key(available_letters, turn_number)
        self.current_state = state
        
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = {}
        
        valid_actions = self._get_valid_actions(available_letters, valid_words)
        
        # Initialize Q-values for new actions
        for action in valid_actions:
            if action not in self.q_table[state]:
                self.q_table[state][action] = self.word_analyzer.get_word_score(action)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(valid_actions) if valid_actions else ""
        else:
            # Exploitation: best known action
            action = max(valid_actions,
                        key=lambda a: self.q_table[state].get(a, 0),
                        default="")
        
        self.last_action = action
        
        self.event_manager.emit(GameEvent(
            type=EventType.AI_DECISION_MADE,
            data={"message": "Q-Learning selected word"},
            debug_data={
                "word": action,
                "state": state,
                "q_value": self.q_table[state].get(action, 0),
                "exploration": random.random() < self.epsilon
            }
        ))
        
        return action

    def update(self, reward: float, 
               next_available_letters: Set[str],
               next_turn_number: int) -> None:
        """
        Update Q-values based on received reward.
        
        Args:
            reward: Reward received for last action
            next_available_letters: Available letters after action
            next_turn_number: Next turn number
        """
        if not (self.current_state and self.last_action):
            return
            
        next_state = self._get_state_key(next_available_letters, next_turn_number)
        
        # Initialize next state in Q-table if not present
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
        
        # Get max Q-value for next state
        next_max_q = max(self.q_table[next_state].values(), default=0)
        
        # Q-learning update
        current_q = self.q_table[self.current_state][self.last_action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[self.current_state][self.last_action] = new_q
        self.total_reward += reward

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission events"""
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        next_letters = set(event.data.get("next_available_letters", []))
        turn_number = event.data.get("turn_number", 0)
        
        if word and self.last_action == word:
            self.update(score, next_letters, turn_number + 1)

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Reset current state and action at turn start"""
        self.current_state = None
        self.last_action = None

    def _handle_game_start(self, event: GameEvent) -> None:
        """Adjust exploration rate at game start"""
        # Gradually reduce exploration as agent learns
        self.epsilon = max(0.01, self.epsilon * 0.95)

    def save_model(self, filepath: str) -> None:
        """Save Q-table to file"""
        # Implementation for model saving
        pass

    def load_model(self, filepath: str) -> None:
        """Load Q-table from file"""
        # Implementation for model loading
        pass

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "total_states": len(self.q_table),
            "total_actions": sum(len(actions) for actions in self.q_table.values()),
            "total_reward": self.total_reward,
            "current_epsilon": self.epsilon
        }