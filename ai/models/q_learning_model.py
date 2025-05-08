"""
Q-Learning implementation for adaptive word selection strategy.
Combines features from both implementations for optimal performance.
"""
from typing import Dict, List, Tuple, Set, Optional
import random
import numpy as np
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from database.repositories.q_learning_repository import QLearningRepository
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    epsilon: float
    memory_size: int
    timestamp: str

class QLearningAgent:
    """
    Q-Learning implementation for adaptive word selection strategy.
    Learns optimal word selection policies through reinforcement learning.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 word_analyzer: WordFrequencyAnalyzer,
                 repository: QLearningRepository,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        """
        Initialize the Q-Learning agent.
        
        Args:
            event_manager: Game event manager
            word_analyzer: Word frequency analyzer
            repository: Q-Learning repository
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Exploration rate
        """
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.repository = repository
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # State tracking
        self.current_state: Optional[str] = None
        self.last_action: Optional[str] = None
        self.total_reward = 0.0
        
        # Training metrics
        self.training_metrics: List[TrainingMetrics] = []
        
        # Q-table: state -> {action -> value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # Load existing data
        self._load_from_repository()
        
        # Setup event subscriptions
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

    def _normalize_state(self, state: str) -> str:
        """
        Normalize state representation by removing turn number.
        
        Args:
            state: State key to normalize
            
        Returns:
            Normalized state representation
        """
        return state.split('_')[0]

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

    def _validate_action(self, action: str, available_letters: Set[str]) -> bool:
        """
        Validate if action can be formed with available letters.
        
        Args:
            action: Word to validate
            available_letters: Set of available letters
            
        Returns:
            True if action is valid, False otherwise
        """
        return set(action.upper()).issubset(available_letters)

    def _adjust_learning_rate(self) -> None:
        """
        Adjust learning rate based on performance.
        Decreases learning rate when performing well, increases when struggling.
        """
        if len(self.training_metrics) > 0:
            recent_loss = self.training_metrics[-1].loss
            if recent_loss < 0.1:  # If performing well
                self.learning_rate *= 0.99  # Decrease learning rate
            else:
                self.learning_rate = min(0.1, self.learning_rate * 1.01)  # Increase learning rate

    def _check_performance_threshold(self) -> bool:
        """
        Check if performance meets threshold based on recent metrics.
        
        Returns:
            True if performance meets threshold, False otherwise
        """
        if len(self.training_metrics) < 10:
            return False
        recent_metrics = self.training_metrics[-10:]
        avg_loss = sum(m.loss for m in recent_metrics) / 10
        return avg_loss < 0.1

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
        try:
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
            
            # Validate selected action
            if not self._validate_action(action, available_letters):
                logger.warning(f"Invalid action selected: {action}")
                action = random.choice(valid_actions) if valid_actions else ""
            
            self.last_action = action
            
            # Emit event with debug data
            self.event_manager.emit(GameEvent(
                type=EventType.AI_DECISION_MADE,
                data={"message": "Q-Learning selected word"},
                debug_data={
                    "word": action,
                    "state": state,
                    "q_value": self.q_table[state].get(action, 0),
                    "exploration": random.random() < self.epsilon,
                    "learning_rate": self.learning_rate
                }
            ))
            
            return action
            
        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            return ""

    def choose_action(self, state: str) -> str:
        """
        Choose action for given state using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        try:
            # Initialize state in Q-table if not present
            if state not in self.q_table:
                self.q_table[state] = {}
            
            # Get valid actions for state
            valid_actions = list(self.q_table[state].keys())
            
            if not valid_actions:
                return ""
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Exploration: random action
                return random.choice(valid_actions)
            else:
                # Exploitation: best known action
                return max(valid_actions,
                         key=lambda a: self.q_table[state].get(a, 0))
                         
        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            return ""

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
        try:
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
            
            # Record training metrics
            self._record_training_metrics(abs(new_q - current_q))
            
            # Update Q-table
            self.q_table[self.current_state][self.last_action] = new_q
            self.total_reward += reward
            
            # Adjust learning rate based on performance
            self._adjust_learning_rate()
            
            # Save to repository periodically
            if len(self.training_metrics) % 10 == 0:  # Every 10 updates
                self._save_to_repository()
                
        except Exception as e:
            logger.error(f"Error in Q-learning update: {str(e)}")

    def _record_training_metrics(self, loss: float) -> None:
        """
        Record training metrics.
        
        Args:
            loss: Current loss value
        """
        self.training_metrics.append(TrainingMetrics(
            loss=loss,
            epsilon=self.epsilon,
            memory_size=len(self.q_table),
            timestamp=datetime.now().isoformat()
        ))

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission events"""
        try:
            word = event.data.get("word", "")
            score = event.data.get("score", 0)
            next_letters = set(event.data.get("next_available_letters", []))
            turn_number = event.data.get("turn_number", 0)
            
            if word and self.last_action == word:
                self.update(score, next_letters, turn_number + 1)
        except Exception as e:
            logger.error(f"Error handling word submission: {str(e)}")

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Reset current state and action at turn start"""
        self.current_state = None
        self.last_action = None

    def _handle_game_start(self, event: GameEvent) -> None:
        """Adjust exploration rate at game start"""
        # Gradually reduce exploration as agent learns
        self.epsilon = max(0.01, self.epsilon * 0.95)

    def _load_from_repository(self) -> None:
        """Load Q-values and training metrics from repository"""
        try:
            # Load Q-values
            q_values = self.repository.get_q_values()
            if q_values:
                self.q_table.clear()
                for state, actions in q_values.items():
                    self.q_table[state].update(actions)
            
            # Load training metrics
            metrics = self.repository.get_training_metrics()
            if metrics:
                self.training_metrics = [
                    TrainingMetrics(
                        loss=m['loss'],
                        epsilon=m['epsilon'],
                        memory_size=m['memory_size'],
                        timestamp=m['timestamp']
                    ) for m in metrics
                ]
        except Exception as e:
            logger.warning(f"Failed to load from repository: {e}")

    def _save_to_repository(self) -> None:
        """Save current state to repository"""
        try:
            self.repository.save_q_values(self.q_table)
            self.repository.save_training_metrics([
                {
                    'loss': m.loss,
                    'epsilon': m.epsilon,
                    'memory_size': m.memory_size,
                    'timestamp': m.timestamp
                } for m in self.training_metrics
            ])
        except Exception as e:
            logger.error(f"Failed to save to repository: {e}")

    def get_stats(self) -> Dict:
        """Get basic agent statistics"""
        return {
            "total_states": len(self.q_table),
            "total_actions": sum(len(actions) for actions in self.q_table.values()),
            "total_reward": self.total_reward,
            "current_epsilon": self.epsilon
        }

    def get_enhanced_stats(self) -> Dict:
        """Get enhanced statistics including performance metrics"""
        stats = self.get_stats()
        stats.update({
            "avg_recent_loss": sum(m.loss for m in self.training_metrics[-10:]) / 10 if self.training_metrics else 0,
            "learning_rate": self.learning_rate,
            "performance_threshold_met": self._check_performance_threshold(),
            "total_unique_states": len(set(self._normalize_state(s) for s in self.q_table.keys())),
            "avg_actions_per_state": sum(len(actions) for actions in self.q_table.values()) / len(self.q_table) if self.q_table else 0,
            "training_metrics": [
                {
                    'loss': m.loss,
                    'epsilon': m.epsilon,
                    'memory_size': m.memory_size,
                    'timestamp': m.timestamp
                } for m in self.training_metrics[-10:]  # Last 10 metrics
            ]
        })
        return stats

    def cleanup(self, days: int = 30) -> int:
        """
        Clean up old entries.
        
        Args:
            days: Number of days to keep
            
        Returns:
            int: Number of entries cleaned
        """
        try:
            return self.repository.cleanup_old_entries(days)
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            return 0 