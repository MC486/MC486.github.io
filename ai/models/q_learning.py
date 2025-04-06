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
                 repo_manager = None, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 exploration_rate: float = 0.2):
        """Initialize Q-Learning agent with event manager, word analyzer, and repository."""
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.repository = repo_manager.get_q_learning_repository() if repo_manager else None
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Q-table: state -> action -> value mapping
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track current state and last action
        self.current_state: Optional[str] = None
        self.last_action: Optional[str] = None
        
        # Load existing Q-values from repository
        if self.repository:
            self._load_from_repository()
        
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
            # Calculate reward based on word score and length
            base_reward = score / 10.0  # Normalize score to reasonable range
            length_bonus = min(len(word) / 10.0, 0.5)  # Bonus for longer words
            reward = base_reward + length_bonus
            
            # Update Q-value
            self._update_q_value(self.current_state, self.last_action, reward)
            
            # Save state periodically
            if random.random() < 0.1:  # 10% chance to save state
                self.save_state()
            
            # Emit event with updated stats
            if self.repository:
                stats = self.repository.get_learning_stats()
                self.event_manager.emit(GameEvent(
                    type=EventType.MODEL_STATE_UPDATE,
                    data={
                        "message": "Q-Learning model updated",
                        "word": word,
                        "score": score,
                        "reward": reward,
                        "total_states": stats.get('total_states', 0),
                        "total_actions": stats.get('total_actions', 0),
                        "average_q_value": stats.get('average_q_value', 0.0)
                    }
                ))
    
    def _handle_game_start(self, event: GameEvent) -> None:
        """Reset agent state at game start"""
        # Reset current state
        self.current_state = None
        self.last_action = None
        
        # Load fresh state from repository
        if self.repository:
            self._load_from_repository()
            
            # Emit event with initial stats
            stats = self.repository.get_learning_stats()
            self.event_manager.emit(GameEvent(
                type=EventType.MODEL_STATE_UPDATE,
                data={
                    "message": "Q-Learning model initialized",
                    "total_states": stats.get('total_states', 0),
                    "total_actions": stats.get('total_actions', 0),
                    "average_q_value": stats.get('average_q_value', 0.0)
                }
            ))
    
    def _handle_turn_start(self, event: GameEvent) -> None:
        """Update state at turn start"""
        available_letters = event.data.get("available_letters", set())
        self.current_state = self._get_state_representation(available_letters)
        
        # Get state statistics from repository
        if self.repository:
            stats = self.repository.get_state_stats(self.current_state)
            if stats:
                self.event_manager.emit(GameEvent(
                    type=EventType.MODEL_STATE_UPDATE,
                    data={
                        "message": "Q-Learning state updated",
                        "state": self.current_state,
                        "total_actions": stats.get('total_actions', 0),
                        "best_action": stats.get('best_action', None),
                        "average_q_value": stats.get('average_q_value', 0.0)
                    }
                ))
    
    def _update_q_value(self, state: str, action: str, reward: float) -> None:
        """Update Q-value for state-action pair and save to repository."""
        # Get current Q-value from repository or local table
        current_q = self.q_table[state][action]
        if self.repository:
            repo_q = self.repository.get_q_value(state, action)
            if repo_q is not None:
                current_q = repo_q
                
        # Get max Q-value for next state
        max_next_q = 0
        if self.current_state:
            if self.repository:
                next_actions = self.repository.get_state_actions(self.current_state)
                if next_actions:
                    max_next_q = max(next_actions.values())
            else:
                max_next_q = max(self.q_table[self.current_state].values()) if self.q_table[self.current_state] else 0
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update local Q-table
        self.q_table[state][action] = new_q
        
        # Save to repository
        if self.repository:
            self.repository.record_q_value(state, action, new_q)
            
        # Save state periodically
        if random.random() < 0.1:  # 10% chance to save state
            self.save_state()
    
    def select_action(self, available_letters: Set[str], valid_words: Set[str]) -> str:
        """Select next word based on Q-values and exploration strategy."""
        state = self._get_state_representation(available_letters)
        
        # Exploration: random selection
        if random.random() < self.exploration_rate:
            valid_candidates = [word for word in valid_words 
                              if set(word).issubset(available_letters)]
            if valid_candidates:
                action = random.choice(valid_candidates)
                # Record exploration in repository
                if self.repository:
                    self.repository.record_exploration(state, action)
                return action
            return ""
        
        # Get Q-values from repository if available
        if self.repository:
            state_actions = self.repository.get_state_actions(state)
            if state_actions:
                valid_actions = {
                    word: q_value for word, q_value in state_actions.items()
                    if word in valid_words and set(word).issubset(available_letters)
                }
                if valid_actions:
                    best_action = max(valid_actions.items(), key=lambda x: x[1])[0]
                    self.last_action = best_action
                    return best_action
        
        # Fall back to local Q-table
        valid_actions = {
            word: self.q_table[state][word] for word in valid_words
            if set(word).issubset(available_letters)
        }
        
        if not valid_actions:
            return ""
        
        # Select action with highest Q-value
        best_action = max(valid_actions.items(), key=lambda x: x[1])[0]
        self.last_action = best_action
        
        return best_action

    def _load_from_repository(self) -> None:
        """Load Q-values from repository."""
        if self.repository:
            # Get all state-action pairs
            state_actions = self.repository.get_all_state_actions()
            
            # Update Q-table with stored values
            for state, actions in state_actions.items():
                for action, q_value in actions.items():
                    self.q_table[state][action] = q_value
                    
    def save_state(self) -> None:
        """Save current Q-values to repository."""
        if self.repository:
            # Save all state-action pairs
            for state, actions in self.q_table.items():
                for action, q_value in actions.items():
                    self.repository.record_q_value(state, action, q_value)
                    
    def load_state(self) -> None:
        """Load Q-values from repository."""
        if self.repository:
            self._load_from_repository()
            
    def get_state_stats(self, state: str) -> Dict:
        """Get statistics for a state from repository."""
        if self.repository:
            return self.repository.get_state_stats(state)
        return {
            "total_actions": len(self.q_table[state]),
            "average_q_value": sum(self.q_table[state].values()) / len(self.q_table[state]) if self.q_table[state] else 0.0,
            "best_action": max(self.q_table[state].items(), key=lambda x: x[1])[0] if self.q_table[state] else None
        }
        
    def get_learning_stats(self) -> Dict:
        """Get learning statistics from repository."""
        if self.repository:
            return self.repository.get_learning_stats()
        return {
            "total_states": len(self.q_table),
            "total_actions": sum(len(actions) for actions in self.q_table.values()),
            "average_q_value": sum(sum(actions.values()) for actions in self.q_table.values()) / 
                             sum(len(actions) for actions in self.q_table.values()) if self.q_table else 0.0
        }
        
    def cleanup(self, days: int = 30) -> None:
        """Cleanup old entries from repository."""
        if self.repository:
            self.repository.cleanup_old_entries(days)

    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for a state-action pair."""
        if self.repository:
            q_value = self.repository.get_q_value(state, action)
            if q_value is not None:
                return q_value
        return self.q_table[state][action]
        
    def get_best_action(self, state: str) -> Optional[str]:
        """Get best action for a state based on Q-values."""
        if self.repository:
            return self.repository.get_best_action(state)
        if state in self.q_table and self.q_table[state]:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
        return None
        
    def get_action_stats(self, state: str, action: str) -> Dict:
        """Get statistics for a state-action pair."""
        if self.repository:
            return self.repository.get_action_stats(state, action)
        return {
            "q_value": self.q_table[state][action],
            "visit_count": 0,
            "last_updated": None
        }
        
    def get_model_stats(self) -> Dict:
        """Get comprehensive statistics about the model."""
        stats = {
            "total_states": len(self.q_table),
            "total_actions": sum(len(actions) for actions in self.q_table.values()),
            "average_q_value": sum(sum(actions.values()) for actions in self.q_table.values()) / 
                             sum(len(actions) for actions in self.q_table.values()) if self.q_table else 0.0,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor
        }
        
        if self.repository:
            repo_stats = self.repository.get_learning_stats()
            stats.update({
                "repository_stats": repo_stats,
                "total_stored_states": repo_stats.get('total_states', 0),
                "total_stored_actions": repo_stats.get('total_actions', 0),
                "average_stored_q_value": repo_stats.get('average_q_value', 0.0)
            })
            
        return stats
        
    def reset(self) -> None:
        """Reset the model state."""
        self.q_table.clear()
        self.current_state = None
        self.last_action = None
        
        if self.repository:
            self.repository.reset()