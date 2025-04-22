"""
Q-Learning Implementation
This module implements a Q-Learning algorithm for the word game AI, learning optimal
strategies through reinforcement learning. It maintains a Q-table of state-action pairs
and updates them based on game outcomes.
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from game_state import GameState

class QLearning:
    def __init__(self):
        # Learning parameters
        self.learning_rate = 0.1    # Alpha: how much to update Q-values
        self.discount_factor = 0.9  # Gamma: importance of future rewards
        self.exploration_rate = 0.2 # Epsilon: chance of random exploration
        
        # Q-table: maps state-action pairs to Q-values
        self.q_table: Dict[str, Dict[Tuple[int, int, str], float]] = {}
        
        # Experience replay buffer
        self.replay_buffer: List[Tuple[str, Tuple[int, int, str], float, str]] = []
        self.buffer_size = 10000
        
        # Target network for stable learning
        self.target_network: Dict[str, Dict[Tuple[int, int, str], float]] = {}
        
        # Update frequency for target network
        self.update_frequency = 1000
        self.update_count = 0

    def get_moves(self, game_state: GameState) -> List[Tuple[int, int, str]]:
        """
        Get best moves using Q-Learning
        Steps:
        1. Get current state representation
        2. Get Q-values for possible actions
        3. Select moves with highest Q-values
        """
        state = self._get_state_representation(game_state)
        
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {}
            
        # Get legal moves
        legal_moves = self._get_legal_moves(game_state)
        
        # Get Q-values for moves
        q_values = []
        for move in legal_moves:
            if move in self.q_table[state]:
                q_values.append((move, self.q_table[state][move]))
            else:
                # Initialize new state-action pair
                self.q_table[state][move] = 0.0
                q_values.append((move, 0.0))
        
        # Sort by Q-value
        q_values.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in q_values[:3]]  # Return top 3 moves

    def update(self, game_history: List[Tuple[int, int, str]]):
        """
        Update Q-values based on game history
        Steps:
        1. Process game history
        2. Update Q-values using TD learning
        3. Update target network periodically
        """
        for i in range(len(game_history) - 1):
            state = self._get_state_representation(game_history[i])
            action = game_history[i]
            next_state = self._get_state_representation(game_history[i + 1])
            reward = self._calculate_reward(game_history[i], game_history[i + 1])
            
            # Store experience in replay buffer
            self.replay_buffer.append((state, action, reward, next_state))
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
            
            # Update Q-values
            self._update_q_value(state, action, reward, next_state)
            
            # Update target network
            self.update_count += 1
            if self.update_count % self.update_frequency == 0:
                self._update_target_network()

    def _update_q_value(self, 
                       state: str,
                       action: Tuple[int, int, str],
                       reward: float,
                       next_state: str):
        """
        Update Q-value using TD learning
        Q(s,a) = Q(s,a) + α[r + γmaxQ(s',a') - Q(s,a)]
        """
        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {}
        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            
        # Get current Q-value
        current_q = self.q_table[state].get(action, 0.0)
        
        # Get maximum Q-value for next state
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q

    def _calculate_reward(self, 
                         current_move: Tuple[int, int, str],
                         next_move: Tuple[int, int, str]) -> float:
        """
        Calculate reward for a move
        Implementation:
        1. Score based on word length
        2. Bonus for strategic placement
        3. Penalty for poor moves
        """
        reward = 0.0
        
        # Base reward on word length
        word_length = len(current_move[2])
        reward += word_length * 0.1
        
        # Strategic placement bonus
        x, y, _ = current_move
        if x == 3 or y == 3:  # Center positions
            reward += 0.5
            
        return reward

    def _get_state_representation(self, game_state: GameState) -> str:
        """
        Convert game state to string representation
        Used as key in Q-table
        """
        # Implementation depends on game state structure
        pass

    def _get_legal_moves(self, game_state: GameState) -> List[Tuple[int, int, str]]:
        """Get all legal moves for current game state"""
        # Implementation depends on game rules
        pass

    def _update_target_network(self):
        """Update target network with current Q-table values"""
        self.target_network = {
            state: {action: q_value for action, q_value in actions.items()}
            for state, actions in self.q_table.items()
        } 