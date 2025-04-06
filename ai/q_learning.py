import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import random
import logging
import os
from dataclasses import dataclass
from datetime import datetime
import hashlib
from database.repositories.q_learning_repository import QLearningRepository

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    epsilon: float
    memory_size: int
    timestamp: str

class QNetwork:
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [64, 32]):
        """
        Initialize the Q-network using TensorFlow.
        
        Args:
            state_size: Size of the state representation
            action_size: Size of the action space
            hidden_layers: List of hidden layer sizes
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Build the network
        layers = [tf.keras.layers.Dense(size, activation='relu') for size in hidden_layers]
        layers.append(tf.keras.layers.Dense(action_size, activation='linear'))
        
        self.model = tf.keras.Sequential(layers)
        
        # Compile with Adam optimizer and MSE loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Initialize training history
        self.training_history = []
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for a given state.
        
        Args:
            state: State representation
            
        Returns:
            np.ndarray: Q-values for each action
        """
        try:
            return self.model.predict(state.reshape(1, -1))[0]
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
        
    def train(self, states: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Train the network on a batch of experiences.
        
        Args:
            states: Batch of states
            targets: Target Q-values
            
        Returns:
            Dict[str, float]: Training metrics
        """
        try:
            history = self.model.fit(states, targets, epochs=1, verbose=0)
            metrics = {
                'loss': history.history['loss'][0],
                'mae': history.history['mae'][0]
            }
            self.training_history.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
            
    def get_training_history(self) -> List[Dict[str, float]]:
        """
        Get the training history.
        
        Returns:
            List[Dict[str, float]]: List of training metrics
        """
        return self.training_history.copy()

class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        try:
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.error(f"Error adding experience to buffer: {str(e)}")
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List[Tuple]: List of (state, action, reward, next_state, done) tuples
        """
        try:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        except Exception as e:
            logger.error(f"Error sampling from buffer: {str(e)}")
            raise RuntimeError(f"Sampling failed: {str(e)}")
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def clear(self) -> None:
        """Clear the replay buffer."""
        self.buffer.clear()

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 repository: Optional[QLearningRepository] = None):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_size: Size of the state representation
            action_size: Size of the action space
            learning_rate: Learning rate for the network
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the replay buffer
            batch_size: Size of training batches
            repository: QLearningRepository instance for persistent storage
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize repository
        self.repository = repository
        
        # Initialize replay buffer for experience replay
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize training metrics
        self.metrics_history = []
        
    def _hash_state(self, state: np.ndarray) -> str:
        """Convert state array to a hash string."""
        return hashlib.sha256(state.tobytes()).hexdigest()
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Chosen action
        """
        try:
            state_hash = self._hash_state(state)
            
            # Get exploration rate from repository
            state_stats = self.repository.get_state_stats(state_hash)
            exploration_rate = state_stats['exploration_rate']
            
            # Use epsilon-greedy with repository-based exploration
            if random.random() < max(self.epsilon, exploration_rate):
                # Try to find least explored action
                least_explored = self.repository.get_least_explored_action(state_hash)
                if least_explored is not None:
                    return int(least_explored)
                return random.randrange(self.action_size)
                
            # Get best action from repository
            best_action = self.repository.get_best_action(state_hash)
            if best_action is not None:
                return int(best_action)
                
            return random.randrange(self.action_size)
            
        except Exception as e:
            logger.error(f"Error choosing action: {str(e)}")
            return random.randrange(self.action_size)
        
    def train(self) -> Optional[TrainingMetrics]:
        """
        Train the agent on a batch of experiences.
        
        Returns:
            Optional[TrainingMetrics]: Training metrics if training occurred
        """
        if len(self.memory) < self.batch_size:
            return None
            
        try:
            # Sample batch of experiences
            batch = self.memory.sample(self.batch_size)
            
            # Process each experience
            for state, action, reward, next_state, done in batch:
                state_hash = self._hash_state(state)
                next_state_hash = self._hash_state(next_state)
                
                # Update Q-value in repository
                self.repository.record_state_action(
                    state_hash=state_hash,
                    action=str(action),
                    reward=reward,
                    next_state_hash=next_state_hash,
                    learning_rate=self.learning_rate,
                    discount_factor=self.gamma
                )
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record metrics
            metrics = TrainingMetrics(
                loss=0.0,  # Loss is now handled by repository
                epsilon=self.epsilon,
                memory_size=len(self.memory),
                timestamp=datetime.now().isoformat()
            )
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            return None
            
    def get_metrics_history(self) -> List[TrainingMetrics]:
        """
        Get the training metrics history.
        
        Returns:
            List[TrainingMetrics]: List of training metrics
        """
        return self.metrics_history.copy()
        
    def get_learning_stats(self) -> Dict:
        """
        Get learning statistics from the repository.
        
        Returns:
            Dict: Learning statistics
        """
        if self.repository:
            return self.repository.get_learning_stats()
        return {}
        
    def cleanup_old_states(self, days: int = 30) -> int:
        """
        Clean up old states from the repository.
        
        Args:
            days: Number of days after which to remove states
            
        Returns:
            int: Number of states removed
        """
        if self.repository:
            return self.repository.cleanup_old_states(days)
        return 0

    def save(self, directory: str) -> None:
        """
        Save the agent's networks and state to disk.
        
        Args:
            directory: Directory to save the agent
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save networks
            self.q_network.model.save(f"{directory}/q_network.h5")
            self.target_network.model.save(f"{directory}/target_network.h5")
            
            # Save agent state
            import pickle
            with open(f"{directory}/agent_state.pkl", 'wb') as f:
                pickle.dump({
                    'epsilon': self.epsilon,
                    'training_steps': self.training_steps,
                    'metrics_history': self.metrics_history
                }, f)
                
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}")
            
    def load(self, directory: str) -> None:
        """
        Load the agent's networks and state from disk.
        
        Args:
            directory: Directory to load the agent from
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
            ValueError: If the files contain invalid data
        """
        try:
            # Load networks
            self.q_network.model = tf.keras.models.load_model(f"{directory}/q_network.h5")
            self.target_network.model = tf.keras.models.load_model(f"{directory}/target_network.h5")
            
            # Load agent state
            import pickle
            with open(f"{directory}/agent_state.pkl", 'rb') as f:
                state = pickle.load(f)
                self.epsilon = state['epsilon']
                self.training_steps = state['training_steps']
                self.metrics_history = state['metrics_history']
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Agent directory not found: {directory}")
        except Exception as e:
            raise ValueError(f"Invalid agent data: {str(e)}") 