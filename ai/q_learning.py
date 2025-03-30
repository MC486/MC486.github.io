import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import random
import logging
import os
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
                 target_update_frequency: int = 100):
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
            target_update_frequency: How often to update the target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.training_steps = 0
        
        # Initialize networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.model.set_weights(self.q_network.model.get_weights())
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Initialize training metrics
        self.metrics_history = []
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            int: Chosen action
        """
        try:
            if random.random() < self.epsilon:
                return random.randrange(self.action_size)
            return np.argmax(self.q_network.predict(state))
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
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
            dones = np.array([x[4] for x in batch])
            
            # Get current Q-values
            current_q_values = self.q_network.predict(states)
            
            # Get next Q-values from target network
            next_q_values = self.target_network.predict(next_states)
            
            # Update Q-values
            for i in range(self.batch_size):
                if dones[i]:
                    current_q_values[i][actions[i]] = rewards[i]
                else:
                    current_q_values[i][actions[i]] = rewards[i] + \
                        self.gamma * np.max(next_q_values[i])
                        
            # Train the network
            metrics = self.q_network.train(states, current_q_values)
            
            # Update target network periodically
            self.training_steps += 1
            if self.training_steps % self.target_update_frequency == 0:
                self.target_network.model.set_weights(self.q_network.model.get_weights())
                
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Record metrics
            training_metrics = TrainingMetrics(
                loss=metrics['loss'],
                epsilon=self.epsilon,
                memory_size=len(self.memory),
                timestamp=datetime.now().isoformat()
            )
            self.metrics_history.append(training_metrics)
            
            return training_metrics
            
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