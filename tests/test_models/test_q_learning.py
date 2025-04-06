import unittest
import numpy as np
from unittest.mock import Mock, patch
from ai.q_learning import QLearningAgent, TrainingMetrics
from database.repositories.q_learning_repository import QLearningRepository

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.state_size = 4
        self.action_size = 2
        self.repository = Mock(spec=QLearningRepository)
        self.agent = QLearningAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            repository=self.repository
        )
    
    def test_initialization(self):
        """Test proper initialization of the agent"""
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.epsilon, 1.0)  # Default epsilon
        self.assertEqual(self.agent.epsilon_min, 0.01)  # Default epsilon_min
        self.assertEqual(self.agent.epsilon_decay, 0.995)  # Default epsilon_decay
        self.assertEqual(self.agent.batch_size, 32)  # Default batch_size
        self.assertEqual(self.agent.learning_rate, 0.001)  # Default learning_rate
        self.assertEqual(self.agent.gamma, 0.99)  # Default gamma
        
    def test_state_hashing(self):
        """Test state hashing functionality"""
        state = np.array([1, 2, 3, 4])
        hash1 = self.agent._hash_state(state)
        hash2 = self.agent._hash_state(state)
        
        # Same state should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different state should produce different hash
        different_state = np.array([1, 2, 3, 5])
        different_hash = self.agent._hash_state(different_state)
        self.assertNotEqual(hash1, different_hash)
    
    def test_choose_action_exploration(self):
        """Test action selection during exploration"""
        state = np.array([1, 2, 3, 4])
        state_hash = self.agent._hash_state(state)
        
        # Mock repository to return high exploration rate
        self.repository.get_state_stats.return_value = {
            'exploration_rate': 1.0,
            'total_actions': 0
        }
        
        # Force exploration by setting epsilon to 1
        self.agent.epsilon = 1.0
        
        action = self.agent.choose_action(state)
        
        # Action should be within valid range
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_size)
        
        # Repository should have been queried
        self.repository.get_state_stats.assert_called_once_with(state_hash)
    
    def test_choose_action_exploitation(self):
        """Test action selection during exploitation"""
        state = np.array([1, 2, 3, 4])
        state_hash = self.agent._hash_state(state)
        
        # Mock repository to return low exploration rate and best action
        self.repository.get_state_stats.return_value = {
            'exploration_rate': 0.0,
            'total_actions': 1
        }
        self.repository.get_best_action.return_value = "1"
        
        # Force exploitation by setting epsilon to 0
        self.agent.epsilon = 0.0
        
        action = self.agent.choose_action(state)
        
        # Action should be the best action from repository
        self.assertEqual(action, 1)
        
        # Repository should have been queried
        self.repository.get_state_stats.assert_called_once_with(state_hash)
        self.repository.get_best_action.assert_called_once_with(state_hash)
    
    def test_train(self):
        """Test training process"""
        # Create a batch of experiences
        batch = [
            (np.array([1, 2, 3, 4]), 0, 1.0, np.array([2, 3, 4, 5]), False),
            (np.array([2, 3, 4, 5]), 1, -1.0, np.array([3, 4, 5, 6]), True)
        ]
        
        # Mock memory to return the batch
        self.agent.memory.sample = Mock(return_value=batch)
        self.agent.memory.__len__ = Mock(return_value=self.agent.batch_size)
        
        # Mock repository methods
        self.repository.record_state_action = Mock()
        
        # Perform training
        metrics = self.agent.train()
        
        # Verify metrics
        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertEqual(metrics.loss, 0.0)  # Loss is handled by repository
        self.assertLess(metrics.epsilon, 1.0)  # Epsilon should have decayed
        self.assertEqual(metrics.memory_size, self.agent.batch_size)
        
        # Verify repository calls
        self.assertEqual(self.repository.record_state_action.call_count, len(batch))
        
        # Verify epsilon decay
        self.assertLess(self.agent.epsilon, 1.0)
    
    def test_get_learning_stats(self):
        """Test getting learning statistics"""
        # Mock repository to return stats
        expected_stats = {
            'total_states': 10,
            'total_actions': 20,
            'average_q_value': 0.5
        }
        self.repository.get_learning_stats.return_value = expected_stats
        
        stats = self.agent.get_learning_stats()
        
        self.assertEqual(stats, expected_stats)
        self.repository.get_learning_stats.assert_called_once()
    
    def test_cleanup_old_states(self):
        """Test cleaning up old states"""
        days = 30
        expected_removed = 5
        self.repository.cleanup_old_states.return_value = expected_removed
        
        removed = self.agent.cleanup_old_states(days)
        
        self.assertEqual(removed, expected_removed)
        self.repository.cleanup_old_states.assert_called_once_with(days)

if __name__ == '__main__':
    unittest.main()