import unittest
import numpy as np
from unittest.mock import Mock, patch
from ai.models.q_learning_model import QLearningAgent as QLearning, TrainingMetrics
from database.repositories.q_learning_repository import QLearningRepository
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from datetime import datetime

class TestQLearning(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.repository = Mock(spec=QLearningRepository)
        self.db_manager = Mock()
        self.db_manager.get_q_learning_repository.return_value = self.repository
        
        self.agent = QLearning(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            repository=self.repository
        )
    
    def test_initialization(self):
        """Test proper initialization of the agent"""
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.9)
        self.assertEqual(self.agent.exploration_rate, 0.2)
        self.assertIsNotNone(self.agent.repository)
        self.assertIsNone(self.agent.current_state)
        self.assertIsNone(self.agent.last_action)
        
    def test_state_representation(self):
        """Test state representation functionality"""
        available_letters = {'A', 'B', 'C', 'D'}
        state = self.agent._get_state_representation(available_letters)
        self.assertEqual(state, 'ABCD')
        
    def test_word_submission_handling(self):
        """Test handling of word submissions"""
        # Set up initial state
        self.agent.current_state = 'ABCD'
        self.agent.last_action = 'WORD'
        
        # Create test event
        event = Mock()
        event.data = {
            'word': 'WORD',
            'score': 10
        }
        
        # Mock repository methods
        self.repository.get_state_stats.return_value = {
            'visits': 1,
            'value': 0.0
        }
        
        # Handle word submission
        self.agent._handle_word_submission(event)
        
        # Verify Q-value update
        self.assertGreater(self.agent.q_table['ABCD']['WORD'], 0)
        
    def test_training_metrics(self):
        """Test training metrics tracking"""
        # Create test metrics
        metrics = TrainingMetrics(
            loss=0.1,
            epsilon=0.5,
            memory_size=100,
            timestamp=datetime.now().isoformat()
        )
        
        # Add metrics
        self.agent.training_metrics.append(metrics)
        
        # Verify metrics
        self.assertEqual(len(self.agent.training_metrics), 1)
        self.assertEqual(self.agent.training_metrics[0].loss, 0.1)
        self.assertEqual(self.agent.training_metrics[0].epsilon, 0.5)
        self.assertEqual(self.agent.training_metrics[0].memory_size, 100)

if __name__ == '__main__':
    unittest.main()