import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.naive_bayes import NaiveBayes

class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.model = NaiveBayes(self.event_manager, self.word_analyzer)
    
    def test_initialization(self):
        """Test proper initialization of the model"""
        self.assertEqual(self.model.total_observations, 0)
        self.assertEqual(len(self.model.word_probabilities), 0)
        self.assertEqual(len(self.model.pattern_probabilities), 0)
        
        # Verify event subscriptions
        self.event_manager.subscribe.assert_any_call(
            EventType.WORD_SUBMITTED, self.model._handle_word_submission
        )
        self.event_manager.subscribe.assert_any_call(
            EventType.GAME_START, self.model._handle_game_start
        )
    
    def test_word_submission_handling(self):
        """Test model updates on word submission"""
        # Submit a test word with positive score
        event = Mock(spec=GameEvent)
        event.data = {"word": "test", "score": 10}
        self.model._handle_word_submission(event)
        
        self.assertEqual(self.model.total_observations, 1)
        self.assertEqual(self.model.word_probabilities["test"], 1)
        self.assertEqual(self.model.pattern_probabilities["prefix_tes"], 1)
        self.assertEqual(self.model.pattern_probabilities["suffix_est"], 1)
    
    def test_game_start_reset(self):
        """Test model reset on game start"""
        # Add some data first
        self.model.word_probabilities["test"] = 1
        self.model.pattern_probabilities["prefix_tes"] = 1
        self.model.total_observations = 1
        
        # Trigger game start
        event = Mock(spec=GameEvent)
        self.model._handle_game_start(event)
        
        self.assertEqual(self.model.total_observations, 0)
        self.assertEqual(len(self.model.word_probabilities), 0)
        self.assertEqual(len(self.model.pattern_probabilities), 0)
    
    def test_probability_estimation(self):
        """Test word probability estimation"""
        # Setup word analyzer mock
        self.word_analyzer.get_word_score.return_value = 0.5
        
        # Test with no observations
        prob = self.model.estimate_word_probability("test")
        self.assertEqual(prob, 0.5)  # Should return word analyzer score
        
        # Add some observations
        self.model._update_probabilities("test")
        self.model._update_probabilities("testing")
        
        # Test probability estimation
        prob = self.model.estimate_word_probability("test")
        self.assertGreater(prob, 0.01)  # Should be above minimum threshold
        self.assertLessEqual(prob, 1.0)  # Should not exceed 1.0

if __name__ == '__main__':
    unittest.main()