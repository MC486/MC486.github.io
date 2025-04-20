import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.naive_bayes import NaiveBayes
from database.manager import DatabaseManager

class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.db_manager = Mock(spec=DatabaseManager)
        
        # Mock word analyzer methods and attributes
        self.word_analyzer.analyzed_words = {}
        self.word_analyzer.get_patterns.return_value = {
            'prefix': 'tes',
            'suffix': 'est',
            'length': '4'
        }
        self.word_analyzer.get_rarity_score.return_value = 0.5
        self.word_analyzer.get_word_frequency.return_value = 1
        self.word_analyzer.total_words = 10
        self.word_analyzer.get_pattern_frequency.return_value = 0.3
        self.word_analyzer.get_pattern_rarity.return_value = 0.4
        self.word_analyzer.get_pattern_success_rate.return_value = 0.5
        self.word_analyzer.get_pattern_weight.return_value = 0.5
        
        # Mock repository methods
        self.repository = Mock()
        self.db_manager.get_naive_bayes_repository.return_value = self.repository
        self.repository.get_word_probability.return_value = 0.0
        self.repository.get_learning_stats.return_value = {
            'total_observations': 0,
            'unique_words': 0,
            'average_probability': 0.0
        }
        
        self.model = NaiveBayes(self.event_manager, self.word_analyzer, self.db_manager)
    
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
        
        # Verify repository initialization
        self.db_manager.get_naive_bayes_repository.assert_called_once()
    
    def test_word_submission_handling(self):
        """Test model updates on word submission"""
        # Mock word analyzer methods
        self.word_analyzer.get_patterns.return_value = {
            'prefix': 'tes',
            'suffix': 'est',
            'length': '4'
        }
        self.word_analyzer.get_rarity_score.return_value = 0.5
        self.word_analyzer.get_word_frequency.return_value = 1
        self.word_analyzer.total_words = 10
        
        # Submit a test word with positive score
        event = Mock(spec=GameEvent)
        event.data = {"word": "test", "score": 10}
        self.model._handle_word_submission(event)
        
        self.assertEqual(self.model.total_observations, 1)
        self.assertGreater(self.model.word_probabilities["test"], 0)
        self.assertGreater(self.model.pattern_probabilities["prefix_tes"], 0)
        self.assertGreater(self.model.pattern_probabilities["suffix_est"], 0)
        
        # Verify repository updates
        self.repository.record_word_probability.assert_called()
    
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
        
        # Verify repository reload
        self.repository.get_learning_stats.assert_called()
    
    def test_probability_estimation(self):
        """Test word probability estimation"""
        # Mock word analyzer methods
        self.word_analyzer.get_patterns.return_value = {
            'prefix': 'tes',
            'suffix': 'est',
            'length': '4'
        }
        self.word_analyzer.get_pattern_weight.return_value = 0.5
        self.word_analyzer.get_rarity_score.return_value = 0.5
        self.word_analyzer.get_word_frequency.return_value = 1
        self.word_analyzer.total_words = 10
        
        # Test with no observations
        prob = self.model.estimate_word_probability("test")
        self.assertGreater(prob, 0)
        self.assertLessEqual(prob, 1.0)
        
        # Add some observations
        self.model._update_probabilities("test")
        self.model._update_probabilities("testing")
        
        # Test probability estimation
        prob = self.model.estimate_word_probability("test")
        self.assertGreater(prob, 0.01)  # Should be above minimum threshold
        self.assertLessEqual(prob, 1.0)  # Should not exceed 1.0
        
        # Verify repository usage
        self.repository.get_word_probability.assert_called()

if __name__ == '__main__':
    unittest.main()