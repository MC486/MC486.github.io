import unittest
from unittest.mock import Mock, patch
from ai.strategy.ai_strategy import AIStrategy
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer

class TestAIStrategy(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.strategy = AIStrategy(self.event_manager, self.word_analyzer)

    def test_initialization(self):
        """Test proper initialization of AIStrategy"""
        self.assertIsNotNone(self.strategy.models)
        self.assertEqual(len(self.strategy.model_weights), 4)  # Four AI models
        self.assertEqual(sum(self.strategy.model_weights.values()), 1.0)  # Weights sum to 1

    def test_get_word_suggestion(self):
        """Test word suggestion generation"""
        available_letters = ['A', 'B', 'C', 'D', 'E']
        
        # Mock model suggestions
        self.strategy.models['markov'].get_suggestion.return_value = ('BACK', 0.8)
        self.strategy.models['naive_bayes'].get_suggestion.return_value = ('DECK', 0.6)
        self.strategy.models['mcts'].get_suggestion.return_value = ('CAKE', 0.7)
        self.strategy.models['q_learning'].get_suggestion.return_value = ('BEAD', 0.5)

        word = self.strategy.get_word_suggestion(available_letters)
        self.assertIsInstance(word, str)
        self.assertTrue(all(letter in available_letters for letter in word))

    def test_model_weight_adjustment(self):
        """Test adjustment of model weights based on success"""
        initial_weights = self.strategy.model_weights.copy()
        
        # Simulate successful word from markov model
        self.strategy._adjust_weights('markov', success=True)
        
        self.assertGreater(
            self.strategy.model_weights['markov'],
            initial_weights['markov']
        )
        self.assertEqual(sum(self.strategy.model_weights.values()), 1.0)

    def test_event_handling(self):
        """Test event system integration"""
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "success": True,
                "model": "markov"
            }
        )
        
        self.strategy._handle_word_submission(event)
        self.assertTrue(self.strategy.models['markov'].update.called)

    def test_strategy_reset(self):
        """Test strategy reset on game start"""
        event = GameEvent(
            type=EventType.GAME_START,
            data={}
        )
        
        initial_weights = self.strategy.model_weights.copy()
        self.strategy._handle_game_start(event)
        
        self.assertEqual(self.strategy.model_weights, initial_weights)
        for model in self.strategy.models.values():
            self.assertTrue(model.reset.called)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty letters list
        word = self.strategy.get_word_suggestion([])
        self.assertEqual(word, '')
        
        # None letters list
        word = self.strategy.get_word_suggestion(None)
        self.assertEqual(word, '')

    def test_model_selection(self):
        """Test model selection based on weights"""
        # Set deterministic weights for testing
        self.strategy.model_weights = {
            'markov': 1.0,
            'naive_bayes': 0.0,
            'mcts': 0.0,
            'q_learning': 0.0
        }
        
        self.strategy.models['markov'].get_suggestion.return_value = ('TEST', 0.9)
        
        word = self.strategy.get_word_suggestion(['T', 'E', 'S', 'T'])
        self.assertEqual(word, 'TEST')

    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        self.strategy.track_performance('markov', 'HELLO', True)
        
        stats = self.strategy.get_performance_stats()
        self.assertIn('markov', stats)
        self.assertGreater(stats['markov']['success_rate'], 0)

if __name__ == '__main__':
    unittest.main()