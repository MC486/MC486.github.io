import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from ai.strategy.ai_strategy import AIStrategy
from core.game_events import GameEvent, EventType
from core.game_events_manager import game_events_manager
from ai.word_analysis import WordFrequencyAnalyzer
from database.manager import DatabaseManager

class TestAIStrategy(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=game_events_manager)
        self.db_manager = Mock(spec=DatabaseManager)
        
        # Mock repositories
        self.word_repo = Mock()
        self.word_repo.get_word_usage.return_value = []
        self.word_repo.get_analyzed_words = Mock(return_value=[])
        self.word_repo.validate_word_with_letters = Mock(return_value=True)
        
        self.category_repo = Mock()
        self.category_repo.get_word_categories = Mock(return_value={})
        
        self.markov_repo = Mock()
        self.markov_repo.record_transition = Mock()
        self.markov_repo.get_model_weight = Mock(return_value=0.25)
        self.markov_repo.get_transitions = Mock(return_value={})
        self.markov_repo.game_id = 1  # Set a game_id to avoid the check
        
        self.naive_bayes_repo = Mock()
        self.naive_bayes_repo.get_model_weight = Mock(return_value=0.25)
        
        self.mcts_repo = Mock()
        self.mcts_repo.get_model_weight = Mock(return_value=0.25)
        
        self.q_learning_repo = Mock()
        self.q_learning_repo.get_model_weight = Mock(return_value=0.25)
        
        # Setup database manager mocks
        self.db_manager.get_word_repository.return_value = self.word_repo
        self.db_manager.get_category_repository.return_value = self.category_repo
        self.db_manager.get_markov_repository.return_value = self.markov_repo
        self.db_manager.get_naive_bayes_repository.return_value = self.naive_bayes_repo
        self.db_manager.get_mcts_repository.return_value = self.mcts_repo
        self.db_manager.get_q_learning_repository.return_value = self.q_learning_repo
        
        # Mock models
        self.markov_chain = MagicMock()
        self.markov_chain.get_suggestion = Mock(return_value=("TEST", 0.9))
        self.markov_chain.get_stats = Mock(return_value={"success_rate": 0.8})
        
        self.mcts = MagicMock()
        self.mcts.get_suggestion = Mock(return_value=("BEST", 0.8))
        self.mcts.get_stats = Mock(return_value={"success_rate": 0.7})
        
        self.naive_bayes = MagicMock()
        self.naive_bayes.get_suggestion = Mock(return_value=("GOOD", 0.7))
        self.naive_bayes.get_stats = Mock(return_value={"success_rate": 0.6})
        
        self.q_agent = MagicMock()
        self.q_agent.get_suggestion = Mock(return_value=("NICE", 0.6))
        self.q_agent.get_stats = Mock(return_value={"success_rate": 0.5})
        
        # Patch model classes
        with patch('ai.models.MarkovChain', return_value=self.markov_chain), \
             patch('ai.models.MCTS', return_value=self.mcts), \
             patch('ai.models.NaiveBayes', return_value=self.naive_bayes), \
             patch('ai.models.QLearning', return_value=self.q_agent), \
             patch('database.repositories.markov_repository.MarkovRepository._check_game_id'), \
             patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=True):
            self.strategy = AIStrategy(
                self.event_manager,
                self.db_manager,
                self.word_repo,
                self.category_repo
            )

    def test_initialization(self):
        """Test proper initialization of AIStrategy"""
        self.assertIsNotNone(self.strategy.models)
        self.assertEqual(len(self.strategy.model_weights), 4)  # Four AI models
        self.assertEqual(sum(self.strategy.model_weights.values()), 1.0)  # Weights sum to 1
        
        # Verify repository initialization
        self.db_manager.get_markov_repository.assert_called()
        self.db_manager.get_naive_bayes_repository.assert_called()
        self.db_manager.get_mcts_repository.assert_called()
        self.db_manager.get_q_learning_repository.assert_called()

    def test_model_weight_adjustment(self):
        """Test adjustment of model weights based on success"""
        initial_weights = self.strategy.model_weights.copy()
        
        # Simulate successful word from markov model
        self.strategy._adjust_weights('markov_test', 10)
        
        self.assertGreater(
            self.strategy.model_weights['markov'],
            initial_weights['markov']
        )
        self.assertEqual(sum(self.strategy.model_weights.values()), 1.0)
        
        # Verify repository updates
        self.markov_repo.update_model_weight.assert_called_once()

    def test_event_handling(self):
        """Test event system integration"""
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "success": True,
                "model": "markov",
                "score": 10
            }
        )
        
        self.strategy._handle_word_submission(event)
        
        # Verify repository updates
        self.markov_repo.record_word_usage.assert_called_once()

    def test_strategy_reset(self):
        """Test strategy reset on game start"""
        event = GameEvent(
            type=EventType.GAME_START,
            data={}
        )
        
        initial_weights = self.strategy.model_weights.copy()
        self.strategy._handle_game_start(event)
        
        self.assertEqual(self.strategy.model_weights, initial_weights)
            
        # Verify repository resets
        self.markov_repo.reset_model.assert_called_once()
        self.naive_bayes_repo.reset_model.assert_called_once()
        self.mcts_repo.reset_model.assert_called_once()
        self.q_learning_repo.reset_model.assert_called_once()

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty letters list
        word = self.strategy.select_word(set(), set(), 1)
        self.assertEqual(word, '')
        
        # None letters list
        word = self.strategy.select_word(None, None, 1)
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

        # Mock the markov chain's get_suggestion method
        self.markov_chain.get_suggestion = Mock(return_value=("TEST", 0.9))
        self.markov_chain.is_trained = True  # Set trained flag to True

        # Mock the word validator
        self.strategy.word_validator.validate_word_with_letters = Mock(return_value=True)

        # Mock the other models to return empty suggestions
        self.mcts.get_suggestion = Mock(return_value=("", 0.0))
        self.naive_bayes.get_suggestion = Mock(return_value=("", 0.0))
        self.q_agent.get_suggestion = Mock(return_value=("", 0.0))

        # Update models in strategy
        self.strategy.models['markov'] = self.markov_chain
        self.strategy.models['mcts'] = self.mcts
        self.strategy.models['naive_bayes'] = self.naive_bayes
        self.strategy.models['q_learning'] = self.q_agent

        word = self.strategy.select_word(set(['T', 'E', 'S', 'T']), set(), 1)
        self.assertEqual(word, 'TEST')

    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        # Set initial stats
        initial_stats = {
            "success_rate": 0.0,
            "total_decisions": 0,
            "successful_words": 0
        }

        # Create a mock for get_stats
        self.markov_chain.get_stats = Mock(return_value=initial_stats)

        # Update models in strategy to ensure we're using our mock
        self.strategy.models['markov'] = self.markov_chain

        # Track performance
        self.strategy.track_performance('markov', 'HELLO', True)

        # Update the mock to return new stats
        updated_stats = {
            "success_rate": 1.0,
            "total_decisions": 1,
            "successful_words": 1
        }
        self.markov_chain.get_stats = Mock(return_value=updated_stats)

        # Get updated stats
        stats = self.strategy.get_performance_stats()
        self.assertIn('markov', stats)
        self.assertGreater(stats['markov']['success_rate'], 0)
        self.assertEqual(stats['markov']['total_decisions'], 1)
        self.assertEqual(stats['markov']['successful_words'], 1)
        
        # Verify repository updates
        self.markov_repo.record_performance.assert_called_once()

if __name__ == '__main__':
    unittest.main()