import unittest
from unittest.mock import Mock, patch
import numpy as np
from ai.models.markov_chain import MarkovChain
from database.repositories.markov_repository import MarkovRepository
from database.manager import DatabaseManager
from core.game_events_manager import GameEventManager
from core.validation.trie import Trie
from ai.word_analysis import WordFrequencyAnalyzer

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        # Create mock dependencies
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.trie = Mock(spec=Trie)
        self.repository = Mock(spec=MarkovRepository)
        
        # Set up required mock methods
        self.repository.get_state_probabilities = Mock(return_value={})
        self.repository.get_transitions = Mock(return_value={})
        self.repository.record_transition = Mock()
        self.repository.bulk_update_transitions = Mock()
        self.word_analyzer.get_analyzed_words = Mock(return_value=set())
        self.trie.search = Mock(return_value=True)
        
        # Create MarkovChain instance
        self.markov_chain = MarkovChain(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            trie=self.trie,
            markov_repository=self.repository,
            order=2
        )

    def test_initialization(self):
        """Test proper initialization of MarkovChain"""
        self.assertEqual(self.markov_chain.order, 2)
        self.assertFalse(self.markov_chain.is_trained)
        self.assertIsNotNone(self.markov_chain.markov_repository)

    def test_initialization_invalid_order(self):
        """Test initialization with invalid order"""
        with self.assertRaises(ValueError):
            MarkovChain(order=0)

    def test_train_without_repository(self):
        """Test training without repository set"""
        chain = MarkovChain(order=2)  # No repository
        with self.assertRaises(RuntimeError):
            chain.train(["HELLO"])

    def test_train_empty_list(self):
        """Test training with empty word list"""
        with self.assertRaises(ValueError):
            self.markov_chain.train([])

    def test_train_invalid_words(self):
        """Test training with invalid words"""
        with self.assertRaises(ValueError):
            self.markov_chain.train(["123", "!@#"])

    def test_train_single_word(self):
        """Test training on a single word"""
        test_word = "HELLO"
        self.markov_chain.train([test_word])

        # Verify repository calls
        self.repository.record_transition.assert_any_call("START", "HE")
        self.repository.record_transition.assert_any_call("HE", "L")
        self.repository.record_transition.assert_any_call("EL", "L")
        self.repository.record_transition.assert_any_call("LL", "O")
        self.assertTrue(self.markov_chain.is_trained)

    def test_train_multiple_words(self):
        """Test training on multiple words"""
        test_words = ["HELLO", "HELP", "HEAP"]
        self.markov_chain.train(test_words)

        # Verify repository calls for common prefixes
        self.repository.record_transition.assert_any_call("START", "HE")
        self.repository.record_transition.assert_any_call("HE", "L")
        self.repository.record_transition.assert_any_call("HE", "A")
        self.assertTrue(self.markov_chain.is_trained)

    def test_generate_word_without_repository(self):
        """Test word generation without repository"""
        chain = MarkovChain(order=2)  # No repository
        with self.assertRaises(RuntimeError):
            chain.generate_word()

    def test_generate_word_not_trained(self):
        """Test word generation without training"""
        with self.assertRaises(RuntimeError):
            self.markov_chain.generate_word()

    def test_generate_word_invalid_length(self):
        """Test word generation with invalid length parameters"""
        self.markov_chain.is_trained = True
        with self.assertRaises(ValueError):
            self.markov_chain.generate_word(available_letters="A")  # Not enough letters for 3-letter word

    def test_generate_word_no_start_states(self):
        """Test word generation with no available start states"""
        self.markov_chain.is_trained = True
        self.repository.get_state_probabilities.return_value = None
        result = self.markov_chain.generate_word()
        self.assertIsNone(result)

    def test_generate_word_success(self):
        """Test successful word generation"""
        self.markov_chain.is_trained = True
        
        # Mock repository responses
        self.repository.get_state_probabilities.side_effect = [
            {"HE": 1.0},  # Start state
            {"L": 0.5, "A": 0.5},  # First transition
            {"L": 1.0},  # Second transition
            {"O": 1.0},  # Third transition
            None  # End generation
        ]
        
        # Mock transitions
        self.repository.get_transitions.return_value = {
            "HE": {"L": 0.5, "A": 0.5},
            "EL": {"L": 1.0},
            "LL": {"O": 1.0}
        }
        
        # Mock trie to accept any word
        self.trie.search.return_value = True

        result = self.markov_chain.generate_word(available_letters={"H", "E", "L", "O"})
        self.assertIsNotNone(result)
        self.assertTrue(3 <= len(result) <= 15)

    def test_get_state_probabilities_without_repository(self):
        """Test getting probabilities without repository"""
        chain = MarkovChain(order=2)  # No repository
        with self.assertRaises(RuntimeError):
            chain.get_state_probabilities("HE")

    def test_get_state_probabilities_not_trained(self):
        """Test getting probabilities without training"""
        with self.assertRaises(RuntimeError):
            self.markov_chain.get_state_probabilities("HE")

    def test_get_state_probabilities_success(self):
        """Test successful probability retrieval"""
        self.markov_chain.is_trained = True
        expected_probs = {"L": 0.5, "A": 0.5}
        self.repository.get_state_probabilities.return_value = expected_probs
        
        result = self.markov_chain.get_state_probabilities("HE")
        self.assertEqual(result, expected_probs)
        self.repository.get_state_probabilities.assert_called_once_with("HE")

if __name__ == '__main__':
    unittest.main()