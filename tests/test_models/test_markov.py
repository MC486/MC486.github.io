import unittest
from unittest.mock import Mock, patch
import numpy as np
from ai.markov_chain import MarkovChain
from database.repositories.markov_repository import MarkovRepository
from database.manager import DatabaseManager

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        # Create a mock repository
        self.repository = Mock(spec=MarkovRepository)
        self.markov_chain = MarkovChain(order=2, repository=self.repository)

    def test_initialization(self):
        """Test proper initialization of MarkovChain"""
        self.assertEqual(self.markov_chain.order, 2)
        self.assertFalse(self.markov_chain.is_trained)
        self.assertIsNotNone(self.markov_chain.repository)

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
            self.markov_chain.generate_word(max_length=2, min_length=3)

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

        result = self.markov_chain.generate_word(max_length=5, min_length=3)
        self.assertIsNotNone(result)
        self.assertTrue(3 <= len(result) <= 5)

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
        self.repository.get_state_probabilities.assert_called_once_with("he")

if __name__ == '__main__':
    unittest.main()