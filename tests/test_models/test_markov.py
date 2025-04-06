import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.markov_chain import MarkovChain

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.markov_chain = MarkovChain(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            order=2
        )

    def test_initialization(self):
        """Test proper initialization of MarkovChain"""
        self.assertEqual(self.markov_chain.order, 2)
        self.assertEqual(len(self.markov_chain.transitions), 0)
        self.assertEqual(len(self.markov_chain.endings), 0)
        self.assertEqual(self.markov_chain.words_generated, 0)
        self.assertEqual(self.markov_chain.successful_words, 0)

    def test_train_single_word(self):
        """Test training on a single word"""
        test_word = "HELLO"
        self.markov_chain._train_single_word(test_word)

        # Check transitions were created
        self.assertIn("^H", self.markov_chain.transitions)
        self.assertIn("HE", self.markov_chain.transitions)
        self.assertIn("EL", self.markov_chain.transitions)
        self.assertIn("LL", self.markov_chain.transitions)
        self.assertIn("LO", self.markov_chain.transitions)

        # Check ending was stored
        self.assertIn("LO", self.markov_chain.endings)

    def test_train_multiple_words(self):
        """Test training on multiple words"""
        test_words = ["HELLO", "HELP", "HEAP"]
        self.markov_chain.train(test_words)

        # Check common prefix transitions
        self.assertIn("HE", self.markov_chain.transitions)
        self.assertTrue(len(self.markov_chain.transitions["HE"]) >= 3)  # Should have L and A transitions

    def test_generate_word(self):
        """Test word generation"""
        # Train on sample words
        self.markov_chain.train(["HELLO", "HELP", "HEAP"])
        
        # Setup word analyzer mock
        self.word_analyzer.get_word_score.return_value = 1.0
        
        # Test generation with available letters
        available_letters = set("HELPO")
        generated = self.markov_chain.generate_word(
            available_letters=available_letters,
            min_length=3,
            max_length=5
        )

        self.assertIsInstance(generated, str)
        self.assertTrue(set(generated).issubset(available_letters))
        self.assertTrue(3 <= len(generated) <= 5)

    def test_generate_word_insufficient_letters(self):
        """Test generation with insufficient letters"""
        self.markov_chain.train(["HELLO"])
        available_letters = set("ABC")
        
        generated = self.markov_chain.generate_word(
            available_letters=available_letters,
            min_length=3,
            max_length=5
        )

        self.assertEqual(generated, "")

    def test_event_handling(self):
        """Test event handling"""
        # Simulate word submission event
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO"}
        )
        
        self.markov_chain._handle_word_submission(event)
        self.assertIn("HE", self.markov_chain.transitions)

    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        # Train and generate some words
        self.markov_chain.train(["HELLO", "HELP"])
        self.word_analyzer.get_word_score.return_value = 1.0
        
        self.markov_chain.generate_word(set("HELPO"), 3, 5)
        self.markov_chain.generate_word(set("HELPO"), 3, 5)
        
        stats = self.markov_chain.get_performance_stats()
        self.assertIn("words_generated", stats)
        self.assertIn("successful_words", stats)
        self.assertIn("success_rate", stats)

    def test_normalize_probabilities(self):
        """Test probability normalization"""
        self.markov_chain._train_single_word("HELLO")
        self.markov_chain._normalize_probabilities()
        
        # Check probabilities sum to 1 for each state
        for state in self.markov_chain.transitions:
            prob_sum = sum(self.markov_chain.transitions[state].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty word list
        self.markov_chain.train([])
        self.assertEqual(len(self.markov_chain.transitions), 0)
        
        # None available letters
        generated = self.markov_chain.generate_word(set())
        self.assertEqual(generated, "")
        
        # Invalid word
        self.markov_chain._train_single_word("")
        self.assertEqual(len(self.markov_chain.transitions), 0)

if __name__ == '__main__':
    unittest.main()