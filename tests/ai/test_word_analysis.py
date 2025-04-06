import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer

class TestWordFrequencyAnalyzer(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.analyzer = WordFrequencyAnalyzer(self.event_manager)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.analyzer.total_letters, 0)
        self.assertEqual(len(self.analyzer.letter_frequencies), 0)
        self.assertEqual(len(self.analyzer.word_lengths), 0)
        self.assertEqual(len(self.analyzer.letter_pairs), 0)
        self.assertEqual(len(self.analyzer.position_frequencies), 0)

    def test_analyze_single_word(self):
        """Test analysis of a single word"""
        self.analyzer._analyze_single_word("HELLO")
        
        # Check letter frequencies
        self.assertEqual(self.analyzer.letter_frequencies['H'], 1)
        self.assertEqual(self.analyzer.letter_frequencies['E'], 1)
        self.assertEqual(self.analyzer.letter_frequencies['L'], 2)
        self.assertEqual(self.analyzer.letter_frequencies['O'], 1)
        
        # Check word length
        self.assertEqual(self.analyzer.word_lengths[5], 1)
        
        # Check letter pairs
        self.assertIn('H', self.analyzer.letter_pairs)
        self.assertEqual(self.analyzer.letter_pairs['H']['E'], 1)
        
        # Check position frequencies
        self.assertEqual(self.analyzer.position_frequencies[0]['H'], 1)
        self.assertEqual(self.analyzer.position_frequencies[1]['E'], 1)

    def test_analyze_word_list(self):
        """Test analysis of multiple words"""
        words = ["HELLO", "HELP", "HEAP"]
        self.analyzer.analyze_word_list(words)
        
        # Check common patterns
        self.assertEqual(self.analyzer.letter_frequencies['H'], 3)
        self.assertEqual(self.analyzer.letter_frequencies['E'], 3)
        self.assertEqual(self.analyzer.word_lengths[4], 3)

    def test_probability_calculations(self):
        """Test probability calculations"""
        self.analyzer.analyze_word_list(["HELLO", "HELP"])
        self.analyzer._calculate_probabilities()
        
        # Test letter probabilities
        self.assertGreater(self.analyzer.get_letter_probability('H'), 0)
        self.assertLessEqual(self.analyzer.get_letter_probability('H'), 1)
        
        # Test position probabilities
        self.assertGreater(self.analyzer.get_position_probability('H', 0), 0)
        self.assertLessEqual(self.analyzer.get_position_probability('H', 0), 1)

    def test_word_scoring(self):
        """Test word scoring functionality"""
        self.analyzer.analyze_word_list(["HELLO", "HELP", "HEAP"])
        
        # Test known word
        score_hello = self.analyzer.get_word_score("HELLO")
        self.assertGreater(score_hello, 0)
        
        # Test unknown word
        score_unknown = self.analyzer.get_word_score("XYZ")
        self.assertGreaterEqual(score_unknown, 0)

    def test_next_letter_probability(self):
        """Test letter transition probabilities"""
        self.analyzer.analyze_word_list(["HELLO", "HELP"])
        
        # Test known transition
        prob_he = self.analyzer.get_next_letter_probability('H', 'E')
        self.assertGreater(prob_he, 0)
        
        # Test unknown transition
        prob_unknown = self.analyzer.get_next_letter_probability('X', 'Y')
        self.assertEqual(prob_unknown, 0)

    def test_event_handling(self):
        """Test event system integration"""
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO"}
        )
        
        self.analyzer._handle_word_submission(event)
        self.assertEqual(self.analyzer.total_words, 1)
        self.assertEqual(self.analyzer.total_letters, 5)

    def test_game_start_reset(self):
        """Test reset on game start"""
        self.analyzer.analyze_word_list(["HELLO"])
        
        event = GameEvent(
            type=EventType.GAME_START,
            data={}
        )
        
        self.analyzer._handle_game_start(event)
        self.assertEqual(self.analyzer.total_words, 0)
        self.assertEqual(self.analyzer.total_letters, 0)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty word
        self.analyzer._analyze_single_word("")
        self.assertEqual(self.analyzer.total_words, 0)
        
        # None word
        self.analyzer._analyze_single_word(None)
        self.assertEqual(self.analyzer.total_words, 0)
        
        # Invalid position
        prob = self.analyzer.get_position_probability('A', -1)
        self.assertEqual(prob, 0)

    def test_case_insensitivity(self):
        """Test case insensitive analysis"""
        self.analyzer._analyze_single_word("Hello")
        self.analyzer._analyze_single_word("HELLO")
        
        # Should treat both words as same
        self.assertEqual(self.analyzer.total_words, 2)
        self.assertEqual(self.analyzer.letter_frequencies['H'], 2)

if __name__ == '__main__':
    unittest.main()