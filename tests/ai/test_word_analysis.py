# tests/ai/test_word_analysis.py
# Unit tests for word frequency analysis and pattern recognition.

import unittest
from unittest.mock import Mock, patch
from ai.word_analysis import WordFrequencyAnalyzer
from core.game_events import GameEvent, EventType, GameEventManager

class TestWordFrequencyAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up the analyzer for each test."""
        self.event_manager = Mock(spec=GameEventManager)
        self.analyzer = WordFrequencyAnalyzer(self.event_manager)

    def test_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer.word_validator)
        self.assertEqual(self.analyzer.total_words, 0)
        self.assertEqual(self.analyzer.total_letters, 0)

    def test_word_list_analysis(self):
        """Test analysis of a list of words."""
        test_words = ["apple", "banana", "cherry"]
        self.analyzer.analyze_word_list(test_words)
        
        # Check word length frequencies
        self.assertEqual(self.analyzer.word_lengths[5], 1)  # "apple"
        self.assertEqual(self.analyzer.word_lengths[6], 2)  # "banana", "cherry"
        
        # Check letter frequencies
        self.assertEqual(self.analyzer.letter_frequencies['A'], 4)  # In "apple", "banana"
        self.assertEqual(self.analyzer.letter_frequencies['P'], 2)  # In "apple"

    def test_invalid_word_handling(self):
        """Test that invalid words are properly handled."""
        test_words = ["apple", "invalid123", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Only valid words should be counted
        self.assertEqual(self.analyzer.total_words, 2)
        self.assertNotIn(8, self.analyzer.word_lengths)  # Length of "invalid123"

    def test_letter_probability_calculation(self):
        """Test letter probability calculations."""
        test_words = ["apple", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Check probabilities
        prob_a = self.analyzer.get_letter_probability('A')
        self.assertGreater(prob_a, 0)
        self.assertLessEqual(prob_a, 1)

    def test_next_letter_probability(self):
        """Test next letter probability calculations."""
        test_words = ["apple", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Check transition probabilities
        prob_p_after_a = self.analyzer.get_next_letter_probability('A', 'P')
        self.assertGreater(prob_p_after_a, 0)
        self.assertLessEqual(prob_p_after_a, 1)

    def test_position_probability(self):
        """Test position-based letter probability calculations."""
        test_words = ["apple", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Check position probabilities
        prob_a_first = self.analyzer.get_position_probability('A', 0)
        self.assertGreater(prob_a_first, 0)
        self.assertLessEqual(prob_a_first, 1)

    def test_word_score_calculation(self):
        """Test word score calculations."""
        test_words = ["apple", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Check word scores
        score_apple = self.analyzer.get_word_score("apple")
        self.assertGreater(score_apple, 0)
        
        # Invalid word should score 0
        score_invalid = self.analyzer.get_word_score("invalid123")
        self.assertEqual(score_invalid, 0)

    def test_event_handling(self):
        """Test event handling for word submissions."""
        test_event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "apple"}
        )
        
        self.analyzer._handle_word_submission(test_event)
        
        # Check that event was processed
        self.assertEqual(self.analyzer.total_words, 1)
        self.assertGreater(self.analyzer.total_letters, 0)
        
        # Check that update event was emitted
        self.event_manager.emit.assert_called_with(
            Mock(type=EventType.MODEL_STATE_UPDATE)
        )

    def test_game_start_reset(self):
        """Test that analysis data is reset on game start."""
        # First add some data
        test_words = ["apple", "banana"]
        self.analyzer.analyze_word_list(test_words)
        
        # Reset on game start
        self.analyzer._handle_game_start(Mock())
        
        # Check that all data is cleared
        self.assertEqual(self.analyzer.total_words, 0)
        self.assertEqual(self.analyzer.total_letters, 0)
        self.assertEqual(len(self.analyzer.letter_frequencies), 0)
        self.assertEqual(len(self.analyzer.word_lengths), 0)
        self.assertEqual(len(self.analyzer.letter_pairs), 0)
        self.assertEqual(len(self.analyzer.position_frequencies), 0)

    def test_empty_word_list(self):
        """Test handling of empty word list."""
        self.analyzer.analyze_word_list([])
        self.assertEqual(self.analyzer.total_words, 0)
        self.assertEqual(self.analyzer.total_letters, 0)

    def test_single_letter_words(self):
        """Test handling of single letter words."""
        test_words = ["a", "i"]
        self.analyzer.analyze_word_list(test_words)
        self.assertEqual(self.analyzer.word_lengths[1], 2)
        self.assertEqual(self.analyzer.letter_frequencies['A'], 1)
        self.assertEqual(self.analyzer.letter_frequencies['I'], 1)
