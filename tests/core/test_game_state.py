# tests/test_game_state.py
# Unit tests for game state logic.

import unittest
from unittest.mock import Mock, patch
from engine.game_state import GameState
from core.game_events import GameEventManager

class TestGameState(unittest.TestCase):
    def setUp(self):
        """
        Sets up the game state for each test.
        """
        self.event_manager = Mock(spec=GameEventManager)
        self.state = GameState(self.event_manager)
        self.state.shared_letters = ['A', 'T', 'R', 'S']
        self.state.boggle_letters = ['E', 'L', 'O', 'P', 'U', 'N']

    def test_word_scoring_and_storage(self):
        """
        Tests that a word is scored correctly and stored in used words.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            mock_validate.return_value = True
            self.state.process_turn("LOPE")
            self.assertGreater(self.state.human_player.score, 0)
            self.assertIn("LOPE", self.state.human_player.used_words)

    def test_repeat_word_penalty(self):
        """
        Tests that repeated words are penalized, resulting in a lower score.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            mock_validate.return_value = True
            self.state.process_turn("LOPE")
            score_1 = self.state.human_player.score
            self.state.process_turn("LOPE")
            score_2 = self.state.human_player.score
            self.assertLess(score_2 - score_1, score_1)

    def test_word_usage_count_increment(self):
        """
        Tests that word usage counts are correctly incremented.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            mock_validate.return_value = True
            self.state.process_turn("LOPE")
            self.assertEqual(self.state.human_player.word_usage_counts["LOPE"], 1)
            self.state.process_turn("LOPE")
            self.assertEqual(self.state.human_player.word_usage_counts["LOPE"], 2)

    def test_invalid_word_rejected(self):
        """
        Tests that invalid words are not accepted or scored.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            mock_validate.return_value = False
            self.state.process_turn("INVALIDWORD")
            self.assertNotIn("INVALIDWORD", self.state.human_player.used_words)
            self.assertEqual(self.state.human_player.score, 0)

    def test_word_validation_with_available_letters(self):
        """
        Tests that word validation considers available letters.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            # Test with valid letters
            mock_validate.return_value = True
            self.state.process_turn("PLATE")
            self.assertIn("PLATE", self.state.human_player.used_words)
            
            # Test with invalid letters
            mock_validate.return_value = False
            self.state.process_turn("ZEBRA")
            self.assertNotIn("ZEBRA", self.state.human_player.used_words)

    def test_event_emission(self):
        """
        Tests that events are emitted correctly during gameplay.
        """
        with patch('core.validation.word_validator.WordValidator.validate_word_with_letters') as mock_validate:
            mock_validate.return_value = True
            self.state.process_turn("LOPE")
            self.event_manager.emit.assert_called()

if __name__ == "__main__":
    unittest.main()