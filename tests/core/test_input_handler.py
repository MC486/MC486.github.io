# tests/test_input_handler.py
# Unit test for player input handling.

import unittest
import pytest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from engine.input_handler import InputHandler
from engine.game_state import GameState

class TestInputHandler(unittest.TestCase):
    def setUp(self):
        """
        Sets up the input handler for each test.
        """
        self.event_manager = Mock(spec=GameEventManager)
        self.input_handler = InputHandler(self.event_manager)
        self.game_state = Mock()
        self.game_state.shared_letters = ['A', 'T', 'R', 'S']
        self.game_state.boggle_letters = ['E', 'L', 'O', 'P', 'U', 'N']

    def test_input_normalization(self):
        """
        Tests that input is properly normalized (lowercase and stripped whitespace).
        """
        with patch('builtins.input', return_value='  Boggle  '):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'BOGGLE')
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']})
            )

        with patch('builtins.input', return_value=' Quit '):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'QUIT')
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
            )

        with patch('builtins.input', return_value='  Unicorn  '), \
             patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=True):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'unicorn')
            self.event_manager.emit.assert_not_called()

    def test_quit_command(self):
        """
        Tests that the quit command is handled correctly.
        """
        with patch('builtins.input', return_value='quit'):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'QUIT')
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
            )

    def test_boggle_command(self):
        """
        Tests that the boggle command is handled correctly.
        """
        with patch('builtins.input', return_value='boggle'):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'BOGGLE')
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']})
            )

    def test_valid_word(self):
        """
        Tests that a valid word is accepted.
        """
        with patch('builtins.input', return_value='plate'), \
             patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=True):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'plate')
            self.event_manager.emit.assert_not_called()

    def test_invalid_word(self):
        """
        Tests that an invalid word is rejected and appropriate event is emitted.
        """
        with patch('builtins.input', return_value='zebra'), \
             patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=False):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'zebra')  # The word is returned but will be rejected by game state
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.INVALID_WORD, data={
                    'word': 'zebra',
                    'available_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']
                })
            )

    def test_non_alphabetic_input(self):
        """
        Tests that non-alphabetic input is rejected.
        """
        with patch('builtins.input', return_value='123'):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, '123')  # The input is returned but will be rejected by game state
            self.event_manager.emit.assert_not_called()

    def test_keyboard_interrupt(self):
        """
        Tests that keyboard interrupt is handled correctly.
        """
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = self.input_handler.get_player_word(self.game_state)
            self.assertEqual(result, 'QUIT')
            self.event_manager.emit.assert_called_with(
                Mock(type=EventType.GAME_QUIT, data={'reason': 'keyboard_interrupt'})
            )

if __name__ == '__main__':
    unittest.main()