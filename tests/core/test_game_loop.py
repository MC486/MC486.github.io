# tests/test_game_loop.py
# Unit tests for the GameLoop behavior using mock inputs.

import unittest
from unittest.mock import patch, MagicMock, Mock
from engine.game_loop import GameLoop
from core.game_events import EventType

class TestGameLoop(unittest.TestCase):
    def setUp(self):
        """
        Sets up the game loop for each test.
        """
        self.game = GameLoop()
        self.event_manager = self.game.event_manager

    @patch('builtins.input', side_effect=["TestPlayer", "QUIT"])
    def test_game_loop_quit(self, mock_input):
        """
        Test that the game loop exits when player enters 'QUIT'.
        """
        # Mock methods that would normally require user input or game processing
        self.game.state.display_status = MagicMock()
        self.game.state.process_turn = MagicMock()
        self.game.state.display_game_over = MagicMock()

        self.game.start()

        # Verify game state and event handling
        self.assertFalse(self.game.state.is_game_over)  # Game over is set by event handler
        self.game.state.display_game_over.assert_called_once()
        self.event_manager.emit.assert_called_with(
            Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
        )

    @patch('builtins.input', side_effect=["TestPlayer", "BOGGLE", "QUIT"])
    def test_game_loop_boggle_redraw_then_quit(self, mock_input):
        """
        Test that 'BOGGLE' redraws the boggle letters and game quits on 'QUIT'.
        """
        # Mock components to isolate input behavior
        self.game.state.display_status = MagicMock()
        self.game.state.redraw_boggle_letters = MagicMock()
        self.game.state.process_turn = MagicMock()
        self.game.state.display_game_over = MagicMock()

        self.game.start()

        # Verify boggle redraw and game end
        self.game.state.redraw_boggle_letters.assert_called_once()
        self.game.state.display_game_over.assert_called_once()
        
        # Verify event emissions
        self.event_manager.emit.assert_any_call(
            Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': Mock()})
        )
        self.event_manager.emit.assert_any_call(
            Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
        )

    def test_event_handlers(self):
        """
        Test that event handlers are properly registered and called.
        """
        # Create test events
        quit_event = Mock(type=EventType.GAME_QUIT, data={'reason': 'test'})
        boggle_event = Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': []})
        invalid_word_event = Mock(type=EventType.INVALID_WORD, data={'word': 'test'})

        # Call event handlers
        self.game._handle_game_quit(quit_event)
        self.assertTrue(self.game.state.is_game_over)

        self.game._handle_boggle_request(boggle_event)
        self.game.state.redraw_boggle_letters.assert_called_once()

        self.game._handle_invalid_word(invalid_word_event)
        # Add any specific assertions for invalid word handling

if __name__ == '__main__':
    unittest.main()