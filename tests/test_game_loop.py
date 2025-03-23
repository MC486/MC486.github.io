# tests/test_game_loop.py
# Unit tests for the GameLoop behavior using mock inputs.

import unittest
from unittest.mock import patch, MagicMock
from engine.game_loop import GameLoop

class TestGameLoop(unittest.TestCase):

    @patch('builtins.input', side_effect=["TestPlayer", "QUIT"])
    def test_game_loop_quit(self, mock_input):
        """
        Test that the game loop exits when player enters 'QUIT'.
        """
        game = GameLoop() # Create an instance of the GameLoop.

        # Mock methods that would normally require user input or game processing
        game.state.display_status = MagicMock() # Replace display_status with a mock.
        game.state.process_turn = MagicMock() # Replace process_turn with a mock.
        game.state.display_game_over = MagicMock() # Replace display_game_over with a mock.

        game.start() # Start the game loop.

        # Ensure game_over is not triggered internally
        self.assertFalse(game.state.is_game_over) # Verify that game_over is false.
        game.state.display_game_over.assert_called_once() # Verify display_game_over was called once.

    @patch('builtins.input', side_effect=["TestPlayer", "BOGGLE", "QUIT"])
    def test_game_loop_boggle_redraw_then_quit(self, mock_input):
        """
        Test that 'BOGGLE' redraws the boggle letters and game quits on 'QUIT'.
        """
        game = GameLoop() # Create an instance of the GameLoop.

        # Mock components to isolate input behavior
        game.state.display_status = MagicMock() # Replace display_status with a mock.
        game.state.redraw_boggle_letters = MagicMock() # Replace redraw_boggle_letters with a mock.
        game.state.process_turn = MagicMock() # Replace process_turn with a mock.
        game.state.display_game_over = MagicMock() # Replace display_game_over with a mock.

        game.start() # Start the game loop.

        game.state.redraw_boggle_letters.assert_called_once() # Verify redraw_boggle_letters was called once.
        game.state.display_game_over.assert_called_once() # Verify display_game_over was called once.

if __name__ == '__main__':
    unittest.main() # Run the unit tests.