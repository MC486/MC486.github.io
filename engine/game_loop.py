# engine/game_loop.py
# Manages the lifecycle of the game: setup, turn processing, and exit conditions.

import logging
from engine.input_handler import InputHandler
from engine.game_state import GameState

logger = logging.getLogger(__name__)

class GameLoop:
    def __init__(self):
        """
        Initializes the game loop components.
        """
        self.input_handler = InputHandler()
        self.state = GameState()

    def start(self):
        """
        Starts the game and loops until the AI wins or the player quits.
        """
        logger.info("Starting game loop.")
        self.state.initialize_game()

        while not self.state.is_game_over:
            self.state.display_status()

            player_word = self.input_handler.get_player_word(self.state)
            if player_word == "BOGGLE":
                self.state.redraw_boggle_letters()
                continue
            elif player_word == "QUIT":
                logger.info("Player chose to quit.")
                break

            self.state.process_turn(player_word)

        self.state.display_game_over()
