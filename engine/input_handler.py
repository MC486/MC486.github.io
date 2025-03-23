# engine/input_handler.py
# Handles player input, including word submission and special commands.

import logging

logger = logging.getLogger(__name__)

class InputHandler:
    def get_player_word(self, game_state):
        """
        Prompts the player for a word or command, such as 'boggle' or 'quit'.
        
        Args:
            game_state (GameState): The current game state object, used for context or validation if needed.
        
        Returns:
            str: The player's input, normalized.
        """
        try:
            user_input = input("\nEnter a word, or type 'boggle' to redraw letters, or 'quit' to end the game: ")
            cleaned_input = user_input.strip().lower()
            logger.debug(f"Player entered: {cleaned_input}")
            return cleaned_input
        except (KeyboardInterrupt, EOFError):
            print("\nGame interrupted. Exiting.")
            logger.warning("Game interrupted by user.")
            return "quit"
