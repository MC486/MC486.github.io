# engine/input_handler.py
# Handles user input for gameplay, including words and commands.

import logging

logger = logging.getLogger(__name__)

def get_player_input():
    """
    Prompts the player for a word or a command like 'boggle' or 'quit'.

    Returns:
        str: The player's input in lowercase and stripped of whitespace.
    """
    try:
        user_input = input("\nEnter a word, or type 'boggle' to redraw letters, or 'quit' to end the game: ").strip().lower()
        logger.debug(f"Player entered: {user_input}") # Log the player's input for debugging.
        return user_input
    except (KeyboardInterrupt, EOFError): # Catch keyboard interrupts (Ctrl+C) and end-of-file errors.
        print("\nGame interrupted. Exiting.")
        logger.warning("Game interrupted by user.") # Log the interruption as a warning.
        return 'quit' # Return 'quit' to signal game termination.