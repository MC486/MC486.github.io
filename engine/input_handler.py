# engine/input_handler.py
# Handles player input, including word submission and special commands.

import logging
from collections import Counter

logger = logging.getLogger(__name__)

class InputHandler:
    def get_player_word(self, game_state):
        """
        Prompts the player for a valid word or special command ('boggle', 'quit').
        Ensures all letters in the word are available in the current letter pool.

        Args:
            game_state (GameState): The current game state object containing letter pools.

        Returns:
            str: A valid player word or command.
        """
        shared = game_state.shared_letters
        boggle = game_state.boggle_letters

        available_letters = shared + boggle
        available_counter = Counter(available_letters)

        while True:
            try:
                user_input = input("\nEnter a word, or type 'boggle' to redraw letters, or 'quit' to end the game: ")
                cleaned_input = user_input.strip().lower()
                logger.debug(f"Player entered: {cleaned_input}")

                if cleaned_input == "quit":
                    return "QUIT"
                elif cleaned_input == "boggle":
                    return "BOGGLE"
                elif not cleaned_input.isalpha():
                    print("Invalid input. Please enter a word using alphabetic characters only.")
                    continue

                word_counter = Counter(cleaned_input)

                # Check that the word can be made from available letters
                for letter, count in word_counter.items():
                    if available_counter[letter.upper()] < count:
                        print(f"'{cleaned_input}' cannot be formed with the current letters.")
                        break
                else:
                    return cleaned_input  # Valid word

            except (KeyboardInterrupt, EOFError):
                print("\nGame interrupted. Exiting.")
                logger.warning("Game interrupted by user.")
                return "QUIT"
