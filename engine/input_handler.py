# engine/input_handler.py
# Handles player input, including word submission and special commands.

import logging
from collections import Counter
from core.validation.word_validator import WordValidator
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager

logger = logging.getLogger(__name__)

class InputHandler:
    def __init__(self, event_manager: GameEventManager):
        """
        Initialize the input handler with event manager.
        
        Args:
            event_manager: The game's event manager for emitting events
        """
        self.event_manager = event_manager
        self.word_validator = WordValidator(use_nltk=True)
        
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

        while True:
            try:
                user_input = input("\nEnter a word, or type 'boggle' to redraw letters, or 'quit' to end the game: ")
                cleaned_input = user_input.strip().lower()
                logger.debug(f"Player entered: {cleaned_input}")

                if cleaned_input == "quit":
                    self.event_manager.emit(GameEvent(
                        type=EventType.GAME_QUIT,
                        data={"reason": "player_request"}
                    ))
                    return "QUIT"
                elif cleaned_input == "boggle":
                    self.event_manager.emit(GameEvent(
                        type=EventType.BOGGLE_REQUESTED,
                        data={"current_letters": available_letters}
                    ))
                    return "BOGGLE"
                elif not cleaned_input.isalpha():
                    print("Invalid input. Please enter a word using alphabetic characters only.")
                    continue

                # Validate word using WordValidator
                if not self.word_validator.validate_word_with_letters(cleaned_input, available_letters):
                    print(f"'{cleaned_input}' cannot be formed with the current letters.")
                    self.event_manager.emit(GameEvent(
                        type=EventType.INVALID_WORD,
                        data={
                            "word": cleaned_input.upper(),
                            "available_letters": [l.upper() for l in available_letters]
                        }
                    ))
                    continue

                return cleaned_input.upper()

            except (KeyboardInterrupt, EOFError):
                print("\nGame interrupted. Exiting.")
                logger.warning("Game interrupted by user.")
                self.event_manager.emit(GameEvent(
                    type=EventType.GAME_QUIT,
                    data={"reason": "keyboard_interrupt"}
                ))
                return "QUIT"
