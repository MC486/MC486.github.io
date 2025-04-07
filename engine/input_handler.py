# engine/input_handler.py
# Handles player input, including word submission and special commands.

import logging
from collections import Counter
from core.validation.word_validator import WordValidator
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from database.repositories.word_repository import WordRepository
from database.repositories.category_repository import CategoryRepository
from database.manager import DatabaseManager
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class InputHandler:
    """Handles player input and validation."""
    
    def __init__(self, event_manager: GameEventManager, word_repo: WordRepository, category_repo: CategoryRepository):
        """Initialize the input handler.
        
        Args:
            event_manager (GameEventManager): Event manager for game events
            word_repo: Repository for word usage data
            category_repo: Repository for word categories
        """
        self.event_manager = event_manager
        self.word_repo = word_repo
        self.category_repo = category_repo
        self.word_validator = WordValidator(word_repo)
        self.command_pattern = re.compile(r'^/(\w+)(?:\s+(.+))?$')
        
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

    def process_input(self, user_input: str, current_category: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Process user input and determine the appropriate action.

        Args:
            user_input: The user's input string
            current_category: The current game category (if any)

        Returns:
            Tuple containing action type and relevant data
        """
        if not user_input:
            return 'invalid', {'message': 'Empty input'}

        user_input = user_input.strip()

        # Check if input is a command
        if user_input.startswith('/'):
            return self._process_command(user_input)

        # Process as a word guess
        return self._process_word(user_input, current_category)

    def _process_command(self, command_input: str) -> Tuple[str, Dict]:
        """
        Process a command input.

        Args:
            command_input: The command string

        Returns:
            Tuple containing action type and command data
        """
        match = self.command_pattern.match(command_input)
        if not match:
            return 'invalid', {'message': 'Invalid command format'}

        command, args = match.groups()
        command = command.lower()

        if command == 'help':
            return 'help', self._get_help_data()
        elif command == 'stats':
            return 'stats', self._get_stats_data()
        elif command == 'category':
            if not args:
                return 'category_list', {'categories': self.category_repo.get_categories()}
            return 'category_select', {'category': args}
        elif command == 'quit':
            return 'quit', {}
        else:
            return 'invalid', {'message': f'Unknown command: {command}'}

    def _process_word(self, word: str, category: Optional[str]) -> Tuple[str, Dict]:
        """
        Process a word guess.

        Args:
            word: The word to process
            category: The current category (if any)

        Returns:
            Tuple containing action type and word data
        """
        word = word.strip().lower()

        if not self.word_validator.validate_word(word, category):
            return 'invalid_word', {
                'word': word,
                'message': 'Invalid word',
                'category': category
            }

        # Record word usage
        self.word_repo.record_word_usage(word)

        return 'valid_word', {
            'word': word,
            'stats': self.word_repo.get_word_stats(word)
        }

    def _get_help_data(self) -> Dict:
        """
        Get help information.

        Returns:
            Dictionary containing help data
        """
        return {
            'commands': {
                '/help': 'Show this help message',
                '/stats': 'Show game statistics',
                '/category': 'List available categories',
                '/category <name>': 'Select a category',
                '/quit': 'Exit the game'
            },
            'rules': [
                'Enter words that match the current category',
                'Words must be 3-15 letters long',
                'Only letters A-Z are allowed',
                'Each valid word earns points based on length and letter values'
            ]
        }

    def _get_stats_data(self) -> Dict:
        """
        Get game statistics.

        Returns:
            Dictionary containing statistics data
        """
        total_words = self.word_repo.get_entry_count()
        categories = self.category_repo.get_categories()
        category_stats = {}

        for category in categories:
            stats = self.category_repo.get_category_stats(category)
            category_stats[category] = {
                'total_words': stats.get('word_count', 0),
                'unique_words': stats.get('unique_words', 0),
                'average_length': stats.get('average_length', 0)
            }

        return {
            'total_words': total_words,
            'total_categories': len(categories),
            'categories': category_stats
        }
