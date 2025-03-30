# engine/game_loop.py
# Manages the lifecycle of the game: setup, turn processing, and exit conditions.

import logging
from engine.input_handler import InputHandler
from engine.game_state import GameState
from core.game_events import GameEventManager, EventType

logger = logging.getLogger(__name__)

class GameLoop:
    def __init__(self):
        """
        Initializes the game loop components.
        """
        self.event_manager = GameEventManager()
        self.state = GameState(self.event_manager)
        self.input_handler = InputHandler(self.event_manager)
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
    def _setup_event_subscriptions(self) -> None:
        """Setup all event subscriptions for game loop management"""
        self.event_manager.subscribe(EventType.GAME_QUIT, self._handle_game_quit)
        self.event_manager.subscribe(EventType.BOGGLE_REQUESTED, self._handle_boggle_request)
        self.event_manager.subscribe(EventType.INVALID_WORD, self._handle_invalid_word)
        
    def _handle_game_quit(self, event) -> None:
        """Handle game quit events"""
        logger.info(f"Game quit: {event.data['reason']}")
        self.state.is_game_over = True
        
    def _handle_boggle_request(self, event) -> None:
        """Handle boggle letter redraw requests"""
        self.state.redraw_boggle_letters()
        
    def _handle_invalid_word(self, event) -> None:
        """Handle invalid word submissions"""
        logger.debug(f"Invalid word attempted: {event.data['word']}")

    def start(self):
        """
        Starts the game and loops until the AI wins or the player quits.
        """
        logger.info("Starting game loop.")
        self.state.initialize_game()

        while not self.state.is_game_over:
            self.state.display_status()

            player_word = self.input_handler.get_player_word(self.state)
            if player_word == "QUIT":
                logger.info("Player chose to quit.")
                break
            elif player_word == "BOGGLE":
                continue  # Already handled by event handler

            self.state.process_turn(player_word)

        self.state.display_game_over()
