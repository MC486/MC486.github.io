# engine/game_loop.py
# Manages the lifecycle of the game: setup, turn processing, and exit conditions.

from typing import Optional
import logging
from engine.game_state import GameState
from engine.input_handler import InputHandler
from core.game_events import EventType
from core.game_events_manager import GameEventManager

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
        try:
            self.state.initialize_game()

            while not self.state.is_game_over:
                # Player's turn
                self.state.display_status()
                print("\n=== Your Turn ===")
                player_word = self.input_handler.get_player_word(self.state)
                if player_word == "QUIT":
                    logger.info("Player chose to quit.")
                    break
                elif player_word == "BOGGLE":
                    continue  # Already handled by event handler

                self.state.process_turn(player_word)
                
                # AI's turn
                print("\n=== AI's Turn ===")
                self.state.process_ai_turn()

        except Exception as e:
            logger.error(f"Unexpected error in game loop: {str(e)}", exc_info=True)
            self.state.is_game_over = True
            raise
        finally:
            self.state.display_game_over()
            
            # Display AI learning statistics
            ai_stats = self.state.get_ai_stats()
            if ai_stats:
                print("\n=== AI Learning Statistics ===")
                for model, stats in ai_stats.items():
                    print(f"\n{model.upper()} Model:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
            
            # Cleanup old entries
            self.state.cleanup()
            
            logger.info("Game loop ended")
