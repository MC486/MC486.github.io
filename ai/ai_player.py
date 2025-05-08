from typing import Dict, Set, Optional, List, Any
import logging
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from database.manager import DatabaseManager
from ai.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

class AIPlayer:
    """
    Main AI player interface that interacts with the game.
    Manages AI state, decision making, and game interactions.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 db_manager: DatabaseManager,
                 strategy: Optional[AIStrategy] = None):
        self.event_manager = event_manager
        self.db_manager = db_manager
        self.strategy = strategy or AIStrategy(
            event_manager=event_manager,
            db_manager=db_manager,
            word_repo=db_manager.get_word_repository(),
            category_repo=db_manager.get_category_repository()
        )
        
        # Game state tracking
        self.score: int = 0
        self.word_history: List[str] = []
        self.valid_words: List[str] = []  # Track valid words separately
        self.turn_number: int = 0
        self.difficulty: str = "medium"
        
        # Get repositories
        self.word_repo = db_manager.get_word_usage_repository()
        
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for game interaction"""
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)
        self.event_manager.subscribe(EventType.WORD_VALIDATED, self._handle_word_validation)

    def make_move(self, available_letters: List[str]) -> str:
        """
        Make a move by selecting a word based on available letters.
        
        Args:
            available_letters: List of available letters to form words
            
        Returns:
            Selected word
        """
        if not available_letters:
            return ''
            
        # Get word from strategy
        word = self.strategy.select_word(set(available_letters), set(), self.turn_number)
        
        # Avoid duplicate words
        while word and word in self.word_history:
            word = self.strategy.select_word(set(available_letters), set(), self.turn_number)
            
        if word:
            self.word_repo.record_word_usage(word)
            
        return word

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start event"""
        self.difficulty = event.data.get("difficulty", self.difficulty)
        self.score = 0
        self.turn_number = 0
        self.word_history.clear()
        self.valid_words.clear()
        self.strategy.reset()
        self.word_repo.cleanup_old_entries()

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Handle turn start event"""
        logger.debug(f"Handling turn start event: {event.data}")
        if event.data.get("player") == "ai":
            logger.debug("AI player's turn")
            self.turn_number = event.data.get("turn_number", self.turn_number + 1)
            available_letters = event.data.get("available_letters", [])
            logger.debug(f"Available letters: {available_letters}")
            word = self.make_move(available_letters)
            logger.debug(f"Selected word: {word}")
            self.event_manager.emit(GameEvent(
                type=EventType.WORD_SUBMITTED,
                data={"word": word, "player": "ai"}
            ))

    def _handle_word_validation(self, event: GameEvent) -> None:
        """Handle word validation event"""
        if event.data.get("player") == "ai":
            word = event.data.get("word", "")
            if word:  # Add word to history regardless of validity
                self.word_history.append(word)
            if event.data.get("is_valid", False):
                self.score += event.data.get("score", 0)
                self.valid_words.append(word)
            self.word_repo.record_word_usage(word)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get AI player performance statistics"""
        stats = {
            "total_words": len(self.word_history),
            "valid_words": len(self.valid_words),
            "total_score": self.score
        }
        stats.update(self.word_repo.get_word_stats())
        return stats

    def _get_event_handler(self, event_type: EventType):
        """Get the appropriate event handler for the given event type"""
        handlers = {
            EventType.GAME_START: self._handle_game_start,
            EventType.TURN_START: self._handle_turn_start,
            EventType.WORD_VALIDATED: self._handle_word_validation
        }
        return handlers.get(event_type)

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start event"""
        self.difficulty = event.data.get("difficulty", self.difficulty)
        self.score = 0
        self.turn_number = 0
        self.word_history.clear()
        self.valid_words.clear()
        self.strategy.reset()
        self.word_repo.cleanup_old_entries()

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Handle turn start event"""
        logger.debug(f"Handling turn start event: {event.data}")
        if event.data.get("player") == "ai":
            logger.debug("AI player's turn")
            self.turn_number = event.data.get("turn_number", self.turn_number + 1)
            available_letters = event.data.get("available_letters", [])
            logger.debug(f"Available letters: {available_letters}")
            word = self.make_move(available_letters)
            logger.debug(f"Selected word: {word}")
            self.event_manager.emit(GameEvent(
                type=EventType.WORD_SUBMITTED,
                data={"word": word, "player": "ai"}
            ))

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission event"""
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        player_id = event.data.get("player_id", "")
        
        if player_id == "ai":
            self.score += score

    def _handle_game_end(self, event: GameEvent) -> None:
        """Handle game end event"""
        final_score = event.data.get("ai_score", self.score)
        opponent_score = event.data.get("human_score", 0)
        
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={
                "message": "Game completed",
                "final_score": final_score,
                "opponent_score": opponent_score
            },
            debug_data=self.get_stats()
        ))

    def get_stats(self) -> Dict:
        """Get AI player statistics"""
        return {
            "score": self.score,
            "turns_played": self.turn_number,
            "words_used": len(self.word_history),
            "difficulty": self.difficulty,
            "strategy_stats": self.strategy.get_stats()
        }

    def set_difficulty(self, difficulty: str) -> None:
        """
        Set AI difficulty level.
        
        Args:
            difficulty: New difficulty level
        """
        self.difficulty = difficulty
        self.strategy = AIStrategy(
            self.event_manager,
            self.word_analyzer,
            self.valid_words,
            difficulty
        )