from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager

@dataclass
class TurnData:
    """Represents data for a single turn in the game"""
    word: str
    score: int
    letters_used: Set[str]
    shared_letters: List[str]
    private_letters: List[str]
    turn_number: int
    timestamp: datetime = datetime.now()

@dataclass
class GameRecord:
    """Stores complete record of a game"""
    game_id: str
    start_time: datetime
    end_time: Optional[datetime]
    player_name: str
    final_score: int
    turns: List[TurnData]
    difficulty: str
    total_words_played: int = 0

class GameHistoryTracker:
    """
    Tracks and stores game history for AI training purposes.
    Captures detailed turn-by-turn data and game outcomes.
    """
    def __init__(self, event_manager: GameEventManager):
        self.event_manager = event_manager
        self.current_game: Optional[GameRecord] = None
        self.game_history: List[GameRecord] = []
        
        # Subscribe to game events
        self._setup_event_subscriptions()
        
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for game tracking"""
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.GAME_END, self._handle_game_end)
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start event"""
        self.current_game = GameRecord(
            game_id=f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            end_time=None,
            player_name=event.data.get("player_name", "unknown"),
            final_score=0,
            turns=[],
            difficulty=event.data.get("difficulty", "medium")
        )
        
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={"message": "New game tracking started"},
            debug_data={"game_id": self.current_game.game_id}
        ))

    def _handle_game_end(self, event: GameEvent) -> None:
        """Handle game end event"""
        if self.current_game:
            self.current_game.end_time = datetime.now()
            self.current_game.final_score = event.data.get("final_score", 0)
            self.game_history.append(self.current_game)
            self.current_game = None
            
            self.event_manager.emit(GameEvent(
                type=EventType.MODEL_STATE_UPDATE,
                data={"message": "Game history updated"},
                debug_data={"total_games": len(self.game_history)}
            ))

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission event"""
        if not self.current_game:
            return
            
        turn_data = TurnData(
            word=event.data["word"],
            score=event.data["score"],
            letters_used=set(event.data["word"].upper()),
            shared_letters=event.data.get("shared_letters", []),
            private_letters=event.data.get("private_letters", []),
            turn_number=len(self.current_game.turns) + 1
        )
        
        self.current_game.turns.append(turn_data)
        self.current_game.total_words_played += 1

    def get_game_history(self) -> List[GameRecord]:
        """Get complete game history"""
        return self.game_history.copy()

    def get_current_game_state(self) -> Optional[Dict]:
        """Get current game state summary"""
        if not self.current_game:
            return None
            
        return {
            "game_id": self.current_game.game_id,
            "player_name": self.current_game.player_name,
            "current_turn": len(self.current_game.turns),
            "words_played": self.current_game.total_words_played,
            "running_time": (datetime.now() - self.current_game.start_time).total_seconds()
        }

    def clear_history(self) -> None:
        """Clear all game history"""
        self.game_history.clear()
        self.current_game = None