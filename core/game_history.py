# core/game_history.py
# Tracks and manages game history, including turns, scores, and events.

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from .game_events import GameEvent

logger = logging.getLogger(__name__)

@dataclass
class Turn:
    """
    Represents a single turn in the game.
    
    Attributes:
        player_word (str): Word submitted by the player
        player_score (int): Score earned for the word
        ai_word (Optional[str]): Word submitted by AI (if applicable)
        ai_score (Optional[int]): Score earned by AI (if applicable)
        timestamp (datetime): When the turn occurred
        events (List[GameEvent]): Events that occurred during the turn
    """
    player_word: str
    player_score: int
    ai_word: Optional[str] = None
    ai_score: Optional[int] = None
    timestamp: datetime = datetime.now()
    events: List[GameEvent] = None

    def __post_init__(self):
        if self.events is None:
            self.events = []
        # Normalize case for words
        self.player_word = self.player_word.upper()
        if self.ai_word:
            self.ai_word = self.ai_word.upper()

class GameHistory:
    """
    Manages the history of a game session.
    Tracks turns, scores, and events for analysis and replay.
    """
    
    def __init__(self):
        """
        Initialize an empty game history.
        """
        self.turns: List[Turn] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_player_score: int = 0
        self.total_ai_score: int = 0
        self.event_history: List[GameEvent] = []
        logger.info("GameHistory initialized")
        
    def start_game(self) -> None:
        """
        Mark the start of a new game.
        """
        self.start_time = datetime.now()
        logger.info("Game started")
        
    def end_game(self) -> None:
        """
        Mark the end of the game.
        """
        self.end_time = datetime.now()
        logger.info("Game ended")
        
    def add_turn(self, turn: Turn) -> None:
        """
        Add a new turn to the history.
        
        Args:
            turn (Turn): The turn to add
        """
        self.turns.append(turn)
        self.total_player_score += turn.player_score
        if turn.ai_score is not None:
            self.total_ai_score += turn.ai_score
        logger.debug(f"Turn added: {turn.player_word} ({turn.player_score} points)")
        
    def add_event(self, event: GameEvent) -> None:
        """
        Add a game event to the history.
        
        Args:
            event (GameEvent): The event to add
        """
        self.event_history.append(event)
        logger.debug(f"Event added: {event.type}")
        
    def get_turn_count(self) -> int:
        """
        Get the total number of turns played.
        
        Returns:
            int: Number of turns
        """
        return len(self.turns)
        
    def get_duration(self) -> Optional[float]:
        """
        Get the duration of the game in seconds.
        
        Returns:
            Optional[float]: Duration in seconds, or None if game hasn't ended
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
        
    def get_player_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the player's performance.
        
        Returns:
            Dict[str, Any]: Player statistics
        """
        return {
            'total_score': self.total_player_score,
            'turns_played': self.get_turn_count(),
            'average_score_per_turn': self.total_player_score / self.get_turn_count() if self.turns else 0,
            'duration': self.get_duration()
        }
        
    def get_ai_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the AI's performance.
        
        Returns:
            Dict[str, Any]: AI statistics
        """
        return {
            'total_score': self.total_ai_score,
            'turns_played': self.get_turn_count(),
            'average_score_per_turn': self.total_ai_score / self.get_turn_count() if self.turns else 0
        }
        
    def clear(self) -> None:
        """
        Clear the game history.
        Useful for starting a new game.
        """
        self.turns.clear()
        self.event_history.clear()
        self.start_time = None
        self.end_time = None
        self.total_player_score = 0
        self.total_ai_score = 0
        logger.info("Game history cleared") 