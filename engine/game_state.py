import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from core.letter_pool import generate_letter_pool
from core.word_scoring import score_word
from core.validation.word_validator import WordValidator
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager

logger = logging.getLogger(__name__)

class GamePhase(Enum):
    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"

@dataclass
class PlayerState:
    id: str
    name: Optional[str] = None
    score: int = 0
    used_words: Set[str] = None
    word_usage_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.used_words is None:
            self.used_words = set()
        if self.word_usage_counts is None:
            self.word_usage_counts = {}

class GameState:
    """
    Manages the current game state and handles game progress.
    Integrates event system while maintaining original functionality.
    """
    def __init__(self, event_manager: GameEventManager):
        # Original initialization
        self.word_validator = WordValidator(use_nltk=True)
        self.is_game_over = False
        
        # Enhanced state tracking
        self.event_manager = event_manager
        self.phase = GamePhase.SETUP
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Player states
        self.human_player = PlayerState("human")
        self.ai_player = PlayerState("ai")
        
        # Letter pools (maintaining original structure)
        self.shared_letters: List[str] = []
        self.boggle_letters: List[str] = []
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
    def _setup_event_subscriptions(self) -> None:
        """Setup all event subscriptions for game state management"""
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.GAME_END, self._handle_game_end)
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        
    def initialize_game(self) -> None:
        """Initializes player, letters, and game tracking."""
        # Original initialization
        self.human_player.name = input("🕺🏼Enter your name: ").strip()
        self.shared_letters, self.boggle_letters = generate_letter_pool()
        
        # New event-based initialization
        self.phase = GamePhase.IN_PROGRESS
        self.start_time = datetime.now()
        self.is_game_over = False
        
        # Emit game start event
        self.event_manager.emit(GameEvent(
            type=EventType.GAME_START,
            data={
                "player_name": self.human_player.name,
                "start_time": self.start_time,
                "shared_letters": self.shared_letters,
                "boggle_letters": self.boggle_letters
            }
        ))
        
        logger.info(f"🐦‍🔥 Game started for player {self.human_player.name}.")

    def redraw_boggle_letters(self) -> None:
        """Regenerates the player's boggle letters."""
        _, self.boggle_letters = generate_letter_pool()
        
        self.event_manager.emit(GameEvent(
            type=EventType.LETTERS_REDRAWN,
            data={"new_boggle_letters": self.boggle_letters}
        ))
        
        print("✨New boggle letters drawn.✨")

    def display_status(self) -> None:
        """Displays the current game state to the player."""
        print(f"\nShared Letters: {' '.join(self.shared_letters)}")
        print(f"Boggle Letters: {' '.join(self.boggle_letters)}")
        print(f"Current Score: {self.human_player.score} 🏆")

    def process_turn(self, word: str) -> None:
        """Validates and scores the player word, stores data, and checks for AI win."""
        if not word:
            print("🚫 No word entered.🚫")
            return

        # Validate word using WordValidator
        available_letters = self.shared_letters + self.boggle_letters
        if not self.word_validator.validate_word_with_letters(word, available_letters):
            print(f"🤔'{word}' is not a valid word or cannot be formed with current letters. Try again.🤔")
            return

        # Get word usage count
        repeat_count = self.human_player.word_usage_counts.get(word, 0)
        if repeat_count > 0:
            print("⚠️ You already used this word. Score will be reduced. ⚠️")

        # Score calculation
        score = score_word(word, repeat_count)
        self.human_player.score += score
        self.human_player.used_words.add(word)
        self.human_player.word_usage_counts[word] = repeat_count + 1

        # Emit word submission event
        self.event_manager.emit(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": word,
                "player_id": "human",
                "score": score,
                "repeat_count": repeat_count
            }
        ))

        print(f"Word '{word}' scored {score} points. 🎉")
        print(f"New Score: {self.human_player.score} 📈")

        # AI interaction (placeholder for now)
        ai_guess = "PLACEHOLDER"
        if ai_guess == word:
            print(f"🤖 AI guessed your word! It was '{ai_guess}'. 🤯")
            self.end_game()

    def end_game(self) -> None:
        """Ends the game and displays final score."""
        self.is_game_over = True
        self.phase = GamePhase.ENDED
        self.end_time = datetime.now()
        
        # Emit game end event
        self.event_manager.emit(GameEvent(
            type=EventType.GAME_END,
            data={
                "end_time": self.end_time,
                "human_score": self.human_player.score,
                "ai_score": self.ai_player.score,
                "duration": (self.end_time - self.start_time).total_seconds()
            }
        ))
        
        print(f"\nGame Over! 🏁 Final Score for {self.human_player.name}: {self.human_player.score} 🏆")
        logger.info("Game ended.")

    def get_game_summary(self) -> Dict:
        """Get current game state summary"""
        return {
            "phase": self.phase,
            "human_player": {
                "name": self.human_player.name,
                "score": self.human_player.score,
                "words_used": len(self.human_player.used_words)
            },
            "ai_player": {
                "score": self.ai_player.score,
                "words_used": len(self.ai_player.used_words)
            },
            "shared_letters": self.shared_letters.copy(),
            "boggle_letters": self.boggle_letters.copy(),
            "duration": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }