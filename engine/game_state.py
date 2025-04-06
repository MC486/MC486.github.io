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
from ai.ai_strategy import AIStrategy
from database.manager import DatabaseManager
from database.repositories.word_repository import WordRepository

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
    last_played_word: Optional[str] = None
    
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
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        self.db_manager.initialize_database()
        
        # AI Strategy with database integration
        self.ai_strategy = AIStrategy(
            event_manager=event_manager,
            difficulty='medium',
            db_manager=self.db_manager
        )
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        # New word repository
        self.word_repo = WordRepository(self.db_manager)
        
    def _setup_event_subscriptions(self) -> None:
        """Setup all event subscriptions for game state management"""
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.GAME_END, self._handle_game_end)
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        
    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start event"""
        self.phase = GamePhase.IN_PROGRESS
        self.start_time = event.data.get("start_time", datetime.now())
        self.is_game_over = False
        
        # Update player name if provided
        if "player_name" in event.data:
            self.human_player.name = event.data["player_name"]
            
        # Update letter pools if provided
        if "shared_letters" in event.data:
            self.shared_letters = event.data["shared_letters"]
        if "boggle_letters" in event.data:
            self.boggle_letters = event.data["boggle_letters"]
            
        logger.info(f"Game started for player {self.human_player.name}")
        
    def _handle_game_end(self, event: GameEvent) -> None:
        """Handle game end event"""
        self.phase = GamePhase.ENDED
        self.end_time = event.data.get("end_time", datetime.now())
        self.is_game_over = True
        
        # Update final scores if provided
        if "human_score" in event.data:
            self.human_player.score = event.data["human_score"]
        if "ai_score" in event.data:
            self.ai_player.score = event.data["ai_score"]
            
        logger.info(f"Game ended. Final score: {self.human_player.score}")
        
    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission event"""
        word = event.data.get("word", "")
        player_id = event.data.get("player_id", "")
        score = event.data.get("score", 0)
        repeat_count = event.data.get("repeat_count", 0)
        
        if not word or not player_id:
            return
            
        # Update appropriate player's state
        player = self.human_player if player_id == "human" else self.ai_player
        # Score is already updated in process_turn
        player.used_words.add(word)
        player.word_usage_counts[word] = repeat_count + 1
        
        logger.debug(f"Word '{word}' submitted by {player_id} for {score} points")

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
        # Store the last played word
        self.human_player.last_played_word = word

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

    def display_game_over(self) -> None:
        """Display the final game state and summary."""
        if not self.is_game_over:
            return
            
        print("\n=== Game Over ===")
        print(f"Player: {self.human_player.name}")
        print(f"Final Score: {self.human_player.score}")
        print(f"Words Used: {len(self.human_player.used_words)}")
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"Game Duration: {duration:.1f} seconds")
            
        # Display AI learning statistics
        ai_stats = self.get_ai_stats()
        if ai_stats:
            print("\n=== AI Learning Statistics ===")
            print(f"Total Words Analyzed: {len(ai_stats.get('word_analyzer', []))}")
            print(f"Words Used: {len(ai_stats.get('used_words', []))}")
            
            # Display category statistics
            category_stats = ai_stats.get('category_stats', {})
            if category_stats:
                print("\n=== Category Statistics ===")
                print(f"Total Categories: {category_stats.get('total_categories', 0)}")
                print("\nCategory Word Counts:")
                for category, count in category_stats.get('category_word_counts', {}).items():
                    print(f"  {category}: {count} words")
            
            # Display model-specific statistics
            print("\n=== Model Statistics ===")
            for model, stats in ai_stats.items():
                if model not in ['word_analyzer', 'category_stats', 'used_words', 'word_success', 'confidence_threshold']:
                    print(f"\n{model.replace('_', ' ').title()}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
            
        print("================\n")

    def process_ai_turn(self) -> None:
        """Process the AI's turn, including word selection and scoring."""
        # Get available letters
        available_letters = self.shared_letters + self.boggle_letters
        logger.info(f"AI's turn - Available letters: {available_letters}")
        
        # Get AI's word selection using the strategy
        logger.info("AI attempting to choose a word...")
        ai_word = self.ai_strategy.choose_word(available_letters)
        
        if not ai_word:
            logger.warning("AI failed to find a valid word")
            print("🤖 AI couldn't find a valid word.")
            return
            
        # Validate word
        logger.info(f"AI selected word: {ai_word}")
        if not self.word_validator.validate_word_with_letters(ai_word, available_letters):
            logger.error(f"AI's word selection '{ai_word}' was invalid")
            print("🤖 AI's word selection was invalid.")
            return
            
        # Get word usage count
        repeat_count = self.ai_player.word_usage_counts.get(ai_word, 0)
        logger.info(f"Word '{ai_word}' has been used {repeat_count} times before")
        
        # Score calculation
        score = score_word(ai_word, repeat_count)
        self.ai_player.score += score
        self.ai_player.used_words.add(ai_word)
        self.ai_player.word_usage_counts[ai_word] = repeat_count + 1
        
        # Emit word submission event
        self.event_manager.emit(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": ai_word,
                "player_id": "ai",
                "score": score,
                "repeat_count": repeat_count
            }
        ))
        
        print(f"🤖 AI played '{ai_word}' for {score} points.")
        print(f"AI Score: {self.ai_player.score} 📊")

        # Check if AI guessed the player's last played word
        if self.human_player.last_played_word and ai_word == self.human_player.last_played_word:
            print(f"🤖 AI guessed your word! It was '{ai_word}'. 🤯")
            self.end_game()

    def cleanup(self) -> None:
        """Clean up old entries in repositories."""
        if hasattr(self, 'ai_strategy'):
            self.ai_strategy.cleanup()
            
    def get_ai_stats(self) -> Dict:
        """Get AI learning statistics."""
        if hasattr(self, 'ai_strategy'):
            return self.ai_strategy.get_learning_stats()
        return {}