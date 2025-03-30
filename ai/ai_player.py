from typing import Dict, Set, Optional
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.strategy.ai_strategy import AIStrategy

class AIPlayer:
    """
    Main AI player interface that interacts with the game.
    Manages AI state, decision making, and game interactions.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 valid_words: Set[str],
                 difficulty: str = "medium"):
        self.event_manager = event_manager
        self.valid_words = valid_words
        self.difficulty = difficulty
        
        # Initialize components
        self.word_analyzer = WordFrequencyAnalyzer(event_manager)
        self.strategy = AIStrategy(
            event_manager,
            self.word_analyzer,
            valid_words,
            difficulty
        )
        
        # Game state tracking
        self.current_shared_letters: Set[str] = set()
        self.current_private_letters: Set[str] = set()
        self.turn_number: int = 0
        self.score: int = 0
        self.used_words: Set[str] = set()
        
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for game interaction"""
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_END, self._handle_game_end)

    def make_move(self) -> str:
        """
        Make a move by selecting a word based on current game state.
        
        Returns:
            Selected word
        """
        self.event_manager.emit(GameEvent(
            type=EventType.AI_TURN_START,
            data={
                "turn_number": self.turn_number,
                "score": self.score
            }
        ))
        
        # Get word from strategy
        selected_word = self.strategy.select_word(
            self.current_shared_letters,
            self.current_private_letters,
            self.turn_number
        )
        
        if selected_word:
            self.used_words.add(selected_word)
            
            self.event_manager.emit(GameEvent(
                type=EventType.AI_WORD_SELECTED,
                data={
                    "word": selected_word,
                    "turn_number": self.turn_number
                },
                debug_data={
                    "used_words_count": len(self.used_words),
                    "available_letters": list(self.current_shared_letters | self.current_private_letters)
                }
            ))
            
        return selected_word

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start event"""
        self.difficulty = event.data.get("difficulty", self.difficulty)
        self.score = 0
        self.turn_number = 0
        self.used_words.clear()
        
        # Initialize word analyzer with game dictionary
        initial_words = event.data.get("valid_words", set())
        if initial_words:
            self.word_analyzer.analyze_word_list(list(initial_words))

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Handle turn start event"""
        self.turn_number = event.data.get("turn_number", self.turn_number + 1)
        self.current_shared_letters = set(event.data.get("shared_letters", []))
        self.current_private_letters = set(event.data.get("private_letters", []))

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
            "words_used": len(self.used_words),
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