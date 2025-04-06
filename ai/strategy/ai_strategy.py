from typing import Dict, List, Set, Optional, Tuple
import logging
from collections import defaultdict
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.markov_chain import MarkovChain
from ai.models.naive_bayes import NaiveBayes
from ai.models.mcts import MCTS
from ai.models.q_learning import QLearningAgent
from core.validation.word_validator import WordValidator

logger = logging.getLogger(__name__)

class AIStrategy:
    """
    Coordinates between different AI models to make strategic decisions.
    Uses a combination of Markov Chain, MCTS, Naive Bayes, and Q-learning.
    """
    def __init__(self, event_manager: GameEventManager, difficulty: str = 'medium'):
        """
        Initialize the AI strategy with all components.
        
        Args:
            event_manager (GameEventManager): Event manager for game events
            difficulty (str): Game difficulty level ('easy', 'medium', 'hard')
        """
        self.event_manager = event_manager
        self.difficulty = difficulty
        self.word_validator = WordValidator()
        self.word_analyzer = WordFrequencyAnalyzer()
        
        # Initialize AI components
        self.markov_chain = MarkovChain(order=2)
        self.mcts = MCTS()
        self.naive_bayes = NaiveBayes()
        self.q_agent = QLearningAgent(
            event_manager=event_manager,
            word_analyzer=self.word_analyzer,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.2
        )
        
        # Set up confidence thresholds based on difficulty
        self.confidence_thresholds = {
            'easy': 0.3,
            'medium': 0.5,
            'hard': 0.7
        }
        
        # Subscribe to relevant game events
        self._setup_event_subscriptions()
        logger.info(f"AIStrategy initialized with difficulty: {difficulty}")
        
        # Model weights based on difficulty
        self.model_weights = self._initialize_weights(difficulty)
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_words = 0

    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for strategy updates."""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)

    def _initialize_weights(self, difficulty: str) -> Dict[str, float]:
        """Initialize model weights based on difficulty level."""
        weights = {
            "naive_bayes": 0.3,
            "markov": 0.3,
            "mcts": 0.2,
            "q_learning": 0.2
        }
        
        if difficulty == "easy":
            weights["naive_bayes"] = 0.5
            weights["markov"] = 0.3
            weights["mcts"] = 0.1
            weights["q_learning"] = 0.1
        elif difficulty == "hard":
            weights["naive_bayes"] = 0.2
            weights["markov"] = 0.2
            weights["mcts"] = 0.3
            weights["q_learning"] = 0.3
            
        return weights

    def _handle_turn_start(self, event: GameEvent) -> None:
        """Handle turn start events"""
        # Reset any per-turn state
        pass

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission events"""
        # Update model weights based on success
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        
        if score > 0:
            self._adjust_weights(word, score)

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start events"""
        self.difficulty = event.data.get("difficulty", self.difficulty)
        self.model_weights = self._initialize_weights(self.difficulty)

    def _adjust_weights(self, word: str, score: int) -> None:
        """Adjust model weights based on success"""
        # Implementation for dynamic weight adjustment
        pass

    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        return {
            "total_decisions": self.total_decisions,
            "successful_words": self.successful_words,
            "success_rate": self.successful_words / self.total_decisions if self.total_decisions > 0 else 0,
            "current_weights": self.model_weights
        }

    def select_word(self, 
                    shared_letters: Set[str], 
                    private_letters: Set[str],
                    turn_number: int) -> str:
        """
        Select best word using combination of AI models.
        
        Args:
            shared_letters: Set of shared letters
            private_letters: Set of private letters
            turn_number: Current turn number
            
        Returns:
            Selected word
        """
        self.total_decisions += 1
        available_letters = shared_letters.union(private_letters)
        
        self.event_manager.emit(GameEvent(
            type=EventType.AI_ANALYSIS_START,
            data={"message": "Starting word selection"},
            debug_data={
                "turn_number": turn_number,
                "available_letters": list(available_letters)
            }
        ))
        
        # Get candidates from each model
        candidates = self._generate_candidates(available_letters, turn_number)
        
        # Score candidates using weighted combination
        scored_words = self._score_candidates(candidates, available_letters)
        
        # Select best word
        selected_word = self._select_best_word(scored_words)
        
        if selected_word:
            self.successful_words += 1
            
        self.event_manager.emit(GameEvent(
            type=EventType.AI_DECISION_MADE,
            data={"message": "Strategy selected word"},
            debug_data={
                "word": selected_word,
                "candidates_count": len(candidates),
                "success_rate": self.successful_words / self.total_decisions
            }
        ))
        
        return selected_word

    def _generate_candidates(self, 
                           available_letters: Set[str], 
                           turn_number: int) -> Set[str]:
        """Generate candidate words from all models"""
        candidates = set()
        
        # Markov Chain candidates
        markov_words = self.markov_chain.generate_word(available_letters)
        if markov_words and self.word_validator.validate_word_with_letters(markov_words, available_letters):
            candidates.add(markov_words)
        
        # MCTS candidates
        mcts_word = self.mcts.run(list(available_letters), [], len(available_letters))
        if mcts_word and self.word_validator.validate_word_with_letters(mcts_word, available_letters):
            candidates.add(mcts_word)
        
        # Q-Learning candidates
        q_word = self.q_agent.select_action(available_letters, self.valid_words, turn_number)
        if q_word and self.word_validator.validate_word_with_letters(q_word, available_letters):
            candidates.add(q_word)
        
        return candidates

    def _score_candidates(self, 
                         candidates: Set[str], 
                         available_letters: Set[str]) -> List[Tuple[str, float]]:
        """Score candidates using weighted model combinations"""
        scored_words = []
        
        for word in candidates:
            # Validate word before scoring
            if not self.word_validator.validate_word_with_letters(word, available_letters):
                continue
                
            # Get scores from each model
            naive_bayes_score = self.naive_bayes.estimate_word_probability(word)
            markov_score = self.word_analyzer.get_word_score(word)
            
            # Combine scores using model weights
            combined_score = (
                self.model_weights["naive_bayes"] * naive_bayes_score +
                self.model_weights["markov"] * markov_score
            )
            
            scored_words.append((word, combined_score))
            
        return sorted(scored_words, key=lambda x: x[1], reverse=True)

    def _select_best_word(self, scored_words: List[Tuple[str, float]]) -> str:
        """Select best word from scored candidates"""
        return scored_words[0][0] if scored_words else ""