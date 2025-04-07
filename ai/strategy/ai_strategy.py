from typing import Dict, List, Set, Optional, Tuple, Any
import logging
from collections import defaultdict
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.models.markov_chain import MarkovChain
from ai.models.naive_bayes import NaiveBayes
from ai.models.mcts import MCTS
from ai.models.q_learning import QLearning
from core.validation.word_validator import WordValidator
from core.validation.trie import Trie
from ai.word_analysis import WordFrequencyAnalyzer
from database.manager import DatabaseManager
from database.repositories.markov_repository import MarkovRepository
from database.repositories.word_repository import WordRepository
from database.repositories.category_repository import CategoryRepository
import numpy as np

logger = logging.getLogger(__name__)

class AIStrategy:
    """
    Coordinates between different AI models to make strategic decisions.
    Uses a combination of Markov Chain, MCTS, Naive Bayes, and Q-learning.
    """
    def __init__(
        self,
        event_manager: GameEventManager,
        db_manager: DatabaseManager,
        word_repo: WordRepository,
        category_repo: CategoryRepository,
        difficulty: str = 'medium'
    ):
        """
        Initialize the AI strategy with all components.
        
        Args:
            event_manager (GameEventManager): Event manager for game events
            db_manager (DatabaseManager): Database manager instance
            word_repo (WordRepository): Repository for word usage data
            category_repo (CategoryRepository): Repository for word categories
            difficulty (str): Game difficulty level ('easy', 'medium', 'hard')
        """
        self.event_manager = event_manager
        self.db_manager = db_manager
        self.difficulty = difficulty
        
        # Initialize core components
        self.word_validator = WordValidator(word_repo)
        self.trie = Trie()
        self.word_analyzer = WordFrequencyAnalyzer(
            db_manager=self.db_manager,
            word_repo=word_repo,
            category_repo=category_repo
        )
        self.markov_repository = MarkovRepository(self.db_manager)
        
        # Initialize word analyzer with empty list since we're using on-demand validation
        self.word_analyzer.analyze_word_list([])
        
        # Initialize AI components
        self.markov_chain = MarkovChain(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            trie=self.trie,
            markov_repository=self.markov_repository,
            order=2
        )
        
        # Brief pre-training with common words
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        ]
        self.markov_chain.train(common_words)
        
        self.mcts = MCTS(
            valid_words=self.word_analyzer.get_analyzed_words(),
            max_depth=5,
            num_simulations=1000,
            db_manager=self.db_manager
        )
        self.naive_bayes = NaiveBayes(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            repo_manager=self.db_manager
        )
        self.q_agent = QLearning(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            q_learning_repository=self.db_manager.get_q_learning_repository()
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
        mcts_word = self.mcts.run(list(available_letters), [])
        if mcts_word and self.word_validator.validate_word_with_letters(mcts_word, available_letters):
            candidates.add(mcts_word)
        
        # Q-Learning candidates
        state = self._convert_letters_to_state(available_letters)
        action = self.q_agent.choose_action(state)
        q_word = self._convert_action_to_word(action, available_letters)
        if q_word and self.word_validator.validate_word_with_letters(q_word, available_letters):
            candidates.add(q_word)
        
        return candidates
        
    def _convert_letters_to_state(self, letters: Set[str]) -> np.ndarray:
        """Convert letters to state vector for Q-learning."""
        state = np.zeros(26)  # One-hot encoding for each letter
        for letter in letters:
            index = ord(letter.lower()) - ord('a')
            state[index] = 1
        return state
        
    def _convert_action_to_word(self, action: int, available_letters: Set[str]) -> Optional[str]:
        """Convert Q-learning action to word."""
        # For now, just return None since we need to implement proper action-to-word conversion
        return None

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
            
            # Combine scores using model weights
            combined_score = (
                self.model_weights["naive_bayes"] * naive_bayes_score
            )
            
            scored_words.append((word, combined_score))
            
        return sorted(scored_words, key=lambda x: x[1], reverse=True)

    def _select_best_word(self, scored_words: List[Tuple[str, float]]) -> str:
        """Select best word from scored candidates"""
        return scored_words[0][0] if scored_words else ""

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics from all AI components.
        
        Returns:
            Dict[str, Any]: Dictionary containing learning statistics
        """
        stats = {
            'markov_chain': {
                'transitions': len(self.markov_chain.transitions),
                'start_probabilities': len(self.markov_chain.start_probabilities)
            },
            'mcts': {
                'simulations': self.mcts.num_simulations,
                'max_depth': self.mcts.max_depth
            },
            'naive_bayes': {
                'word_count': len(self.naive_bayes.word_probabilities)
            },
            'q_learning': {
                'state_count': len(self.q_agent.q_table),
                'exploration_rate': self.q_agent.exploration_rate
            }
        }
        return stats

    def choose_word(self, available_letters: List[str]) -> str:
        """
        Choose a word using the available letters.
        
        Args:
            available_letters (List[str]): List of available letters
            
        Returns:
            str: The chosen word
        """
        # Convert list to set
        letter_set = set(available_letters)
        return self.select_word(letter_set, letter_set, 0)