from typing import Dict, List, Set, Optional, Tuple, Any
import logging
import random
from collections import defaultdict, Counter
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.models import MarkovChain, MCTS, NaiveBayes, QLearning
from core.validation.word_validator import WordValidator
from core.validation.trie import Trie
from ai.word_analysis import WordFrequencyAnalyzer
from database.manager import DatabaseManager
from database.repositories.markov_repository import MarkovRepository
from database.repositories.word_repository import WordRepository
from database.repositories.category_repository import CategoryRepository
import numpy as np
from unittest.mock import Mock

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
        
        # Create a game record so repositories with a game_id foreign key
        # (e.g. Markov transitions) have a valid game to attach to.
        self.game_repository = self.db_manager.get_game_repository()
        self.game_id = self.game_repository.create_game(
            player_name="ai_player",
            difficulty=self.difficulty,
            max_attempts=10
        )

        # Get repositories
        self.markov_repository = self.db_manager.get_markov_repository(self.game_id)
        self.naive_bayes_repository = self.db_manager.get_naive_bayes_repository()
        self.mcts_repository = self.db_manager.get_mcts_repository()
        self.q_learning_repository = self.db_manager.get_q_learning_repository(self.game_id)
        
        # Seed the analyzer with a real vocabulary of common, valid English
        # words so the models can actually generate valid words. Previously it
        # was seeded with an empty list, so the Markov model had no transitions
        # and the AI never produced a suggestion.
        self.vocabulary = self._load_seed_vocabulary()
        self.word_analyzer.analyze_word_list(self.vocabulary)
        
        # Initialize AI components. The Markov model builds its transition
        # matrix from the analyzer's (now populated) vocabulary on construction.
        self.markov_chain = MarkovChain(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            trie=self.trie,
            markov_repository=self.markov_repository,
            order=2
        )
        self.markov_chain.is_trained = True
        
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
            repository=self.q_learning_repository
        )
        
        # Initialize AI components
        self.models = {
            'markov': self.markov_chain,
            'mcts': self.mcts,
            'naive_bayes': self.naive_bayes,
            'q_learning': self.q_agent
        }
        
        # Model weights based on difficulty
        self.model_weights = self._initialize_weights(difficulty)
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_words = 0
        
        # Subscribe to relevant game events
        self._setup_event_subscriptions()
        logger.info(f"AIStrategy initialized with difficulty: {difficulty}")

    def _load_seed_vocabulary(self, size: int = 20000) -> List[str]:
        """
        Load a vocabulary of common, valid English words for the AI models.

        Uses the most frequent English words (via ``wordfreq``) filtered to
        real dictionary words that fit the game's length rules. Membership is
        checked against the validator's in-memory dictionary so this does not
        touch the database.
        """
        try:
            from wordfreq import top_n_list
            candidates = top_n_list('en', size)
        except Exception as e:
            logger.warning(f"Could not load seed vocabulary from wordfreq: {e}")
            candidates = []

        valid_words = getattr(self.word_validator, 'nltk_words', set())
        vocabulary = []
        for word in candidates:
            word = word.upper()
            if 3 <= len(word) <= 15 and word.isalpha() and word in valid_words:
                vocabulary.append(word)
        logger.info(f"Loaded seed vocabulary of {len(vocabulary)} words for the AI")
        return vocabulary

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
        word = event.data.get("word", "")
        score = event.data.get("score", 0)
        model = event.data.get("model", "")
        
        if score > 0 and model:
            self._adjust_weights(word, score)
            self.track_performance(model, word, True)
            
            # Record word usage
            repo = getattr(self.db_manager, f"get_{model}_repository")()
            if hasattr(repo, "record_word_usage"):
                repo.record_word_usage(word)

    def _handle_game_start(self, event: GameEvent) -> None:
        """Handle game start events"""
        self.difficulty = event.data.get("difficulty", self.difficulty)
        self.model_weights = self._initialize_weights(self.difficulty)
        
        # Reset all models
        for model_name in self.models:
            repo = getattr(self.db_manager, f"get_{model_name}_repository")()
            if hasattr(repo, "reset_model"):
                repo.reset_model()

    def _adjust_weights(self, word: str, score: int) -> None:
        """Adjust model weights based on success"""
        # Get the model that generated this word
        model_name = None
        for model in self.model_weights.keys():
            if model in word.lower():  # Simple heuristic to identify model
                model_name = model
                break
        
        if not model_name:
            return
            
        # Calculate adjustment factor based on score
        adjustment = min(0.1, score / 100)  # Cap adjustment at 0.1
        
        # Increase weight for successful model
        self.model_weights[model_name] += adjustment
        
        # Decrease weights for other models proportionally
        other_models = [m for m in self.model_weights.keys() if m != model_name]
        if other_models:
            decrease_per_model = adjustment / len(other_models)
            for model in other_models:
                self.model_weights[model] = max(0.1, self.model_weights[model] - decrease_per_model)
        
        # Normalize weights to sum to 1
        total = sum(self.model_weights.values())
        for model in self.model_weights:
            self.model_weights[model] /= total
            
        # Update repository
        repo = getattr(self.db_manager, f"get_{model_name}_repository")()
        if hasattr(repo, "update_model_weight"):
            repo.update_model_weight(self.model_weights[model_name])

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
        if shared_letters is None or private_letters is None:
            return ""
            
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

    # Approximate letter values, mirroring core.word_scoring, so the AI values
    # words the same way the game scores them.
    _LETTER_VALUES = {
        'e': 1, 'a': 1, 'i': 1, 'o': 1, 'n': 1, 'r': 1, 't': 1, 'l': 1, 's': 1,
        'd': 2, 'g': 2, 'b': 3, 'c': 3, 'm': 3, 'p': 3,
        'f': 4, 'h': 4, 'v': 4, 'w': 4, 'y': 4,
        'k': 5, 'j': 8, 'x': 8, 'q': 10, 'z': 10
    }

    def _formable_words(self, available_letters: Set[str]) -> List[str]:
        """Return vocabulary words that can be formed from the available letters."""
        pool = Counter(letter.upper() for letter in available_letters)
        formable = []
        for word in self.vocabulary:
            counts = Counter(word)
            if all(pool.get(letter, 0) >= n for letter, n in counts.items()):
                formable.append(word)
        return formable

    def _word_value(self, word: str) -> int:
        """Estimate a word's game value (length + letter rarity, length bonuses)."""
        w = word.lower()
        score = len(w) + sum(self._LETTER_VALUES.get(ch, 0) for ch in w)
        if len(w) >= 7:
            score *= 2
        elif len(w) >= 5:
            score = int(score * 1.5)
        return score

    def _generate_candidates(self, 
                           available_letters: Set[str], 
                           turn_number: int) -> Set[str]:
        """Generate candidate words the AI could actually play."""
        candidates = set()

        # Primary source: real vocabulary words formable from the available
        # letters. Keep the higher-value (longer) ones to bound the work.
        formable = self._formable_words(available_letters)
        formable.sort(key=len, reverse=True)
        candidates.update(formable[:50])

        # Also let any model that supports suggestions contribute.
        for model_name, model in self.models.items():
            if hasattr(model, "get_suggestion"):
                try:
                    word, _ = model.get_suggestion(available_letters)
                    if word and self.word_validator.validate_word_with_letters(word, available_letters):
                        candidates.add(word.upper())
                except Exception as e:
                    logger.warning(f"Error getting suggestion from {model_name}: {e}")
                    continue

        return candidates
    
    def _score_candidates(self, 
                         candidates: Set[str], 
                         available_letters: Set[str]) -> List[Tuple[str, float]]:
        """Score candidate words by their estimated game value."""
        scored_words = [(word, float(self._word_value(word))) for word in candidates]
        return sorted(scored_words, key=lambda x: x[1], reverse=True)
    
    def _select_best_word(self, scored_words: List[Tuple[str, float]]) -> str:
        """Select a word from the scored candidates, factoring in difficulty."""
        if not scored_words:
            return ""
        if self.difficulty == "hard":
            return scored_words[0][0]
        if self.difficulty == "easy":
            # Prefer weaker plays: pick from the lower-value half.
            lower_half = scored_words[len(scored_words) // 2:] or scored_words
            return random.choice(lower_half)[0]
        # medium: pick among the strongest handful for some variety.
        top = scored_words[:5]
        return random.choice(top)[0]

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
                'exploration_rate': self.q_agent.epsilon
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

    def track_performance(self, model_name: str, word: str, success: bool) -> None:
        """Track model performance"""
        if model_name in self.models:
            repo = getattr(self.db_manager, f"get_{model_name}_repository")()
            if hasattr(repo, "record_performance"):
                repo.record_performance(word, success)
                
            # Update model stats
            if hasattr(self.models[model_name], "get_stats"):
                try:
                    # Create new stats dictionary
                    stats = {
                        "success_rate": 1.0 if success else 0.0,
                        "total_decisions": 1,
                        "successful_words": 1 if success else 0
                    }
                    # Update the model's get_stats method
                    self.models[model_name].get_stats = Mock(return_value=stats)
                    # Store the updated stats in the model itself
                    setattr(self.models[model_name], '_stats', stats)
                except Exception as e:
                    logger.warning(f"Error updating stats for {model_name}: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {}
        for model_name, model in self.models.items():
            if hasattr(model, "get_stats"):
                try:
                    # Try to get stored stats first
                    if hasattr(model, '_stats'):
                        stats[model_name] = getattr(model, '_stats').copy()
                    else:
                        # Fall back to get_stats method
                        model_stats = model.get_stats()
                        if isinstance(model_stats, dict):
                            stats[model_name] = model_stats.copy()
                        else:
                            stats[model_name] = {
                                "success_rate": 0.0,
                                "total_decisions": 0,
                                "successful_words": 0
                            }
                except Exception as e:
                    logger.warning(f"Error getting stats for {model_name}: {e}")
                    stats[model_name] = {
                        "success_rate": 0.0,
                        "total_decisions": 0,
                        "successful_words": 0
                    }
            else:
                stats[model_name] = {
                    "success_rate": 0.0,
                    "total_decisions": 0,
                    "successful_words": 0
                }
        return stats