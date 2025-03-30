from typing import Dict, List, Set, Optional, Tuple
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.markov_chain import MarkovChain
from ai.models.naive_bayes import NaiveBayes
from ai.models.mcts import MCTS
from ai.models.q_learning import QLearningAgent

class AIStrategy:
    """
    Coordinates between different AI models to make optimal word selections.
    Combines strengths of each model based on game state and difficulty.
    """
    def __init__(self, 
                 event_manager: GameEventManager,
                 word_analyzer: WordFrequencyAnalyzer,
                 valid_words: Set[str],
                 difficulty: str = "medium"):
        self.event_manager = event_manager
        self.word_analyzer = word_analyzer
        self.valid_words = valid_words
        self.difficulty = difficulty
        
        # Initialize AI models
        self.markov_chain = MarkovChain(event_manager, word_analyzer)
        self.naive_bayes = NaiveBayes(event_manager, word_analyzer)
        self.mcts = MCTS(event_manager, word_analyzer, valid_words)
        self.q_learning = QLearningAgent(event_manager, word_analyzer)
        
        # Model weights based on difficulty
        self.model_weights = self._initialize_weights(difficulty)
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_words = 0
        
        self._setup_event_subscriptions()

    def _initialize_weights(self, difficulty: str) -> Dict[str, float]:
        """Initialize model weights based on difficulty level"""
        if difficulty == "easy":
            return {
                "markov": 0.4,    # More predictable patterns
                "naive_bayes": 0.3,
                "mcts": 0.2,
                "q_learning": 0.1
            }
        elif difficulty == "medium":
            return {
                "markov": 0.25,
                "naive_bayes": 0.25,
                "mcts": 0.25,
                "q_learning": 0.25
            }
        else:  # hard
            return {
                "markov": 0.1,
                "naive_bayes": 0.2,
                "mcts": 0.3,
                "q_learning": 0.4    # More strategic decisions
            }

    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions"""
        self.event_manager.subscribe(EventType.TURN_START, self._handle_turn_start)
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)

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
        if markov_words:
            candidates.add(markov_words)
        
        # MCTS candidates
        mcts_word = self.mcts.run(list(available_letters), [], len(available_letters))
        if mcts_word:
            candidates.add(mcts_word)
        
        # Q-Learning candidates
        q_word = self.q_learning.select_action(available_letters, self.valid_words, turn_number)
        if q_word:
            candidates.add(q_word)
        
        return candidates.intersection(self.valid_words)

    def _score_candidates(self, 
                         candidates: Set[str], 
                         available_letters: Set[str]) -> List[Tuple[str, float]]:
        """Score candidates using weighted model combinations"""
        scored_words = []
        
        for word in candidates:
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