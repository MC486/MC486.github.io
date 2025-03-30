# ai/ai_strategy.py
# Coordinates between different AI models to make strategic decisions.

from typing import List, Dict, Any, Optional
import logging
from .markov_chain import MarkovChain
from .mcts import MCTS
from .naive_bayes import WordNaiveBayes
from .q_learning import QLearningAgent
from core.game_events import GameEvent, GameEventManager

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
        
        # Initialize AI components
        self.markov_chain = MarkovChain(order=2)
        self.mcts = MCTS()
        self.naive_bayes = WordNaiveBayes()
        self.q_agent = QLearningAgent(
            state_size=100,  # Adjust based on your state representation
            action_size=26,  # One for each letter
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
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
        
    def _setup_event_subscriptions(self) -> None:
        """
        Set up event subscriptions for the AI strategy.
        """
        self.event_manager.subscribe('word_submitted', self.on_word_submitted)
        self.event_manager.subscribe('game_start', self.on_game_start)
        self.event_manager.subscribe('difficulty_changed', self.on_difficulty_changed)
        self.event_manager.subscribe('turn_start', self.on_turn_start)
        
    def on_word_submitted(self, data: Dict[str, Any]) -> None:
        """
        Handle word submission events.
        Learn from player's word choices.
        
        Args:
            data (Dict[str, Any]): Event data containing word and score
        """
        word = data.get('word', '').lower()
        if word:
            self.markov_chain.train([word])
            self.naive_bayes.train([word], ['valid'])
            logger.debug(f"AI learned from word: {word}")
            
    def on_game_start(self, data: Dict[str, Any]) -> None:
        """
        Handle game start events.
        Initialize AI components with game state.
        
        Args:
            data (Dict[str, Any]): Event data containing initial game state
        """
        self.markov_chain.train(data.get('word_list', []))
        self.naive_bayes.train(
            data.get('word_list', []),
            ['valid'] * len(data.get('word_list', []))
        )
        logger.info("AI components initialized with game state")
        
    def on_difficulty_changed(self, data: Dict[str, Any]) -> None:
        """
        Handle difficulty change events.
        Adjust AI behavior based on new difficulty.
        
        Args:
            data (Dict[str, Any]): Event data containing new difficulty
        """
        self.difficulty = data.get('difficulty', 'medium')
        logger.info(f"AI difficulty adjusted to: {self.difficulty}")
        
    def on_turn_start(self, data: Dict[str, Any]) -> None:
        """
        Handle turn start events.
        Prepare AI for its turn.
        
        Args:
            data (Dict[str, Any]): Event data containing current game state
        """
        # Update AI state with current game state
        self._update_state(data)
        
    def choose_word(self, available_letters: List[str]) -> Optional[str]:
        """
        Choose a word using combined AI strategies.
        
        Args:
            available_letters (List[str]): List of available letters
            
        Returns:
            Optional[str]: Chosen word or None if no valid word found
        """
        # Generate candidate words using Markov Chain
        candidates = self._generate_candidates(available_letters)
        if not candidates:
            return None
            
        # Score candidates using Naive Bayes
        scored_candidates = self._score_candidates(candidates)
        if not scored_candidates:
            return None
            
        # Use MCTS to explore possibilities
        mcts_word = self._explore_with_mcts(scored_candidates, available_letters)
        
        # Use Q-learning to make final decision
        final_word = self._make_final_decision(mcts_word, scored_candidates)
        
        return final_word
        
    def _generate_candidates(self, available_letters: List[str]) -> List[str]:
        """
        Generate candidate words using Markov Chain.
        
        Args:
            available_letters (List[str]): Available letters
            
        Returns:
            List[str]: List of candidate words
        """
        candidates = []
        for _ in range(10):  # Generate 10 candidates
            word = self.markov_chain.generate_word(max_length=8)
            if word and all(letter in available_letters for letter in word):
                candidates.append(word)
        return candidates
        
    def _score_candidates(self, candidates: List[str]) -> Dict[str, float]:
        """
        Score candidate words using Naive Bayes.
        
        Args:
            candidates (List[str]): List of candidate words
            
        Returns:
            Dict[str, float]: Dictionary mapping words to their scores
        """
        scores = {}
        for word in candidates:
            probs = self.naive_bayes.predict_proba(word)
            scores[word] = probs.get('valid', 0)
        return scores
        
    def _explore_with_mcts(self, scored_candidates: Dict[str, float],
                          available_letters: List[str]) -> Optional[str]:
        """
        Use MCTS to explore word possibilities.
        
        Args:
            scored_candidates (Dict[str, float]): Scored candidate words
            available_letters (List[str]): Available letters
            
        Returns:
            Optional[str]: Best word found by MCTS
        """
        if not scored_candidates:
            return None
            
        best_word = max(scored_candidates.items(), key=lambda x: x[1])[0]
        return self.mcts.choose_action(best_word)
        
    def _make_final_decision(self, mcts_word: Optional[str],
                           scored_candidates: Dict[str, float]) -> Optional[str]:
        """
        Make final word choice using Q-learning.
        
        Args:
            mcts_word (Optional[str]): Word suggested by MCTS
            scored_candidates (Dict[str, float]): Scored candidate words
            
        Returns:
            Optional[str]: Final chosen word
        """
        if not scored_candidates:
            return None
            
        # Use MCTS word if confidence is high enough
        if mcts_word and scored_candidates.get(mcts_word, 0) >= self.confidence_thresholds[self.difficulty]:
            return mcts_word
            
        # Otherwise, use highest scoring candidate
        return max(scored_candidates.items(), key=lambda x: x[1])[0]
        
    def _update_state(self, game_state: Dict[str, Any]) -> None:
        """
        Update AI state with current game state.
        
        Args:
            game_state (Dict[str, Any]): Current game state
        """
        # Update Q-learning state
        state_features = self._extract_state_features(game_state)
        self.q_agent.update_state(state_features)
        
    def _extract_state_features(self, game_state: Dict[str, Any]) -> List[float]:
        """
        Extract relevant features from game state.
        
        Args:
            game_state (Dict[str, Any]): Current game state
            
        Returns:
            List[float]: List of state features
        """
        # Implement feature extraction based on your game state representation
        # This should return a list of numerical features
        return [] 