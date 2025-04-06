# ai/ai_strategy.py
# Coordinates between different AI models to make strategic decisions.

from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.category_analysis import CategoryAnalyzer
from ai.models.markov_chain import MarkovChain
from ai.models.naive_bayes import NaiveBayes
from ai.models.mcts import MCTS
from ai.models.q_learning import QLearningAgent
from core.validation.word_validator import WordValidator
from core.validation.trie import Trie
from database.repositories.word_repository import WordRepository
from database.manager import DatabaseManager

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
        self.word_analyzer = WordFrequencyAnalyzer(event_manager=event_manager)
        self.category_analyzer = CategoryAnalyzer()
        self.trie = Trie()
        self.db_manager = DatabaseManager()
        self.word_repo = WordRepository(self.db_manager)
        self.word_validator = WordValidator(use_nltk=True)
        
        # Track word usage and success
        self.used_words = set()
        self.word_success = defaultdict(float)
        
        # Simple caching for word scores
        self.word_score_cache = {}
        
        # Initialize word list
        self._initialize_word_list()
        
        # Initialize AI components
        self.markov_chain = MarkovChain(
            event_manager=event_manager,
            word_analyzer=self.word_analyzer,
            trie=self.trie,
            order=2
        )
        self.mcts = MCTS(valid_words=set(self.word_analyzer.get_analyzed_words()))
        self.naive_bayes = NaiveBayes(
            event_manager=event_manager,
            word_analyzer=self.word_analyzer
        )
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
        
    def _initialize_word_list(self) -> None:
        """Initialize the word list from the word validator."""
        from utils.word_list_loader import load_word_list
        words = load_word_list()
        
        # Add words to both analyzer and trie
        self.word_analyzer.analyze_word_list(words)
        for word in words:
            self.trie.insert(word)
            
        logger.info(f"Initialized word list with {len(self.word_analyzer.get_analyzed_words())} words")
        
    def _setup_event_subscriptions(self) -> None:
        """
        Set up event subscriptions for the AI strategy.
        """
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self.on_word_submitted)
        self.event_manager.subscribe(EventType.GAME_START, self.on_game_start)
        self.event_manager.subscribe(EventType.DIFFICULTY_CHANGED, self.on_difficulty_changed)
        self.event_manager.subscribe(EventType.TURN_START, self.on_turn_start)
        
    def on_word_submitted(self, event: GameEvent) -> None:
        """
        Handle word submission events.
        Learn from player's word choices.
        
        Args:
            event (GameEvent): Event containing word and score
        """
        word = event.data.get('word', '').upper()  # Convert to uppercase
        if word:
            self.markov_chain.train([word])
            self.naive_bayes.train([word], ['valid'])
            logger.debug(f"AI learned from word: {word}")
            
    def on_game_start(self, event: GameEvent) -> None:
        """
        Handle game start events.
        Initialize AI components with game state.
        
        Args:
            event (GameEvent): Event containing initial game state
        """
        word_list = [word.upper() for word in event.data.get('word_list', [])]  # Ensure uppercase
        self.markov_chain.train(word_list)
        self.naive_bayes.train(
            word_list,
            ['valid'] * len(word_list)
        )
        logger.info("AI components initialized with game state")
        
    def on_difficulty_changed(self, event: GameEvent) -> None:
        """
        Handle difficulty change events.
        Adjust AI behavior based on new difficulty.
        
        Args:
            event (GameEvent): Event containing new difficulty
        """
        self.difficulty = event.data.get('difficulty', 'medium')
        logger.info(f"AI difficulty adjusted to: {self.difficulty}")
        
    def on_turn_start(self, event: GameEvent) -> None:
        """
        Handle turn start events.
        Prepare AI for its turn.
        
        Args:
            event (GameEvent): Event containing current game state
        """
        # Update AI state with current game state
        self._update_state(event.data)
        
    def choose_word(self, available_letters: List[str]) -> Optional[str]:
        """
        Choose a word using combined AI strategies.
        
        Args:
            available_letters (List[str]): List of available letters
            
        Returns:
            Optional[str]: Chosen word or None if no valid word found
        """
        # Store available letters for use in scoring
        self.current_available_letters = available_letters
        
        logger.info(f"AI attempting to choose word with available letters: {available_letters}")
        
        # Generate candidate words using Markov Chain
        candidates = self._generate_candidates(available_letters)
        if not candidates:
            logger.warning("No valid candidates generated from Markov Chain")
            return None
            
        # Score candidates using multiple models and word history
        scored_candidates = self._score_candidates(candidates)
        if not scored_candidates:
            logger.warning("No candidates scored by multiple models")
            return None
            
        # Use MCTS to explore possibilities
        mcts_word = self._explore_with_mcts(scored_candidates, available_letters)
        if mcts_word:
            logger.info(f"MCTS suggested word: {mcts_word}")
        
        # Use Q-learning to make final decision
        final_word = self._make_final_decision(mcts_word, scored_candidates)
        if final_word:
            logger.info(f"AI chose word: {final_word}")
        else:
            logger.warning("AI failed to make a final decision")
        
        return final_word
        
    def _generate_candidates(self, available_letters: List[str]) -> List[str]:
        """
        Generate candidate words using Markov Chain and WordFreq.
        
        Args:
            available_letters (List[str]): List of available letters
            
        Returns:
            List[str]: List of candidate words
        """
        candidates = []
        min_length = 3  # Minimum word length requirement
        max_total_attempts = 200  # Maximum total attempts across all tries
        total_attempts = 0
        
        logger.info(f"Markov Chain attempting to generate words with letters: {available_letters}")
        
        # Keep trying until we find at least one valid word or hit max attempts
        while not candidates and total_attempts < max_total_attempts:
            # Generate candidate words using Markov Chain
            for attempt in range(50):  # Try up to 50 times to generate valid words
                total_attempts += 1
                if total_attempts >= max_total_attempts:
                    logger.warning(f"Reached maximum total attempts ({max_total_attempts})")
                    break
                    
                word = self.markov_chain.generate_word(available_letters)
                if word and len(word) >= min_length:
                    # Validate word against available letters
                    letter_counts = {}
                    for letter in available_letters:
                        letter_counts[letter] = letter_counts.get(letter, 0) + 1
                        
                    # Check if word can be formed with available letters
                    can_form = True
                    for letter in word:
                        if letter not in letter_counts or letter_counts[letter] <= 0:
                            can_form = False
                            break
                        letter_counts[letter] -= 1
                        
                    if can_form and word not in candidates:
                        # Get word frequency to filter out extremely rare words
                        try:
                            from wordfreq import word_frequency
                            freq = word_frequency(word.lower(), 'en')
                            # Only accept words that are reasonably common
                            # (frequency > 1e-6 means appears at least once per million words)
                            # Unless we're really struggling to find words
                            if freq > 1e-6 or (total_attempts > 150 and len(candidates) == 0):
                                candidates.append(word)
                                logger.info(f"Added valid candidate: {word} (frequency: {freq:.6f})")
                            else:
                                logger.debug(f"Rejected rare word: {word} (frequency: {freq:.6f})")
                        except Exception as e:
                            # If WordFreq fails, accept the word (better than nothing)
                            logger.warning(f"WordFreq check failed for {word}: {e}")
                            candidates.append(word)
                            
            if not candidates and total_attempts < max_total_attempts:
                logger.warning(f"No valid candidates found after {total_attempts} attempts, retrying...")
                # If we still don't have candidates, lower the threshold slightly
                min_length = max(3, min_length - 1)  # Don't go below 3 letters
                logger.info(f"Lowered minimum length to {min_length}")
                        
        if not candidates:
            logger.warning(f"Failed to generate any valid candidates after {total_attempts} attempts")
        else:
            logger.info(f"Markov Chain generated {len(candidates)} valid candidates: {candidates}")
            
        return candidates
        
    def _score_candidates(self, candidates: List[str]) -> Dict[str, float]:
        """
        Score candidate words using multiple models and word history.
        Uses WordFreq to favor less common words for higher scores.
        
        Args:
            candidates (List[str]): List of candidate words
            
        Returns:
            Dict[str, float]: Dictionary mapping words to their scores
        """
        scores = {}
        logger.info(f"Scoring {len(candidates)} candidates using multiple models")
        
        for word in candidates:
            # Get scores from each model
            naive_bayes_score = self.naive_bayes.estimate_word_probability(word)
            
            # Use cached word score if available
            if word in self.word_score_cache:
                markov_score = self.word_score_cache[word]
            else:
                markov_score = self.word_analyzer.get_word_score(word)
                self.word_score_cache[word] = markov_score
            
            # Get word success history
            success_rate = self.word_success.get(word, 0.5)  # Default to 0.5 for new words
            
            # Calculate letter distribution score
            letter_dist_score = self._calculate_letter_distribution_score(word)
            
            logger.debug(f"Word: {word}")
            logger.debug(f"  Naive Bayes score: {naive_bayes_score:.3f}")
            logger.debug(f"  Markov Chain score: {markov_score:.3f}")
            logger.debug(f"  Success rate: {success_rate:.3f}")
            logger.debug(f"  Letter distribution score: {letter_dist_score:.3f}")
            
            # Combine scores with balanced weights
            combined_score = (
                0.25 * naive_bayes_score +
                0.25 * markov_score +  # Reduced weight for word frequency
                0.25 * success_rate +
                0.25 * letter_dist_score  # Added letter distribution score
            )
            
            scores[word] = combined_score
            logger.debug(f"  Combined score: {combined_score:.3f}")
            
        logger.info(f"Scored {len(scores)} candidates: {scores}")
        return scores
        
    def _calculate_letter_distribution_score(self, word: str) -> float:
        """
        Calculate a score based on how well the word's letter distribution
        matches the available letters.
        
        Args:
            word (str): Word to score
            
        Returns:
            float: Score between 0 and 1
        """
        # Count letter frequencies in word
        word_letters = {}
        for letter in word:
            word_letters[letter] = word_letters.get(letter, 0) + 1
            
        # Compare with available letter distribution
        score = 0.0
        total_letters = len(word)
        
        for letter, count in word_letters.items():
            # Get frequency of this letter in available letters
            available_count = sum(1 for l in self.current_available_letters if l == letter)
            if available_count > 0:
                # Score based on how well we're using available letters
                score += min(count, available_count) / total_letters
                
        return score
        
    def _explore_with_mcts(self, scored_candidates: Dict[str, float],
                          available_letters: List[str]) -> Optional[str]:
        """
        Use MCTS to explore word possibilities.
        Considers word frequency in exploration.
        
        Args:
            scored_candidates (Dict[str, float]): Scored candidate words
            available_letters (List[str]): Available letters
            
        Returns:
            Optional[str]: Best word found by MCTS
        """
        if not scored_candidates:
            logger.warning("No candidates available for MCTS exploration")
            return None
            
        logger.info(f"MCTS exploring with letters: {available_letters}")
        
        # Filter candidates to ensure minimum length and avoid used words
        valid_candidates = {
            word: score for word, score in scored_candidates.items() 
            if len(word) >= 3 and word not in self.used_words
        }
        if not valid_candidates:
            logger.warning("No valid candidates for MCTS after filtering")
            return None
            
        # Split available letters into shared and private
        shared_letters = available_letters[:4]  # First 4 letters are shared
        private_letters = available_letters[4:]  # Remaining letters are private
        
        logger.debug(f"MCTS using shared letters: {shared_letters}")
        logger.debug(f"MCTS using private letters: {private_letters}")
        
        # Run MCTS with the available letters
        mcts_word = self.mcts.run(shared_letters, private_letters, max(len(word) for word in valid_candidates))
        
        if mcts_word:
            logger.info(f"MCTS suggested word: {mcts_word}")
            # Only return MCTS word if it's not been used before and has good score
            if mcts_word not in self.used_words:
                word_score = self.word_analyzer.get_word_score(mcts_word)
                logger.debug(f"MCTS word score: {word_score:.3f}")
                if word_score > 0.1:  # Threshold for word rarity
                    logger.info(f"MCTS selected valid word: {mcts_word}")
                    return mcts_word
                else:
                    logger.debug(f"MCTS word too common: {mcts_word}")
            else:
                logger.debug(f"MCTS word already used: {mcts_word}")
        else:
            logger.warning("MCTS failed to find a word")
            
        return None
        
    def _make_final_decision(self, mcts_word: Optional[str],
                           scored_candidates: Dict[str, float]) -> Optional[str]:
        """
        Make final word choice using Q-learning and model combination.
        
        Args:
            mcts_word (Optional[str]): Word suggested by MCTS
            scored_candidates (Dict[str, float]): Scored candidate words
            
        Returns:
            Optional[str]: Final chosen word
        """
        if not scored_candidates:
            logger.warning("No candidates available for final decision")
            return None
            
        logger.info("Making final word selection")
        
        # Use MCTS word if confidence is high enough
        if mcts_word:
            mcts_score = scored_candidates.get(mcts_word, 0)
            logger.debug(f"MCTS word score: {mcts_score:.3f}")
            if mcts_score >= self.confidence_thresholds[self.difficulty]:
                logger.info(f"Selected MCTS word: {mcts_word} (confidence: {mcts_score:.3f})")
                return mcts_word
                
        # Get Q-learning suggestion
        logger.debug("Getting Q-learning suggestion")
        q_word = self.q_agent.select_action(set(''.join(scored_candidates.keys())), 
                                          set(scored_candidates.keys()))
        
        # If Q-learning suggests a valid word, use it
        if q_word and q_word in scored_candidates:
            logger.info(f"Selected Q-learning word: {q_word}")
            return q_word
            
        # Otherwise, use highest scoring candidate
        best_word = max(scored_candidates.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected highest scoring word: {best_word} (score: {scored_candidates[best_word]:.3f})")
        return best_word
        
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
        features = []
        
        # Add word success rates
        if 'word_scores' in game_state:
            features.extend([
                sum(game_state['word_scores'].values()) / len(game_state['word_scores']),
                max(game_state['word_scores'].values(), default=0),
                min(game_state['word_scores'].values(), default=0)
            ])
            
        # Add letter distribution features
        if 'available_letters' in game_state:
            letter_counts = defaultdict(int)
            for letter in game_state['available_letters']:
                letter_counts[letter] += 1
            features.extend([
                len(letter_counts),
                max(letter_counts.values(), default=0),
                sum(letter_counts.values()) / len(letter_counts)
            ])
            
        # Add game progress features
        if 'turn_number' in game_state:
            features.append(game_state['turn_number'])
            
        return features 

    def get_word_score(self, word: str) -> float:
        """
        Calculate a combined score for a word using multiple factors.
        
        Args:
            word: Word to score
            
        Returns:
            Score between 0 and 1
        """
        if word in self.word_score_cache:
            return self.word_score_cache[word]
            
        # Get base score from word analyzer
        base_score = self.word_analyzer.get_word_score(word)
        
        # Get category score
        category_score = self.category_analyzer.get_category_score(word)
        
        # Combine scores with weights
        final_score = (
            0.7 * base_score +  # Base score has higher weight
            0.3 * category_score  # Category score
        )
        
        # Cache the score
        self.word_score_cache[word] = final_score
        return final_score

    def get_ai_stats(self) -> Dict[str, Any]:
        """
        Get statistics about AI performance and learning.
        
        Returns:
            Dict containing AI statistics
        """
        stats = {
            "word_analyzer": self.word_analyzer.get_analyzed_words(),
            "category_stats": self.category_analyzer.get_category_stats(),
            "used_words": list(self.used_words),
            "word_success": dict(self.word_success),
            "confidence_threshold": self.confidence_thresholds[self.difficulty]
        }
        
        # Add model-specific stats
        stats.update({
            "markov_chain": self.markov_chain.get_stats(),
            "naive_bayes": self.naive_bayes.get_stats(),
            "mcts": self.mcts.get_stats(),
            "q_learning": self.q_agent.get_stats()
        })
        
        return stats 