from typing import Dict, List, Set, DefaultDict, Tuple
from collections import defaultdict
import math
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from core.validation.word_validator import WordValidator
from wordfreq import word_frequency
import logging
from database.repositories.word_repository import WordRepository
from database.manager import DatabaseManager

logger = logging.getLogger(__name__)

class WordFrequencyAnalyzer:
    """
    Analyzes word patterns, letter frequencies, and relationships for AI decision making.
    Provides statistical data used by various AI models.
    """
    def __init__(self, event_manager: GameEventManager):
        self.event_manager = event_manager
        self.db_manager = DatabaseManager()
        self.word_repo = WordRepository(self.db_manager)
        self.word_validator = WordValidator(use_nltk=True)
        
        # Letter frequency tracking
        self.letter_frequencies: DefaultDict[str, int] = defaultdict(int)
        self.total_letters = 0
        
        # Word pattern tracking
        self.word_lengths: DefaultDict[int, int] = defaultdict(int)
        self.total_words = 0
        
        # Letter pair frequencies (for bigram analysis)
        self.letter_pairs: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Position-based letter frequencies
        self.position_frequencies: DefaultDict[int, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Store analyzed words
        self.analyzed_words: Set[str] = set()
        
        # Subscribe to relevant events
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for word analysis."""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)

    def analyze_word_list(self, words: List[str]) -> None:
        """
        Analyze a list of words to build initial frequency data.
        
        Args:
            words: List of valid words to analyze
        """
        self.event_manager.emit(GameEvent(
            type=EventType.AI_ANALYSIS_START,
            data={"message": "Starting word list analysis"},
            debug_data={"word_count": len(words)}
        ))
        
        for word in words:
            word = word.upper()  # Convert to uppercase before validation
            if len(word) >= 3 and self.word_validator.validate_word(word):  # Enforce 3-letter minimum
                self._analyze_single_word(word)
                self.analyzed_words.add(word)
                
        self._calculate_probabilities()
        
        self.event_manager.emit(GameEvent(
            type=EventType.AI_ANALYSIS_COMPLETE,
            data={"message": "Word list analysis complete"},
            debug_data={"word_count": len(self.analyzed_words)}
        ))

    def _analyze_single_word(self, word: str) -> None:
        """
        Analyze patterns in a single word.
        
        Args:
            word: Word to analyze (must be uppercase)
        """
        if not word or not word.isalpha():
            return
            
        # Update word length frequency
        self.word_lengths[len(word)] += 1
        self.total_words += 1
        
        # Update letter frequencies
        for i, letter in enumerate(word):
            self.letter_frequencies[letter] += 1
            self.total_letters += 1
            self.position_frequencies[i][letter] += 1
            
            # Update letter pairs
            if i < len(word) - 1:
                self.letter_pairs[letter][word[i + 1]] += 1

    def _calculate_probabilities(self) -> None:
        """Calculate probability distributions from frequency data."""
        self.letter_probabilities = {
            letter: count / self.total_letters
            for letter, count in self.letter_frequencies.items()
        }
        
        self.length_probabilities = {
            length: count / self.total_words
            for length, count in self.word_lengths.items()
        }

    def get_letter_probability(self, letter: str) -> float:
        """
        Get the probability of a letter occurring.
        
        Args:
            letter: Letter to check
            
        Returns:
            Probability of the letter occurring
        """
        if not letter or not letter.isalpha():
            return 0.0
        return self.letter_probabilities.get(letter.upper(), 0.0)

    def get_next_letter_probability(self, current: str, next_letter: str) -> float:
        """
        Get probability of next_letter following current letter.
        
        Args:
            current: Current letter
            next_letter: Potential next letter
            
        Returns:
            Probability of the letter sequence
        """
        if not current or not next_letter or not current.isalpha() or not next_letter.isalpha():
            return 0.0
            
        current = current.upper()
        next_letter = next_letter.upper()
        
        if current not in self.letter_pairs:
            return 0.0
            
        total_follows = sum(self.letter_pairs[current].values())
        return self.letter_pairs[current][next_letter] / total_follows

    def get_position_probability(self, letter: str, position: int) -> float:
        """
        Get probability of letter occurring at specific position.
        
        Args:
            letter: Letter to check
            position: Position in word
            
        Returns:
            Probability of letter at position
        """
        if not letter or not letter.isalpha():
            return 0.0
            
        letter = letter.upper()
        if position not in self.position_frequencies:
            return 0.0
            
        total_at_position = sum(self.position_frequencies[position].values())
        return self.position_frequencies[position][letter] / total_at_position

    def get_word_score(self, word: str) -> float:
        """
        Calculate a probability-based score for a word.
        Higher scores indicate rarer/more interesting words.
        
        Args:
            word: Word to score
            
        Returns:
            Score between 0 and 1, where higher scores indicate rarer words
        """
        word = word.upper()
        if not self.word_validator.validate_word(word):
            return 0.0
            
        # Get WordFreq frequency score
        try:
            freq = word_frequency(word.lower(), 'en')
            # Convert frequency to a score between 0 and 1
            # We want a bell curve that peaks for moderately common words:
            # - Very rare words (freq < 1e-6) get low scores
            # - Very common words (freq > 1e-2) get moderate scores
            # - Moderately common words (freq around 1e-4) get highest scores
            freq_log = -math.log10(freq) if freq > 0 else 12  # Convert to log scale
            # Center the bell curve around frequency 1e-4 (log10 = -4)
            freq_score = math.exp(-((freq_log + 4) ** 2) / 8)  # Bell curve with width parameter 8
        except Exception as e:
            logging.warning(f"Failed to get WordFreq score for {word}: {e}")
            freq_score = 0.5  # Default to middle score if WordFreq fails
            
        # Get base score from word length probability
        length_prob = self.length_probabilities.get(len(word), 0.0)
        if length_prob == 0:
            return 0.0
            
        # Calculate letter position and transition scores
        position_scores = []
        transition_scores = []
        
        # Track if we've seen this word before
        word_seen = word in self.analyzed_words
        
        for i, letter in enumerate(word):
            # Get position probability
            pos_prob = self.get_position_probability(letter, i)
            # Use a small non-zero value for unknown positions
            position_scores.append(pos_prob if pos_prob > 0 else 0.01)
            
            # Get transition probability
            if i < len(word) - 1:
                trans_prob = self.get_next_letter_probability(letter, word[i + 1])
                # Use a small non-zero value for unknown transitions
                transition_scores.append(trans_prob if trans_prob > 0 else 0.01)
                
        # Calculate average scores
        avg_position_score = sum(position_scores) / len(position_scores)
        avg_transition_score = sum(transition_scores) / len(transition_scores) if transition_scores else 0.5
        
        # Combine scores with weights
        # Higher weights for known words and common letter combinations
        final_score = (
            0.2 * length_prob +
            0.2 * avg_position_score +
            0.2 * avg_transition_score +
            0.3 * freq_score +  # WordFreq score has highest weight
            0.1 * (1.0 if word_seen else 0.0)  # Bonus for words we've seen before
        )
        
        # Normalize score to be between 0 and 1
        final_score = min(1.0, max(0.0, final_score))
        
        # Log scoring details
        logging.debug(f"Word: {word}")
        logging.debug(f"  Length probability: {length_prob:.3f}")
        logging.debug(f"  Average position score: {avg_position_score:.3f}")
        logging.debug(f"  Average transition score: {avg_transition_score:.3f}")
        logging.debug(f"  WordFreq score: {freq_score:.3f} (frequency: {freq:.6f})")
        logging.debug(f"  Word seen before: {word_seen}")
        logging.debug(f"  Final score: {final_score:.3f}")
        
        return final_score

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission events to update analysis."""
        word = event.data.get("word", "").upper()
        if word and len(word) >= 3 and self.word_validator.validate_word(word):  # Enforce 3-letter minimum
            self._analyze_single_word(word)
            self._calculate_probabilities()
            
            self.event_manager.emit(GameEvent(
                type=EventType.MODEL_STATE_UPDATE,
                data={"message": "Word frequency analysis updated"},
                debug_data={
                    "word": word,
                    "word_score": self.get_word_score(word)
                }
            ))

    def _handle_game_start(self, event: GameEvent) -> None:
        """Reset analysis data at game start."""
        self.letter_frequencies.clear()
        self.word_lengths.clear()
        self.letter_pairs.clear()
        self.position_frequencies.clear()
        self.total_letters = 0
        self.total_words = 0

    def get_analyzed_words(self) -> List[str]:
        """
        Get the list of words that have been analyzed.
        
        Returns:
            List[str]: List of analyzed words
        """
        return list(self.analyzed_words)