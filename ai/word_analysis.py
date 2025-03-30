from typing import Dict, List, Set, DefaultDict, Tuple
from collections import defaultdict
import math
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from core.validation.word_validator import WordValidator

class WordFrequencyAnalyzer:
    """
    Analyzes word patterns, letter frequencies, and relationships for AI decision making.
    Provides statistical data used by various AI models.
    """
    def __init__(self, event_manager: GameEventManager):
        self.event_manager = event_manager
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
            # Validate word before analysis
            word = word.upper()
            if self.word_validator.validate_word(word):
                self._analyze_single_word(word)
            
        self._calculate_probabilities()

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
        
        Args:
            word: Word to score
            
        Returns:
            Probability score for the word
        """
        word = word.upper()
        if not self.word_validator.validate_word(word):
            return 0.0
            
        score = self.length_probabilities.get(len(word), 0.0)
        
        # Multiply by letter probabilities
        for i, letter in enumerate(word):
            score *= self.get_position_probability(letter, i)
            
            # Include transition probabilities
            if i < len(word) - 1:
                score *= self.get_next_letter_probability(letter, word[i + 1])
                
        return score

    def _handle_word_submission(self, event: GameEvent) -> None:
        """Handle word submission events to update analysis."""
        word = event.data.get("word", "").upper()
        if word and self.word_validator.validate_word(word):
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