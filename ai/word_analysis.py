from typing import Dict, List, Set, DefaultDict, Tuple
from collections import defaultdict
import math
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from core.validation.word_validator import WordValidator
from wordfreq import word_frequency
import logging
from database.repositories.word_repository import WordRepository
from database.repositories.category_repository import CategoryRepository
from database.manager import DatabaseManager
import nltk
from typing import Any
import random

logger = logging.getLogger(__name__)

class WordFrequencyAnalyzer:
    """
    Analyzes word patterns, letter frequencies, and relationships for AI decision making.
    Provides statistical data used by various AI models.
    """
    def __init__(self, db_manager: DatabaseManager, word_repo: WordRepository, category_repo: CategoryRepository):
        """
        Initialize the word analyzer.
        
        Args:
            db_manager (DatabaseManager): Database manager instance
            word_repo: Repository for word usage data
            category_repo: Repository for word categories
        """
        self.db_manager = db_manager
        self.word_repo = word_repo
        self.category_repo = category_repo
        self.analyzed_words: Dict[str, Dict[str, Any]] = {}
        self.word_frequencies: Dict[str, int] = defaultdict(int)
        self.pattern_frequencies: Dict[str, int] = defaultdict(int)
        self.letter_frequencies: Dict[str, int] = defaultdict(int)
        self.word_lengths: Dict[int, int] = defaultdict(int)
        self.total_words = 0
        self.total_letters = 0
        self.position_frequencies: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.letter_pairs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.length_probabilities: Dict[int, float] = {}
        self.letter_probabilities: Dict[str, float] = {}
        self.event_manager = GameEventManager()
        self.word_validator = WordValidator(self.word_repo)
        
        # Set up event subscriptions
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for word analysis."""
        self.event_manager.subscribe(EventType.WORD_SUBMITTED, self._handle_word_submission)
        self.event_manager.subscribe(EventType.GAME_START, self._handle_game_start)

    def analyze_word_list(self, words: List[str]) -> None:
        """
        Analyze a list of words to build frequency data.
        
        Args:
            words: List of words to analyze
        """
        # Reset analysis data
        self._initialize_analysis()
        
        # Load initial word frequencies from repository
        usage_data = self.word_repo.get_word_usage()
        for word_data in usage_data:
            word = word_data["word"].upper()
            self._analyze_single_word(word)
            self.analyzed_words[word] = {
                'length': len(word),
                'frequency': word_data.get('frequency', 1)
            }
            
        # Calculate initial probabilities
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
                
        # Calculate probabilities
        self._calculate_probabilities()

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
        if not self.word_validator.is_valid_word(word):
            return 0.0
            
        # Get WordFreq frequency score
        try:
            freq_score = word_frequency(word.lower(), 'en')
        except:
            freq_score = 0.0
            
        # Get length score (favor medium length words)
        length = len(word)
        length_score = 1.0 - abs(length - 7) / 7  # Peak at 7 letters
        
        # Get letter rarity score
        letter_score = sum(self.get_letter_probability(c) for c in word) / len(word)
        
        # Combine scores with weights
        weights = {
            'frequency': 0.4,
            'length': 0.3,
            'letters': 0.3
        }
        
        final_score = (
            weights['frequency'] * (1.0 - freq_score) +  # Invert freq score to favor rare words
            weights['length'] * length_score +
            weights['letters'] * letter_score
        )
        
        return min(1.0, max(0.0, final_score))

    def _handle_word_submission(self, event: GameEvent) -> None:
        """
        Handle word submission events by analyzing the submitted word.
        
        Args:
            event: GameEvent containing the submitted word
        """
        if not event.data or "word" not in event.data:
            return
            
        word = event.data["word"].upper()
        if self.word_validator.validate_word(word):
            self._analyze_single_word(word)
            self.analyzed_words[word] = {
                "score": self.get_word_score(word),
                "frequency": self.word_frequencies.get(word, 0)
            }
            self._calculate_probabilities()

    def _handle_game_start(self, event: GameEvent) -> None:
        """
        Handle game start events by resetting analysis if needed.
        
        Args:
            event: GameEvent for game start
        """
        # Reset analysis if requested in event data
        if event.data and event.data.get("reset_analysis", False):
            self._initialize_analysis()

    def get_analyzed_words(self) -> List[str]:
        """
        Get the list of analyzed words.
        
        Returns:
            List of analyzed words
        """
        # Get words from repository instead of preloaded list
        usage_data = self.word_repo.get_word_usage()
        return [word_data["word"].upper() for word_data in usage_data]

    def analyze_word_usage(self) -> Dict[str, any]:
        """
        Analyze word usage patterns from the database.
        
        Returns:
            Dictionary containing usage statistics
        """
        usage_data = self.word_repo.get_word_usage()
        analysis = {
            "total_words": len(usage_data),
            "word_frequencies": defaultdict(int),
            "length_frequencies": defaultdict(int),
            "pattern_frequencies": defaultdict(int)
        }
        
        for word_data in usage_data:
            word = word_data["word"].upper()
            frequency = word_data["frequency"]
            analysis["word_frequencies"][word] = frequency
            analysis["length_frequencies"][len(word)] += frequency
            
            # Analyze patterns (e.g. prefixes, suffixes)
            if len(word) >= 3:
                prefix = word[:3]
                suffix = word[-3:]
                analysis["pattern_frequencies"][f"prefix_{prefix}"] += frequency
                analysis["pattern_frequencies"][f"suffix_{suffix}"] += frequency
                
        return analysis

    def _initialize_analysis(self) -> None:
        """Initialize or reset analysis data structures."""
        self.analyzed_words.clear()
        self.word_frequencies.clear()
        self.pattern_frequencies.clear()
        self.letter_frequencies.clear()
        self.word_lengths.clear()
        self.total_words = 0
        self.total_letters = 0
        self.position_frequencies.clear()
        self.letter_pairs.clear()

    def get_patterns(self, word: str) -> Dict[str, str]:
        """
        Get patterns from a word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary mapping pattern types to patterns
        """
        word = word.upper()
        return {
            'prefix': word[:3] if len(word) >= 3 else word,
            'suffix': word[-3:] if len(word) >= 3 else word,
            'length': str(len(word))
        }
        
    def get_pattern_frequency(self, pattern: str) -> float:
        """
        Get frequency of a pattern.
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Frequency of the pattern
        """
        total_patterns = sum(self.pattern_frequencies.values())
        return self.pattern_frequencies.get(pattern, 0) / total_patterns if total_patterns > 0 else 0
        
    def get_pattern_rarity(self, pattern: str) -> float:
        """
        Get rarity score of a pattern.
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Rarity score between 0 and 1
        """
        freq = self.get_pattern_frequency(pattern)
        if freq == 0:
            return 1.0  # Very rare
        return 1.0 - freq  # Invert frequency for rarity
        
    def get_pattern_success_rate(self, pattern: str) -> float:
        """
        Get success rate of words with this pattern.
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Success rate between 0 and 1
        """
        # For now, return a neutral value
        # This could be improved by tracking pattern success in the repository
        return 0.5
        
    def get_pattern_weight(self, pattern_type: str) -> float:
        """
        Get weight for a pattern type.
        
        Args:
            pattern_type: Type of pattern
            
        Returns:
            Weight between 0 and 1
        """
        weights = {
            'prefix': 0.4,
            'suffix': 0.3,
            'length': 0.3
        }
        return weights.get(pattern_type, 0.0)
        
    def get_rarity_score(self, word: str) -> float:
        """
        Get rarity score for a word.
        
        Args:
            word: Word to check
            
        Returns:
            Rarity score between 0 and 1
        """
        word = word.upper()
        freq = self.word_frequencies.get(word, 0)
        if freq == 0:
            return 1.0  # Very rare
        return 1.0 - (freq / max(self.word_frequencies.values()))
        
    def get_word_frequency(self, word: str) -> int:
        """
        Get frequency count for a word.
        
        Args:
            word: Word to check
            
        Returns:
            Number of times the word has been seen
        """
        word = word.upper()
        return self.word_frequencies.get(word, 0)

    def get_popular_words(self, limit: int = 10) -> List[Dict]:
        """
        Get the most frequently used words.

        Args:
            limit: Maximum number of words to return

        Returns:
            List of word statistics dictionaries
        """
        return self.word_repo.get_most_frequent_words(limit)

    def get_rare_words(self, limit: int = 10) -> List[Dict]:
        """
        Get the least frequently used words.

        Args:
            limit: Maximum number of words to return

        Returns:
            List of word statistics dictionaries
        """
        return self.word_repo.get_least_frequent_words(limit)

    def get_word_stats(self, word: str) -> Dict:
        """
        Get detailed statistics for a word.

        Args:
            word: The word to analyze

        Returns:
            Dictionary containing word statistics
        """
        return self.word_repo.get_word_stats(word)

    def get_category_stats(self, category: str) -> Dict:
        """
        Get statistics for a word category.

        Args:
            category: The category to analyze

        Returns:
            Dictionary containing category statistics
        """
        return self.category_repo.get_category_stats(category)

    def get_word_difficulty(self, word: str) -> float:
        """
        Calculate the difficulty score of a word.

        Args:
            word: The word to analyze

        Returns:
            Difficulty score between 0 and 1
        """
        stats = self.get_word_stats(word)
        frequency = stats.get('frequency', 0)
        length = len(word)
        
        # Normalize length (assuming max length of 15)
        length_factor = length / 15
        
        # Normalize frequency (using log scale)
        if frequency > 0:
            frequency_factor = 1 - (math.log(frequency) / math.log(1000))
        else:
            frequency_factor = 1
            
        # Weight factors (can be adjusted)
        length_weight = 0.4
        frequency_weight = 0.6
        
        difficulty = (length_factor * length_weight) + (frequency_factor * frequency_weight)
        return min(max(difficulty, 0), 1)  # Ensure result is between 0 and 1

    def get_category_difficulty(self, category: str) -> float:
        """
        Calculate the average difficulty of words in a category.

        Args:
            category: The category to analyze

        Returns:
            Average difficulty score between 0 and 1
        """
        words = self.category_repo.get_category_words(category)
        if not words:
            return 0
            
        total_difficulty = sum(self.get_word_difficulty(word) for word in words)
        return total_difficulty / len(words)

    def get_learning_progress(self) -> Dict:
        """
        Get overall learning progress statistics.

        Returns:
            Dictionary containing learning progress metrics
        """
        total_words = self.word_repo.get_entry_count()
        total_categories = len(self.category_repo.get_categories())
        
        stats = {
            'total_words': total_words,
            'total_categories': total_categories,
            'average_frequency': self.word_repo.get_average_frequency(),
            'unique_words_today': self.word_repo.get_unique_words_today(),
            'categories': {}
        }
        
        for category in self.category_repo.get_categories():
            stats['categories'][category] = {
                'word_count': len(self.category_repo.get_category_words(category)),
                'difficulty': self.get_category_difficulty(category)
            }
            
        return stats