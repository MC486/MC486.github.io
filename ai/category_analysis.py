from typing import Dict, List, Set, DefaultDict, Tuple, Optional, Any
from collections import defaultdict
import logging
from database.repositories.word_repository import WordRepository
from database.repositories.category_repository import CategoryRepository
from database.manager import DatabaseManager
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager

logger = logging.getLogger(__name__)

class CategoryAnalyzer:
    """
    Analyzes word categories and their relationships for AI decision making.
    Provides category-based statistical data used by various AI models.
    """
    def __init__(self, word_repo: WordRepository, category_repo: CategoryRepository):
        """
        Initialize the category analyzer.

        Args:
            word_repo: Repository for word usage data
            category_repo: Repository for word categories
        """
        self.word_repo = word_repo
        self.category_repo = category_repo
        
        # Category frequency tracking
        self.category_frequencies: DefaultDict[int, int] = defaultdict(int)
        self.total_categories = 0
        
        # Category-word relationships
        self.category_words: DefaultDict[int, Set[str]] = defaultdict(set)
        
        # Initialize category data
        self._initialize_categories()

    def _initialize_categories(self) -> None:
        """Initialize category data from the database."""
        categories = self.category_repo.get_all_categories()
        for category in categories:
            words = self.word_repo.get_words_by_category(category.id)
            self.category_words[category.id] = set(words)
            self.category_frequencies[category.id] = len(words)
            self.total_categories += len(words)

    def get_category_score(self, word: str) -> float:
        """
        Calculate a score based on the word's category frequency.
        
        Args:
            word: Word to score
            
        Returns:
            Score between 0 and 1, where higher scores indicate rarer categories
        """
        word_data = self.word_repo.get_word_by_spelling(word)
        if not word_data or not word_data.category_id:
            return 0.5  # Default score for uncategorized words
            
        category_freq = self.category_frequencies.get(word_data.category_id, 0)
        if category_freq == 0:
            return 0.5
            
        # Convert frequency to a score between 0 and 1
        # Higher scores for words in less common categories
        score = 1.0 - (category_freq / self.total_categories)
        return max(0.1, min(0.9, score))  # Keep score between 0.1 and 0.9

    def get_category_stats(self) -> Dict[str, Any]:
        """
        Get statistics about word categories.
        
        Returns:
            Dict containing category statistics
        """
        categories = self.category_repo.get_all_categories()
        stats = {
            "total_categories": len(categories),
            "category_counts": {},
            "category_word_counts": {}
        }
        
        for category in categories:
            stats["category_counts"][category.name] = self.category_frequencies.get(category.id, 0)
            stats["category_word_counts"][category.name] = len(self.category_words.get(category.id, set()))
            
        return stats

    def get_words_by_category(self, category_id: int) -> List[str]:
        """
        Get all words in a specific category.
        
        Args:
            category_id: ID of the category
            
        Returns:
            List of words in the category
        """
        return list(self.category_words.get(category_id, set()))

    def get_categories_by_word(self, word: str) -> List[int]:
        """
        Get all categories a word belongs to.
        
        Args:
            word: Word to check
            
        Returns:
            List of category IDs
        """
        word_data = self.word_repo.get_word_by_spelling(word)
        if not word_data or not word_data.category_id:
            return []
        return [word_data.category_id]

    def update_category_frequencies(self, word: str) -> None:
        """
        Update category frequencies when a word is used.
        
        Args:
            word: Word that was used
        """
        word_data = self.word_repo.get_word_by_spelling(word)
        if word_data and word_data.category_id:
            self.category_frequencies[word_data.category_id] += 1
            self.total_categories += 1 