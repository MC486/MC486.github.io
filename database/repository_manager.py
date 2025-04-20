import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from .manager import DatabaseManager
from .repositories.word_repository import WordRepository
from .repositories.category_repository import CategoryRepository
from .repositories.naive_bayes_repository import NaiveBayesRepository
from .repositories.mcts_repository import MCTSRepository
from .repositories.q_learning_repository import QLearningRepository
from .repositories.markov_repository import MarkovRepository
from .repositories.game_repository import GameRepository

logger = logging.getLogger(__name__)

class RepositoryManager:
    """
    Manages all repositories and their lifecycle.
    Handles initialization, cleanup, and monitoring of repositories.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the repository manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.repositories: Dict[str, Any] = {}
        self.last_cleanup: Dict[str, datetime] = {}
        self.cleanup_intervals = {
            'word': timedelta(hours=24),  # Clean word repository daily
            'category': timedelta(hours=24),  # Clean category repository daily
            'naive_bayes': timedelta(hours=12),  # Clean Naive Bayes data every 12 hours
            'mcts': timedelta(hours=12),  # Clean MCTS data every 12 hours
            'q_learning': timedelta(hours=12),  # Clean Q-learning data every 12 hours
            'markov_chain': timedelta(hours=12),  # Clean Markov Chain data every 12 hours
        }
        
        # Initialize all repositories
        self._initialize_repositories()
        
    def _initialize_repositories(self) -> None:
        """Initialize all repositories."""
        self.repositories['word'] = WordRepository(self.db_manager)
        self.repositories['category'] = CategoryRepository(self.db_manager)
        self.repositories['naive_bayes'] = NaiveBayesRepository(self.db_manager)
        self.repositories['mcts'] = MCTSRepository(self.db_manager)
        self.repositories['q_learning'] = QLearningRepository(self.db_manager)
        self.repositories['markov_chain'] = MarkovRepository(self.db_manager)
        self.repositories['game'] = GameRepository(self.db_manager)
        
        # Initialize last cleanup times
        now = datetime.now()
        for repo_name in self.repositories:
            self.last_cleanup[repo_name] = now
            
        logger.info("All repositories initialized")
        
    def get_repository(self, name: str) -> Any:
        """
        Get a repository by name.
        
        Args:
            name: Name of the repository
            
        Returns:
            Repository instance
        """
        return self.repositories.get(name)
        
    def cleanup_old_entries(self, force: bool = False, days: int = 30) -> None:
        """
        Clean up old entries in all repositories.
        
        Args:
            force: If True, perform cleanup regardless of interval
            days: Number of days to keep entries for (default: 30)
        """
        now = datetime.now()
        
        for repo_name, repo in self.repositories.items():
            interval = self.cleanup_intervals.get(repo_name)
            last_clean = self.last_cleanup.get(repo_name)
            
            if force or (interval and last_clean and (now - last_clean) >= interval):
                try:
                    logger.info(f"Cleaning up {repo_name} repository")
                    repo.cleanup_old_entries(days)
                    self.last_cleanup[repo_name] = now
                except Exception as e:
                    logger.error(f"Error cleaning up {repo_name} repository: {str(e)}")
                    
    def get_repository_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all repositories.
        
        Returns:
            Dictionary of repository statistics
        """
        stats = {}
        for repo_name, repo in self.repositories.items():
            try:
                stats[repo_name] = {
                    'last_cleanup': self.last_cleanup.get(repo_name),
                    'next_cleanup': self.last_cleanup.get(repo_name) + self.cleanup_intervals.get(repo_name),
                    'entry_count': repo.get_entry_count(),
                    'size_bytes': repo.get_size_bytes()
                }
            except Exception as e:
                logger.error(f"Error getting stats for {repo_name} repository: {str(e)}")
                stats[repo_name] = {'error': str(e)}
                
        return stats 