# tests/test_main.py
# Unit tests for the main game entry point.

import unittest
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from engine.engine_core import setup_logging, main
from database.manager import DatabaseManager
from database.repository_manager import RepositoryManager
from engine.game_loop import GameLoop
from engine.game_state import GameState
from engine.input_handler import InputHandler
from core.game_events_manager import GameEventManager

class TestMain(unittest.TestCase):
    def setUp(self):
        """
        Sets up the test environment.
        """
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Create temporary logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """
        Cleans up after each test.
        """
        # Remove test logs directory and its contents
        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                except PermissionError:
                    pass  # Ignore permission errors during cleanup
            try:
                self.log_dir.rmdir()
            except (PermissionError, OSError):
                pass  # Ignore if directory is not empty or locked
        
    def test_setup_logging(self):
        """
        Test that logging is properly configured.
        """
        # Setup logging
        log_file = setup_logging()
        
        # Verify logging configuration
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)  # Changed to DEBUG to match implementation
        self.assertEqual(len(root_logger.handlers), 2)  # File and console handlers
        
        # Verify handlers
        handlers = root_logger.handlers
        self.assertTrue(any(isinstance(h, logging.FileHandler) for h in handlers))
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))
        
        # Verify log file was created
        self.assertTrue(Path(log_file).exists())
            
    @patch('engine.game_loop.GameLoop')
    @patch('database.manager.DatabaseManager')
    @patch('database.repository_manager.RepositoryManager')
    @patch('engine.game_state.GameState')
    @patch('engine.input_handler.InputHandler')
    @patch('core.game_events_manager.GameEventManager')
    def test_main_success(self, mock_event_manager, mock_input_handler, mock_game_state, 
                         mock_repo_manager, mock_db_manager, mock_game_loop):
        """
        Test successful game execution.
        """
        # Mock game loop
        mock_loop = MagicMock()
        mock_game_loop.return_value = mock_loop
        
        # Mock database manager
        mock_db = MagicMock()
        mock_db_manager.return_value = mock_db
        
        # Mock repository manager
        mock_repo = MagicMock()
        mock_repo.repositories = {'word': MagicMock(), 'category': MagicMock()}
        mock_repo_manager.return_value = mock_repo
        
        # Mock game state
        mock_state = MagicMock()
        mock_game_state.return_value = mock_state
        
        # Mock input handler
        mock_input = MagicMock()
        mock_input_handler.return_value = mock_input
        
        # Mock event manager
        mock_events = MagicMock()
        mock_event_manager.return_value = mock_events
        
        # Run main
        main()
        
        # Verify game loop was started
        mock_loop.start.assert_called_once()
        mock_repo.cleanup_old_entries.assert_called_once_with(force=True)
        
    @patch('engine.game_loop.GameLoop')
    @patch('database.manager.DatabaseManager')
    @patch('database.repository_manager.RepositoryManager')
    @patch('engine.game_state.GameState')
    @patch('engine.input_handler.InputHandler')
    @patch('core.game_events_manager.GameEventManager')
    def test_main_crash(self, mock_event_manager, mock_input_handler, mock_game_state,
                       mock_repo_manager, mock_db_manager, mock_game_loop):
        """
        Test game crash handling.
        """
        # Mock game loop to raise an exception
        mock_loop = MagicMock()
        mock_loop.start.side_effect = Exception("Test crash")
        mock_game_loop.return_value = mock_loop
        
        # Mock other components
        mock_db = MagicMock()
        mock_db_manager.return_value = mock_db
        
        mock_repo = MagicMock()
        mock_repo.repositories = {'word': MagicMock(), 'category': MagicMock()}
        mock_repo_manager.return_value = mock_repo
        
        # Run main and verify it raises the exception
        with self.assertRaises(Exception):
            main()
            
    def test_log_directory_creation(self):
        """
        Test that log directory is created if it doesn't exist.
        """
        # Remove logs directory if it exists
        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                except PermissionError:
                    pass
            try:
                self.log_dir.rmdir()
            except (PermissionError, OSError):
                pass
            
        # Setup logging
        log_file = setup_logging()
        
        # Verify directory was created and log file exists
        self.assertTrue(self.log_dir.exists())
        self.assertTrue(Path(log_file).exists())

if __name__ == '__main__':
    unittest.main() 