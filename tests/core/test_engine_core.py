# tests/test_main.py
# Unit tests for the main game entry point.

import unittest
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from engine.engine_core import setup_logging, main

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
        self.log_dir = Path("test_logs")
        self.log_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        """
        Cleans up after each test.
        """
        # Remove test logs directory and its contents
        for log_file in self.log_dir.glob("*.log"):
            log_file.unlink()
        self.log_dir.rmdir()
        
    def test_setup_logging(self):
        """
        Test that logging is properly configured.
        """
        with patch('pathlib.Path') as mock_path:
            # Mock the logs directory
            mock_log_dir = MagicMock()
            mock_log_dir.mkdir.return_value = None
            mock_path.return_value = mock_log_dir
            
            # Setup logging
            log_file = setup_logging()
            
            # Verify logging configuration
            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.INFO)
            self.assertEqual(len(root_logger.handlers), 2)  # File and console handlers
            
            # Verify handlers
            handlers = root_logger.handlers
            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in handlers))
            self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in handlers))
            
    @patch('engine.game_loop.GameLoop')
    def test_main_success(self, mock_game_loop):
        """
        Test successful game execution.
        """
        # Mock game loop
        mock_loop = MagicMock()
        mock_game_loop.return_value = mock_loop
        
        # Run main
        main()
        
        # Verify game loop was started
        mock_loop.start.assert_called_once()
        
    @patch('engine.game_loop.GameLoop')
    def test_main_crash(self, mock_game_loop):
        """
        Test game crash handling.
        """
        # Mock game loop to raise an exception
        mock_loop = MagicMock()
        mock_loop.start.side_effect = Exception("Test crash")
        mock_game_loop.return_value = mock_loop
        
        # Run main and verify it raises the exception
        with self.assertRaises(Exception):
            main()
            
    def test_log_directory_creation(self):
        """
        Test that log directory is created if it doesn't exist.
        """
        with patch('pathlib.Path') as mock_path:
            # Mock the logs directory to simulate it not existing
            mock_log_dir = MagicMock()
            mock_log_dir.exists.return_value = False
            mock_path.return_value = mock_log_dir
            
            # Setup logging
            setup_logging()
            
            # Verify directory was created
            mock_log_dir.mkdir.assert_called_once_with(exist_ok=True)

if __name__ == '__main__':
    unittest.main() 