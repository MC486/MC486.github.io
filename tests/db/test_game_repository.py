import unittest
import os
import tempfile
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.game_repository import GameRepository

class TestGameRepository(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database and test repository."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.create_tables()
        
        self.game_repo = GameRepository(self.db_manager)
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_create_game(self):
        """Test creating a new game."""
        game_id = self.game_repo.create_game(
            player_name="test_player",
            difficulty="medium",
            max_attempts=10
        )
        
        self.assertIsNotNone(game_id)
        game = self.game_repo.get_by_id(game_id)
        self.assertEqual(game['player_name'], "test_player")
        self.assertEqual(game['difficulty'], "medium")
        self.assertEqual(game['max_attempts'], 10)
        self.assertEqual(game['status'], "in_progress")
        
    def test_get_active_game(self):
        """Test getting active game for a player."""
        # Create an active game
        game_id = self.game_repo.create_game(
            player_name="test_player",
            difficulty="medium",
            max_attempts=10
        )
        
        # Get active game
        active_game = self.game_repo.get_active_game("test_player")
        self.assertIsNotNone(active_game)
        self.assertEqual(active_game['id'], game_id)
        
        # Complete the game
        self.game_repo.update_game_status(game_id, "completed", 100)
        
        # Verify no active game
        active_game = self.game_repo.get_active_game("test_player")
        self.assertIsNone(active_game)
        
    def test_get_player_games(self):
        """Test getting all games for a player."""
        # Create multiple games
        for i in range(3):
            self.game_repo.create_game(
                player_name="test_player",
                difficulty="medium",
                max_attempts=10
            )
            
        # Get player games
        games = self.game_repo.get_player_games("test_player")
        self.assertEqual(len(games), 3)
        
        # Test limit
        games = self.game_repo.get_player_games("test_player", limit=2)
        self.assertEqual(len(games), 2)
        
    def test_update_game_status(self):
        """Test updating game status."""
        # Create a game
        game_id = self.game_repo.create_game(
            player_name="test_player",
            difficulty="medium",
            max_attempts=10
        )
        
        # Update status
        success = self.game_repo.update_game_status(game_id, "completed", 100)
        self.assertTrue(success)
        
        # Verify update
        game = self.game_repo.get_by_id(game_id)
        self.assertEqual(game['status'], "completed")
        self.assertEqual(game['score'], 100)
        self.assertIsNotNone(game['end_time'])
        
    def test_record_move(self):
        """Test recording game moves."""
        # Create a game
        game_id = self.game_repo.create_game(
            player_name="test_player",
            difficulty="medium",
            max_attempts=10
        )
        
        # Record moves
        move_id1 = self.game_repo.record_move(game_id, "word1", True, "Correct!")
        move_id2 = self.game_repo.record_move(game_id, "word2", False, "Invalid word")
        
        self.assertIsNotNone(move_id1)
        self.assertIsNotNone(move_id2)
        
        # Get moves
        moves = self.game_repo.get_game_moves(game_id)
        self.assertEqual(len(moves), 2)
        self.assertEqual(moves[0]['word'], "word1")
        self.assertEqual(moves[1]['word'], "word2")
        
    def test_get_player_stats(self):
        """Test getting player statistics."""
        # Create completed games
        for i in range(3):
            game_id = self.game_repo.create_game(
                player_name="test_player",
                difficulty="medium",
                max_attempts=10
            )
            self.game_repo.update_game_status(game_id, "completed", 100 * (i + 1))
            
        # Get stats
        stats = self.game_repo.get_player_stats("test_player")
        
        # Verify stats
        self.assertEqual(stats['total_games'], 3)
        self.assertEqual(stats['completed_games'], 3)
        self.assertEqual(stats['average_score'], 200.0)
        self.assertEqual(stats['highest_score'], 300)
        
    def test_get_difficulty_stats(self):
        """Test getting difficulty statistics."""
        # Create games with different difficulties
        difficulties = ["easy", "medium", "hard"]
        for diff in difficulties:
            game_id = self.game_repo.create_game(
                player_name="test_player",
                difficulty=diff,
                max_attempts=10
            )
            self.game_repo.update_game_status(game_id, "completed", 100)
            
        # Get stats for medium difficulty
        stats = self.game_repo.get_difficulty_stats("medium")
        
        # Verify stats
        self.assertEqual(stats['total_games'], 1)
        self.assertEqual(stats['average_score'], 100.0)
        self.assertEqual(stats['completion_rate'], 1.0)
        
    def test_get_recent_games(self):
        """Test getting recent games."""
        # Create multiple games
        for i in range(5):
            self.game_repo.create_game(
                player_name=f"player{i}",
                difficulty="medium",
                max_attempts=10
            )
            
        # Get recent games
        recent = self.game_repo.get_recent_games(3)
        self.assertEqual(len(recent), 3)
        
    def test_get_game_summary(self):
        """Test getting game summary."""
        # Create a game
        game_id = self.game_repo.create_game(
            player_name="test_player",
            difficulty="medium",
            max_attempts=10
        )
        
        # Record moves
        self.game_repo.record_move(game_id, "word1", True, "Correct!")
        self.game_repo.record_move(game_id, "word2", False, "Invalid")
        self.game_repo.record_move(game_id, "word3", True, "Correct!")
        
        # Get summary
        summary = self.game_repo.get_game_summary(game_id)
        
        # Verify summary
        self.assertEqual(summary['total_moves'], 3)
        self.assertEqual(summary['valid_moves'], 2)
        self.assertEqual(summary['invalid_moves'], 1)
        self.assertEqual(len(summary['moves']), 3)

if __name__ == '__main__':
    unittest.main() 