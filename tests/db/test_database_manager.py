import unittest
import os
import tempfile
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_create_tables(self):
        """Test that all tables are created correctly."""
        # Create tables
        self.db_manager.create_tables()
        
        # Verify tables exist
        tables = self.db_manager.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        table_names = {table['name'] for table in tables}
        expected_tables = {
            'games',
            'game_moves',
            'words',
            'categories',
            'ai_metrics',
            'dictionary_domains'
        }
        self.assertTrue(expected_tables.issubset(table_names))
        
    def test_indexes_created(self):
        """Test that all indexes are created correctly."""
        # Create tables and indexes
        self.db_manager.create_tables()
        
        # Verify indexes exist
        indexes = self.db_manager.execute_query("""
            SELECT name, tbl_name FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        
        # Group indexes by table
        table_indexes = {}
        for idx in indexes:
            if idx['tbl_name'] not in table_indexes:
                table_indexes[idx['tbl_name']] = set()
            table_indexes[idx['tbl_name']].add(idx['name'])
            
        # Verify games indexes
        expected_games_indexes = {
            'idx_games_player_name',
            'idx_games_status',
            'idx_games_difficulty',
            'idx_games_start_time'
        }
        self.assertTrue(expected_games_indexes.issubset(table_indexes.get('games', set())))
        
        # Verify game_moves indexes
        expected_moves_indexes = {
            'idx_game_moves_game_id',
            'idx_game_moves_timestamp'
        }
        self.assertTrue(expected_moves_indexes.issubset(table_indexes.get('game_moves', set())))
        
    def test_triggers_created(self):
        """Test that all triggers are created correctly."""
        # Create tables and triggers
        self.db_manager.create_tables()
        
        # Verify triggers exist
        triggers = self.db_manager.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='trigger'
        """)
        trigger_names = {trigger['name'] for trigger in triggers}
        expected_triggers = {
            'update_games_timestamp',
            'update_words_timestamp',
            'update_categories_timestamp',
            'update_domain_word_count'
        }
        self.assertTrue(expected_triggers.issubset(trigger_names))
        
    def test_timestamp_triggers(self):
        """Test that timestamp update triggers work correctly."""
        # Create tables
        self.db_manager.create_tables()
        
        # Insert a game
        game_id = self.db_manager.execute_query("""
            INSERT INTO games (player_name, difficulty, max_attempts)
            VALUES (?, ?, ?)
        """, ("test_player", "medium", 10))[0]['id']
        
        # Get initial timestamps
        initial = self.db_manager.execute_query("""
            SELECT created_at, updated_at FROM games WHERE id = ?
        """, (game_id,))[0]
        
        # Update the game
        self.db_manager.execute_query("""
            UPDATE games SET status = ? WHERE id = ?
        """, ("completed", game_id))
        
        # Get updated timestamps
        updated = self.db_manager.execute_query("""
            SELECT created_at, updated_at FROM games WHERE id = ?
        """, (game_id,))[0]
        
        # Verify timestamps
        self.assertEqual(initial['created_at'], updated['created_at'])
        self.assertNotEqual(initial['updated_at'], updated['updated_at'])
        
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are enforced."""
        # Create tables
        self.db_manager.create_tables()
        
        # Try to insert a game move without a game
        with self.assertRaises(Exception):
            self.db_manager.execute_query("""
                INSERT INTO game_moves (game_id, word, is_valid, feedback)
                VALUES (?, ?, ?, ?)
            """, (999, "test", True, "test"))
            
    def test_check_constraints(self):
        """Test that check constraints are enforced."""
        # Create tables
        self.db_manager.create_tables()
        
        # Try to insert a game with invalid difficulty
        with self.assertRaises(Exception):
            self.db_manager.execute_query("""
                INSERT INTO games (player_name, difficulty, max_attempts)
                VALUES (?, ?, ?)
            """, ("test_player", "invalid", 10))
            
        # Try to insert a game with invalid max_attempts
        with self.assertRaises(Exception):
            self.db_manager.execute_query("""
                INSERT INTO games (player_name, difficulty, max_attempts)
                VALUES (?, ?, ?)
            """, ("test_player", "medium", 0))
            
    def test_domain_word_count_trigger(self):
        """Test that the domain word count trigger works correctly."""
        # Create tables
        self.db_manager.create_tables()
        
        # Create a dictionary domain
        domain_id = self.db_manager.execute_query("""
            INSERT INTO dictionary_domains (name, description)
            VALUES (?, ?)
        """, ("test_domain", "Test domain"))[0]['id']
        
        # Insert a word
        self.db_manager.execute_query("""
            INSERT INTO words (word, domain_id)
            VALUES (?, ?)
        """, ("test_word", domain_id))
        
        # Verify word count
        result = self.db_manager.execute_query("""
            SELECT word_count FROM dictionary_domains WHERE id = ?
        """, (domain_id,))[0]
        self.assertEqual(result['word_count'], 1)

if __name__ == '__main__':
    unittest.main() 