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
        """Set up a temporary database for testing."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db = DatabaseManager(self.db_path)
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_context_manager(self):
        """Test that DatabaseManager works as a context manager."""
        with DatabaseManager(self.db_path) as db:
            self.assertIsInstance(db, DatabaseManager)
            
    def test_create_tables(self):
        """Test that tables are created correctly."""
        self.db.create_tables()
        
        # Check that all tables exist
        self.assertTrue(self.db.table_exists('markov_transitions'))
        self.assertTrue(self.db.table_exists('q_learning_states'))
        self.assertTrue(self.db.table_exists('game_sessions'))
        self.assertTrue(self.db.table_exists('game_turns'))
        self.assertTrue(self.db.table_exists('words'))
        self.assertTrue(self.db.table_exists('categories'))
        self.assertTrue(self.db.table_exists('performance_metrics'))
        self.assertTrue(self.db.table_exists('system_errors'))
        
    def test_drop_tables(self):
        """Test that tables can be dropped."""
        self.db.create_tables()
        self.db.drop_tables()
        
        # Check that all tables are gone
        self.assertFalse(self.db.table_exists('markov_transitions'))
        self.assertFalse(self.db.table_exists('q_learning_states'))
        self.assertFalse(self.db.table_exists('game_sessions'))
        self.assertFalse(self.db.table_exists('game_turns'))
        self.assertFalse(self.db.table_exists('words'))
        self.assertFalse(self.db.table_exists('categories'))
        self.assertFalse(self.db.table_exists('performance_metrics'))
        self.assertFalse(self.db.table_exists('system_errors'))
        
    def test_execute_query(self):
        """Test executing a query and getting results."""
        self.db.create_tables()
        
        # Insert test data
        self.db.execute("""
            INSERT INTO words (word, category_id, frequency)
            VALUES (?, ?, ?)
        """, ('test', 1, 10))
        
        # Query the data
        results = self.db.execute_query("""
            SELECT * FROM words WHERE word = ?
        """, ('test',))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'test')
        self.assertEqual(results[0]['category_id'], 1)
        self.assertEqual(results[0]['frequency'], 10)
        
    def test_execute(self):
        """Test executing a query without results."""
        self.db.create_tables()
        
        # Insert test data
        self.db.execute("""
            INSERT INTO words (word, category_id, frequency)
            VALUES (?, ?, ?)
        """, ('test', 1, 10))
        
        # Verify the data was inserted
        count = self.db.get_scalar("SELECT COUNT(*) FROM words")
        self.assertEqual(count, 1)
        
    def test_execute_many(self):
        """Test executing multiple queries."""
        self.db.create_tables()
        
        # Insert multiple rows
        params = [
            ('test1', 1, 10),
            ('test2', 2, 20),
            ('test3', 3, 30)
        ]
        
        self.db.execute_many("""
            INSERT INTO words (word, category_id, frequency)
            VALUES (?, ?, ?)
        """, params)
        
        # Verify all rows were inserted
        count = self.db.get_scalar("SELECT COUNT(*) FROM words")
        self.assertEqual(count, 3)
        
    def test_get_one(self):
        """Test getting a single row."""
        self.db.create_tables()
        
        # Insert test data
        self.db.execute("""
            INSERT INTO words (word, category_id, frequency)
            VALUES (?, ?, ?)
        """, ('test', 1, 10))
        
        # Get the row
        row = self.db.get_one("""
            SELECT * FROM words WHERE word = ?
        """, ('test',))
        
        self.assertIsNotNone(row)
        self.assertEqual(row['word'], 'test')
        self.assertEqual(row['category_id'], 1)
        self.assertEqual(row['frequency'], 10)
        
    def test_get_scalar(self):
        """Test getting a scalar value."""
        self.db.create_tables()
        
        # Insert test data
        self.db.execute("""
            INSERT INTO words (word, category_id, frequency)
            VALUES (?, ?, ?)
        """, ('test', 1, 10))
        
        # Get the count
        count = self.db.get_scalar("SELECT COUNT(*) FROM words")
        self.assertEqual(count, 1)
        
        # Get the frequency
        frequency = self.db.get_scalar("SELECT frequency FROM words WHERE word = ?", ('test',))
        self.assertEqual(frequency, 10)
        
    def test_transaction_rollback(self):
        """Test that transactions are rolled back on error."""
        self.db.create_tables()
        
        # Try to insert invalid data (should fail due to UNIQUE constraint)
        with self.assertRaises(Exception):
            self.db.execute("""
                INSERT INTO words (word, category_id, frequency)
                VALUES (?, ?, ?)
            """, ('test', 1, 10))
            
            # This should fail due to UNIQUE constraint
            self.db.execute("""
                INSERT INTO words (word, category_id, frequency)
                VALUES (?, ?, ?)
            """, ('test', 1, 10))
            
        # Verify no data was inserted
        count = self.db.get_scalar("SELECT COUNT(*) FROM words")
        self.assertEqual(count, 0)
        
    def test_index_creation(self):
        """Test that indexes are created correctly."""
        self.db.create_tables()
        
        # Verify indexes exist
        indexes = self.db.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name LIKE 'idx_%'
        """)
        
        index_names = {row['name'] for row in indexes}
        expected_indexes = {
            'idx_markov_current_state',
            'idx_q_learning_state_hash',
            'idx_game_sessions_player',
            'idx_words_category',
            'idx_performance_metrics_name'
        }
        
        self.assertEqual(index_names, expected_indexes)

if __name__ == '__main__':
    unittest.main() 