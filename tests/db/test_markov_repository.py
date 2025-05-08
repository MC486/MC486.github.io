import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.markov_repository import MarkovRepository

class TestMarkovRepository(unittest.TestCase):
    def setUp(self):
        """Set up test database and repository."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.initialize_database()  # Initialize database with schema
        self.markov_repo = MarkovRepository(self.db_manager)
        
    def tearDown(self):
        """Clean up the database connection."""
        self.db_manager.close()
        
    def test_record_transition(self):
        """Test recording a transition."""
        # Record a transition
        self.markov_repo.record_transition("abc", "d")
        
        # Verify the transition was recorded
        result = self.db_manager.execute_query("""
            SELECT count FROM markov_transitions 
            WHERE current_state = ? AND next_state = ?
        """, ("abc", "d"))
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['count'], 1)
        
    def test_get_transition_probability(self):
        """Test getting transition probability."""
        # Record multiple transitions
        self.markov_repo.record_transition("abc", "d", 3)
        self.markov_repo.record_transition("abc", "e", 7)
        
        # Test probability calculation
        prob_d = self.markov_repo.get_transition_probability("abc", "d")
        prob_e = self.markov_repo.get_transition_probability("abc", "e")
        
        self.assertEqual(prob_d, 0.3)  # 3/10
        self.assertEqual(prob_e, 0.7)  # 7/10
        
    def test_get_next_states(self):
        """Test getting next states."""
        # Record multiple transitions
        self.markov_repo.record_transition("abc", "d", 3)
        self.markov_repo.record_transition("abc", "e", 7)
        self.markov_repo.record_transition("abc", "f", 5)
        
        # Get next states
        next_states = self.markov_repo.get_next_states("abc", limit=2)
        
        self.assertEqual(len(next_states), 2)
        self.assertEqual(next_states[0]['next_state'], "e")  # Highest probability
        self.assertEqual(next_states[1]['next_state'], "f")  # Second highest
        
    def test_get_state_stats(self):
        """Test getting state statistics."""
        # Record multiple transitions
        self.markov_repo.record_transition("abc", "d", 3)
        self.markov_repo.record_transition("abc", "e", 7)
        self.markov_repo.record_transition("abc", "f", 5)
        
        # Get stats
        stats = self.markov_repo.get_state_stats("abc")
        
        self.assertEqual(stats['total_transitions'], 15)
        self.assertEqual(stats['unique_next_states'], 3)
        self.assertEqual(stats['most_common_next'], "e")
        self.assertGreater(stats['entropy'], 0)
        
    def test_bulk_update_transitions(self):
        """Test bulk updating transitions."""
        # Prepare transitions
        transitions = [
            ("abc", "d", 3),
            ("abc", "e", 7),
            ("def", "g", 5)
        ]
        
        # Bulk update
        self.markov_repo.bulk_update_transitions(transitions)
        
        # Verify updates
        result = self.db_manager.execute_query("""
            SELECT current_state, next_state, count 
            FROM markov_transitions 
            ORDER BY current_state, next_state
        """)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['count'], 3)
        self.assertEqual(result[1]['count'], 7)
        self.assertEqual(result[2]['count'], 5)
        
    def test_get_chain_stats(self):
        """Test getting chain statistics."""
        # Record transitions
        self.markov_repo.record_transition("abc", "d", 3)
        self.markov_repo.record_transition("abc", "e", 7)
        self.markov_repo.record_transition("def", "g", 5)
        
        # Get stats
        stats = self.markov_repo.get_chain_stats()
        
        self.assertEqual(stats['total_states'], 2)
        self.assertEqual(stats['total_transitions'], 15)
        self.assertGreater(stats['average_entropy'], 0)
        self.assertIsNotNone(stats['most_uncertain_state'])
        self.assertIsNotNone(stats['most_certain_state'])
        
    def test_cleanup_old_transitions(self):
        """Test cleaning up old transitions."""
        # Record a transition
        self.markov_repo.record_transition("abc", "d")
        
        # Force the updated_at to be old
        self.db_manager.execute_query("""
            UPDATE markov_transitions 
            SET updated_at = datetime('now', '-31 days')
        """)
        
        # Cleanup
        removed = self.markov_repo.cleanup_old_transitions(days=30)
        
        self.assertEqual(removed, 1)
        
        # Verify removal
        result = self.db_manager.execute_query("""
            SELECT COUNT(*) as count FROM markov_transitions
        """)
        self.assertEqual(result[0]['count'], 0)

if __name__ == '__main__':
    unittest.main() 