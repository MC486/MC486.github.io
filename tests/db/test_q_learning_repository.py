import unittest
import os
import tempfile
import sys
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from database.manager import DatabaseManager
from database.repositories.q_learning_repository import QLearningRepository

class TestQLearningRepository(unittest.TestCase):
    def setUp(self):
        """Set up test database and repository."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.initialize_database()  # Initialize database with schema
        self.q_learning_repo = QLearningRepository(self.db_manager)

    def tearDown(self):
        """Clean up the database connection."""
        self.db_manager.close()

    def test_record_state_action(self):
        # Test recording a new state-action pair
        state_hash = "test_state_1"
        action = "test_action_1"
        q_value = 0.5
        visit_count = 1

        self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify the state-action pair was recorded
        result = self.q_learning_repo.get_q_value(state_hash, action)
        self.assertIsNotNone(result)
        self.assertEqual(result['q_value'], q_value)
        self.assertEqual(result['visit_count'], visit_count)

    def test_update_q_value(self):
        # Test updating an existing state-action pair
        state_hash = "test_state_2"
        action = "test_action_2"
        initial_q_value = 0.5
        initial_visit_count = 1

        # Record initial state
        self.q_learning_repo.record_state_action(state_hash, action, initial_q_value, initial_visit_count)
        
        # Update Q-value
        new_q_value = 0.7
        self.q_learning_repo.update_q_value(state_hash, action, new_q_value)
        
        # Verify the update
        result = self.q_learning_repo.get_q_value(state_hash, action)
        self.assertEqual(result['q_value'], new_q_value)
        self.assertEqual(result['visit_count'], initial_visit_count + 1)

    def test_get_q_value(self):
        # Test retrieving Q-value for non-existent state-action pair
        result = self.q_learning_repo.get_q_value("nonexistent_state", "nonexistent_action")
        self.assertIsNone(result)

    def test_get_state_actions(self):
        # Test retrieving all actions for a state
        state_hash = "test_state_3"
        actions = ["action_1", "action_2", "action_3"]
        
        # Record multiple actions for the same state
        for action in actions:
            self.q_learning_repo.record_state_action(state_hash, action, 0.5, 1)
        
        # Verify all actions are retrieved
        result = self.q_learning_repo.get_state_actions(state_hash)
        self.assertEqual(len(result), len(actions))
        for action in result:
            self.assertIn(action['action'], actions)

    def test_get_best_action(self):
        # Test retrieving the best action for a state
        state_hash = "test_state_4"
        actions = [
            ("action_1", 0.3),
            ("action_2", 0.7),
            ("action_3", 0.5)
        ]
        
        # Record actions with different Q-values
        for action, q_value in actions:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, 1)
        
        # Verify the best action is retrieved
        best_action = self.q_learning_repo.get_best_action(state_hash)
        self.assertEqual(best_action['action'], "action_2")
        self.assertEqual(best_action['q_value'], 0.7)

    def test_get_state_statistics(self):
        # Test retrieving statistics for a state
        state_hash = "test_state_5"
        actions = [
            ("action_1", 0.3, 2),
            ("action_2", 0.7, 3),
            ("action_3", 0.5, 1)
        ]
        
        # Record actions with different Q-values and visit counts
        for action, q_value, visit_count in actions:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify statistics are calculated correctly
        stats = self.q_learning_repo.get_state_statistics(state_hash)
        self.assertEqual(stats['total_actions'], len(actions))
        self.assertEqual(stats['total_visits'], sum(visit_count for _, _, visit_count in actions))
        self.assertAlmostEqual(stats['average_q_value'], sum(q_value for _, q_value, _ in actions) / len(actions))

    def test_cleanup_old_states(self):
        # Test cleaning up old states
        # Create some old states
        old_timestamp = datetime.now().timestamp() - 86400  # 1 day old
        self.q_learning_repo.record_state_action("old_state_1", "action_1", 0.5, 1)
        self.q_learning_repo.record_state_action("old_state_2", "action_2", 0.6, 1)
        
        # Create some recent states
        self.q_learning_repo.record_state_action("recent_state_1", "action_1", 0.7, 1)
        self.q_learning_repo.record_state_action("recent_state_2", "action_2", 0.8, 1)
        
        # Clean up states older than 12 hours
        self.q_learning_repo.cleanup_old_states(43200)  # 12 hours in seconds
        
        # Verify old states are removed
        self.assertIsNone(self.q_learning_repo.get_q_value("old_state_1", "action_1"))
        self.assertIsNone(self.q_learning_repo.get_q_value("old_state_2", "action_2"))
        
        # Verify recent states are kept
        self.assertIsNotNone(self.q_learning_repo.get_q_value("recent_state_1", "action_1"))
        self.assertIsNotNone(self.q_learning_repo.get_q_value("recent_state_2", "action_2"))

    def test_get_all_states(self):
        # Test retrieving all states
        states = [
            ("state_1", "action_1", 0.5),
            ("state_2", "action_2", 0.6),
            ("state_3", "action_3", 0.7)
        ]
        
        # Record multiple states
        for state_hash, action, q_value in states:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, 1)
        
        # Verify all states are retrieved
        result = self.q_learning_repo.get_all_states()
        self.assertEqual(len(result), len(states))
        for state in result:
            self.assertTrue(any(state['state_hash'] == s[0] for s in states))

    def test_get_state_action_count(self):
        # Test retrieving the count of state-action pairs
        state_hash = "test_state_6"
        actions = ["action_1", "action_2", "action_3"]
        
        # Record multiple actions for the same state
        for action in actions:
            self.q_learning_repo.record_state_action(state_hash, action, 0.5, 1)
        
        # Verify the count is correct
        count = self.q_learning_repo.get_state_action_count(state_hash)
        self.assertEqual(count, len(actions))

    def test_get_most_visited_states(self):
        # Test retrieving the most visited states
        states = [
            ("state_1", "action_1", 0.5, 3),
            ("state_2", "action_2", 0.6, 5),
            ("state_3", "action_3", 0.7, 2)
        ]
        
        # Record states with different visit counts
        for state_hash, action, q_value, visit_count in states:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify the most visited states are retrieved in correct order
        result = self.q_learning_repo.get_most_visited_states(2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['state_hash'], "state_2")
        self.assertEqual(result[1]['state_hash'], "state_1")

    def test_get_state_visit_count(self):
        # Test getting total visits for a state
        state_hash = "test_state_7"
        actions = [
            ("action_1", 0.5, 2),
            ("action_2", 0.6, 3),
            ("action_3", 0.7, 1)
        ]
        
        # Record actions with different visit counts
        for action, q_value, visit_count in actions:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify total visits are correct
        total_visits = self.q_learning_repo.get_state_visit_count(state_hash)
        self.assertEqual(total_visits, sum(visit_count for _, _, visit_count in actions))

    def test_get_action_visit_count(self):
        # Test getting visits for a specific action
        state_hash = "test_state_8"
        action = "test_action"
        visit_count = 5
        
        # Record action with specific visit count
        self.q_learning_repo.record_state_action(state_hash, action, 0.5, visit_count)
        
        # Verify visit count is correct
        count = self.q_learning_repo.get_action_visit_count(state_hash, action)
        self.assertEqual(count, visit_count)

    def test_get_state_action_stats(self):
        # Test getting detailed stats for a state-action pair
        state_hash = "test_state_9"
        action = "test_action"
        q_value = 0.7
        visit_count = 3
        
        # Record state-action pair
        self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify stats are correct
        stats = self.q_learning_repo.get_state_action_stats(state_hash, action)
        self.assertEqual(stats['q_value'], q_value)
        self.assertEqual(stats['visit_count'], visit_count)
        self.assertIsNotNone(stats['last_updated'])
        self.assertEqual(stats['average_reward'], 0.0)

    def test_get_learning_progress(self):
        # Test getting learning progress metrics
        states = [
            ("state_1", "action_1", 0.5, 2),
            ("state_2", "action_2", 0.6, 3),
            ("state_3", "action_3", 0.7, 1)
        ]
        
        # Record multiple states with different characteristics
        for state_hash, action, q_value, visit_count in states:
            self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Verify progress metrics
        progress = self.q_learning_repo.get_learning_progress()
        self.assertEqual(progress['states_explored'], len(states))
        self.assertEqual(progress['actions_tried'], len(states))
        self.assertGreater(progress['exploration_rate'], 0)
        self.assertGreater(progress['learning_rate'], 0)

    def test_backup_and_restore_q_values(self):
        # Test backing up and restoring Q-values
        state_hash = "test_state_10"
        action = "test_action"
        q_value = 0.8
        visit_count = 4
        
        # Record initial state
        self.q_learning_repo.record_state_action(state_hash, action, q_value, visit_count)
        
        # Create backup
        backup_name = "test_backup"
        self.assertTrue(self.q_learning_repo.backup_q_values(backup_name))
        
        # Modify the state
        self.q_learning_repo.update_q_value(state_hash, action, 0.9)
        
        # Restore from backup
        self.assertTrue(self.q_learning_repo.restore_q_values(backup_name))
        
        # Verify restored values
        result = self.q_learning_repo.get_q_value(state_hash, action)
        self.assertEqual(result['q_value'], q_value)
        self.assertEqual(result['visit_count'], visit_count)

if __name__ == '__main__':
    unittest.main() 