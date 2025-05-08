from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from ..manager import DatabaseManager
from .base_repository import BaseRepository

class QLearningRepository(BaseRepository):
    """Repository for managing Q-learning states and values."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the Q-learning repository."""
        super().__init__(db_manager, "q_learning_states")
        
    def record_state_action(self, state_hash: str, action: str, q_value: float, visit_count: int = 1) -> None:
        """
        Record a state-action pair.
        
        Args:
            state_hash: Hash of the state
            action: Action taken
            q_value: Q-value for the state-action pair
            visit_count: Number of times this state-action pair has been visited
        """
        # First try to update existing record
        self.db.execute_query("""
            UPDATE q_learning_states
            SET q_value = ?,
                visit_count = visit_count + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE state_hash = ? AND action = ?
        """, (q_value, visit_count, state_hash, action))
        
        # If no rows were updated, insert new record
        if self.db.get_scalar("SELECT changes()") == 0:
            self.db.execute_query("""
                INSERT INTO q_learning_states (state_hash, action, q_value, visit_count)
                VALUES (?, ?, ?, ?)
            """, (state_hash, action, q_value, visit_count))
        
    def get_q_value(self, state_hash: str, action: str) -> Optional[Dict[str, Any]]:
        """
        Get the Q-value for a state-action pair.
        
        Args:
            state_hash: Hash of the state
            action: Action taken
            
        Returns:
            Dictionary containing q_value and visit_count, or None if not found
        """
        result = self.db.execute_query("""
            SELECT q_value, visit_count
            FROM q_learning_states
            WHERE state_hash = ? AND action = ?
        """, (state_hash, action))
        
        return result[0] if result else None
        
    def update_q_value(self, state_hash: str, action: str, new_q_value: float) -> bool:
        """
        Update the Q-value for a state-action pair.
        
        Args:
            state_hash: Hash of the state
            action: Action taken
            new_q_value: New Q-value to set
            
        Returns:
            True if successful
        """
        self.db.execute_query("""
            UPDATE q_learning_states
            SET q_value = ?,
                visit_count = visit_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE state_hash = ? AND action = ?
        """, (new_q_value, state_hash, action))
        return True
        
    def get_state_actions(self, state_hash: str) -> List[Dict[str, Any]]:
        """
        Get all actions for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            List of dictionaries containing action data
        """
        return self.db.execute_query("""
            SELECT action, q_value, visit_count
            FROM q_learning_states
            WHERE state_hash = ?
            ORDER BY q_value DESC
        """, (state_hash,))
        
    def get_best_action(self, state_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get the best action for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            Dictionary containing action data or None if no actions found
        """
        result = self.db.execute_query("""
            SELECT action, q_value, visit_count
            FROM q_learning_states
            WHERE state_hash = ?
            ORDER BY q_value DESC
            LIMIT 1
        """, (state_hash,))
        
        return result[0] if result else None
        
    def cleanup_old_states(self, days: int = 30) -> int:
        """
        Clean up old state-action pairs.
        
        Args:
            days: Number of days to keep entries for
            
        Returns:
            Number of entries removed
        """
        # Delete old states
        self.db.execute("""
            DELETE FROM q_learning_states
            WHERE updated_at < datetime('now', ?)
        """, (f'-{days} days',))
        
        # Get number of rows affected
        result = self.db.get_scalar("SELECT changes()")
        return result or 0
        
    def get_max_q_value(self, state_hash: str) -> float:
        """
        Get the maximum Q-value for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            Maximum Q-value for the state
        """
        result = self.db.execute_query("""
            SELECT MAX(q_value) as max_q
            FROM q_learning_states
            WHERE state_hash = ?
        """, (state_hash,))
        
        return result[0]['max_q'] if result and result[0]['max_q'] is not None else 0.0
        
    def get_state_statistics(self, state_hash: str) -> Dict:
        """
        Get statistics for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            Dictionary containing:
                - total_actions: Total number of actions tried
                - total_visits: Total number of visits across all actions
                - average_q_value: Average Q-value across all actions
        """
        result = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_actions,
                SUM(visit_count) as total_visits,
                AVG(q_value) as average_q_value
            FROM q_learning_states
            WHERE state_hash = ?
        """, (state_hash,))
        
        if not result:
            return {
                'total_actions': 0,
                'total_visits': 0,
                'average_q_value': 0.0
            }
            
        return {
            'total_actions': result[0]['total_actions'],
            'total_visits': result[0]['total_visits'],
            'average_q_value': result[0]['average_q_value']
        }
        
    def get_learning_stats(self) -> Dict:
        """
        Get overall statistics about the Q-learning process.
        
        Returns:
            Dictionary containing:
                - total_states: Total number of unique states
                - total_actions: Total number of state-action pairs
                - average_q_value: Average Q-value across all pairs
                - most_explored_state: State with most visits
                - least_explored_state: State with fewest visits
        """
        result = self.db.execute_query("""
            WITH state_stats AS (
                SELECT 
                    state_hash,
                    SUM(visit_count) as total_visits,
                    AVG(q_value) as avg_q_value
                FROM q_learning_states
                GROUP BY state_hash
            )
            SELECT 
                COUNT(DISTINCT state_hash) as total_states,
                COUNT(*) as total_actions,
                AVG(q_value) as average_q_value,
                (SELECT state_hash FROM state_stats ORDER BY total_visits DESC LIMIT 1) as most_explored_state,
                (SELECT state_hash FROM state_stats ORDER BY total_visits ASC LIMIT 1) as least_explored_state
            FROM q_learning_states
        """)
        
        return result[0] if result else {
            'total_states': 0,
            'total_actions': 0,
            'average_q_value': 0.0,
            'most_explored_state': None,
            'least_explored_state': None
        }
        
    def get_least_explored_action(self, state_hash: str) -> Optional[str]:
        """
        Get the least explored action for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            Least explored action, or None if all actions explored equally
        """
        result = self.db.execute_query("""
            SELECT action
            FROM q_learning_states
            WHERE state_hash = ?
            ORDER BY visit_count ASC
            LIMIT 1
        """, (state_hash,))
        
        return result[0]['action'] if result else None
        
    def reset_state(self, state_hash: str) -> bool:
        """
        Reset all Q-values for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            True if the reset was successful
        """
        self.db.execute_query("""
            DELETE FROM q_learning_states
            WHERE state_hash = ?
        """, (state_hash,))
        return True
        
    def optimize_q_values(self, min_visits: int = 5) -> int:
        """
        Remove state-action pairs with too few visits.
        
        Args:
            min_visits: Minimum number of visits required
            
        Returns:
            Number of pairs removed
        """
        result = self.db.execute_query("""
            DELETE FROM q_learning_states
            WHERE visit_count < ?
            SELECT changes()
        """, (min_visits,))
        
        return result[0]['changes()'] if result else 0

    def get_state_visit_count(self, state_hash: str) -> int:
        """
        Get the total number of visits for a state.
        
        Args:
            state_hash: Hash of the state
            
        Returns:
            Total number of visits for the state
        """
        result = self.db.execute_query("""
            SELECT SUM(visit_count) as total_visits
            FROM q_learning_states
            WHERE state_hash = ?
        """, (state_hash,))
        
        return result[0]['total_visits'] if result and result[0]['total_visits'] is not None else 0

    def get_action_visit_count(self, state_hash: str, action: str) -> int:
        """
        Get the number of visits for a specific action in a state.
        
        Args:
            state_hash: Hash of the state
            action: Action to check
            
        Returns:
            Number of visits for the action
        """
        result = self.db.execute_query("""
            SELECT visit_count
            FROM q_learning_states
            WHERE state_hash = ? AND action = ?
        """, (state_hash, action))
        
        return result[0]['visit_count'] if result else 0

    def get_state_action_stats(self, state_hash: str, action: str) -> Dict:
        """
        Get detailed statistics for a state-action pair.
        
        Args:
            state_hash: Hash of the state
            action: Action to check
            
        Returns:
            Dictionary containing:
                - q_value: Current Q-value
                - visit_count: Number of visits
                - last_updated: Timestamp of last update
                - average_reward: Average reward received
        """
        result = self.db.execute_query("""
            SELECT 
                q_value,
                visit_count,
                updated_at as last_updated,
                (SELECT AVG(reward) FROM q_learning_rewards 
                 WHERE state_hash = ? AND action = ?) as average_reward
            FROM q_learning_states
            WHERE state_hash = ? AND action = ?
        """, (state_hash, action, state_hash, action))
        
        if not result:
            return {
                'q_value': 0.0,
                'visit_count': 0,
                'last_updated': None,
                'average_reward': 0.0
            }
            
        return {
            'q_value': result[0]['q_value'],
            'visit_count': result[0]['visit_count'],
            'last_updated': result[0]['last_updated'],
            'average_reward': result[0]['average_reward'] or 0.0
        }

    def get_learning_progress(self) -> Dict:
        """
        Get metrics about the learning progress.
        
        Returns:
            Dictionary containing:
                - states_explored: Number of unique states explored
                - actions_tried: Total number of actions tried
                - average_q_value: Average Q-value across all pairs
                - exploration_rate: Current exploration rate
                - learning_rate: Current learning rate
                - success_rate: Rate of successful actions
        """
        result = self.db.execute_query("""
            SELECT 
                COUNT(DISTINCT state_hash) as states_explored,
                COUNT(*) as actions_tried,
                AVG(q_value) as average_q_value,
                AVG(1.0 / (visit_count ** 0.5)) as exploration_rate,
                AVG(1.0 / visit_count) as learning_rate,
                (SELECT COUNT(*) FROM q_learning_rewards WHERE reward > 0) * 1.0 / 
                (SELECT COUNT(*) FROM q_learning_rewards) as success_rate
            FROM q_learning_states
        """)
        
        return result[0] if result else {
            'states_explored': 0,
            'actions_tried': 0,
            'average_q_value': 0.0,
            'exploration_rate': 1.0,
            'learning_rate': 1.0,
            'success_rate': 0.0
        }

    def backup_q_values(self, backup_name: str) -> bool:
        """
        Create a backup of all Q-values.
        
        Args:
            backup_name: Name of the backup
            
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS q_learning_backups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.db.execute_query("""
                INSERT INTO q_learning_backups (name)
                VALUES (?)
            """, (backup_name,))
            
            backup_id = self.db.get_scalar("SELECT last_insert_rowid()")
            
            self.db.execute_query(f"""
                CREATE TABLE q_learning_backup_{backup_id} AS
                SELECT * FROM q_learning_states
            """)
            
            return True
        except Exception as e:
            print(f"Backup failed: {str(e)}")
            return False

    def restore_q_values(self, backup_name: str) -> bool:
        """
        Restore Q-values from a backup.
        
        Args:
            backup_name: Name of the backup to restore
            
        Returns:
            True if restore was successful, False otherwise
        """
        try:
            backup_id = self.db.get_scalar("""
                SELECT id FROM q_learning_backups
                WHERE name = ?
            """, (backup_name,))
            
            if not backup_id:
                return False
                
            self.db.execute_query("""
                DELETE FROM q_learning_states
            """)
            
            self.db.execute_query(f"""
                INSERT INTO q_learning_states
                SELECT * FROM q_learning_backup_{backup_id}
            """)
            
            return True
        except Exception as e:
            print(f"Restore failed: {str(e)}")
            return False

    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the q_learning_states table.
        
        Returns:
            The number of entries
        """
        query = "SELECT COUNT(*) FROM q_learning_states"
        return self.db.get_scalar(query) or 0

    def get_exploration_rate(self, state_hash: str, action: str) -> float:
        """
        Get the exploration rate for a state-action pair.
        
        Args:
            state_hash: Hash of the state
            action: Action taken
            
        Returns:
            Exploration rate (1/sqrt(visit_count))
        """
        result = self.db.execute_query("""
            SELECT visit_count
            FROM q_learning_states
            WHERE state_hash = ? AND action = ?
        """, (state_hash, action))
        
        visit_count = result[0]['visit_count'] if result else 0
        return 1.0 / ((visit_count + 1) ** 0.5)

    def get_state_history(self, state_hash: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of Q-value updates for a state.
        
        Args:
            state_hash: Hash of the state
            limit: Maximum number of records to return
            
        Returns:
            List of dictionaries containing action, q_value, and timestamp
        """
        return self.db.execute_query("""
            SELECT action, q_value, updated_at
            FROM q_learning_states
            WHERE state_hash = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (state_hash, limit)) 