from typing import Dict, List, Optional, Union
from datetime import datetime
from ..manager import DatabaseManager
from .base_repository import BaseRepository

class MCTSRepository(BaseRepository):
    """Repository for MCTS state and action data."""
    
    def __init__(self, db_manager):
        """Initialize the MCTS repository."""
        super().__init__(db_manager, table_name="mcts_states")
        self.db_manager = db_manager
        
    def get_state_actions(self, state):
        """Get all actions and their statistics for a given state."""
        query = """
            SELECT action, avg_reward, visit_count
            FROM mcts_actions
            WHERE state = ?
        """
        return self.db_manager.execute_query(query, (state,))
        
    def get_best_action(self, state):
        """Get the best action for a given state based on average reward."""
        query = """
            SELECT action
            FROM mcts_actions
            WHERE state = ?
            ORDER BY avg_reward DESC
            LIMIT 1
        """
        result = self.db_manager.execute_query(query, (state,))
        return result[0]['action'] if result else None
        
    def record_simulation(self, state, action, reward):
        """Record the results of a simulation."""
        # Update state statistics
        self.db_manager.execute(
            """
            INSERT OR REPLACE INTO mcts_states (state, visit_count)
            VALUES (?, COALESCE((SELECT visit_count FROM mcts_states WHERE state = ?), 0) + 1)
            """,
            (state, state)
        )
        
        # Update action statistics
        self.db_manager.execute(
            """
            INSERT OR REPLACE INTO mcts_actions 
            (state, action, avg_reward, visit_count)
            VALUES (
                ?,
                ?,
                COALESCE(
                    (SELECT (avg_reward * visit_count + ?) / (visit_count + 1)
                     FROM mcts_actions 
                     WHERE state = ? AND action = ?),
                    ?
                ),
                COALESCE((SELECT visit_count FROM mcts_actions WHERE state = ? AND action = ?), 0) + 1
            )
            """,
            (state, action, reward, state, action, reward, state, action)
        )
        
    def cleanup_old_entries(self, max_age_days=30):
        """Remove old entries from the database."""
        self.db_manager.execute(
            """
            DELETE FROM mcts_states
            WHERE last_updated < datetime('now', ?)
            """,
            (f"-{max_age_days} days",)
        )
        
    def get_learning_stats(self):
        """Get statistics about the learning process."""
        query = """
            SELECT 
                COUNT(DISTINCT state) as total_states,
                COUNT(DISTINCT action) as total_actions,
                AVG(visit_count) as avg_visits,
                MAX(avg_reward) as max_reward
            FROM mcts_actions
        """
        return self.db_manager.execute_query(query)[0]

    def get_state_action_stats(self, state: str, action: str) -> Dict:
        """
        Get statistics for a state-action pair.
        
        Args:
            state: Game state
            action: Action taken
            
        Returns:
            Dict containing:
                - reward: Average reward
                - visit_count: Number of visits
                - last_updated: Last update timestamp
        """
        result = self.db.execute_query("""
            SELECT reward, visit_count, updated_at
            FROM mcts_simulations
            WHERE state = ? AND action = ?
        """, (state, action))
        
        if not result:
            return {
                'reward': 0.0,
                'visit_count': 0,
                'last_updated': None
            }
            
        return {
            'reward': result[0]['reward'],
            'visit_count': result[0]['visit_count'],
            'last_updated': result[0]['updated_at']
        } 