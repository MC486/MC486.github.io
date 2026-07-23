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
        """Get all actions and their statistics for a given state.

        Returns a dict keyed by action: ``{action: {reward, visit_count}}``.
        """
        query = """
            SELECT action, avg_reward, visit_count
            FROM mcts_actions
            WHERE state = ?
        """
        rows = self.db_manager.execute_query(query, (state,))
        return {
            row['action']: {
                'reward': row['avg_reward'],
                'visit_count': row['visit_count']
            }
            for row in rows
        }
        
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
        """Remove old entries from the database. Returns the number removed."""
        count = self.db_manager.get_scalar(
            """
            SELECT COUNT(*) FROM mcts_states
            WHERE updated_at < datetime('now', ?)
            """,
            (f"-{max_age_days} days",)
        ) or 0
        self.db_manager.execute(
            """
            DELETE FROM mcts_states
            WHERE updated_at < datetime('now', ?)
            """,
            (f"-{max_age_days} days",)
        )
        return count
        
    def get_learning_stats(self):
        """Get statistics about the learning process."""
        query = """
            SELECT 
                COUNT(DISTINCT state) as total_states,
                COUNT(DISTINCT action) as total_actions,
                AVG(visit_count) as avg_visits,
                MAX(avg_reward) as max_reward,
                AVG(avg_reward) as average_reward
            FROM mcts_actions
        """
        stats = self.db_manager.execute_query(query)[0]
        most_visited = self.db_manager.execute_query("""
            SELECT state FROM mcts_states
            ORDER BY visit_count DESC
            LIMIT 1
        """)
        stats['most_visited_state'] = most_visited[0]['state'] if most_visited else None
        return stats

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
            SELECT avg_reward, visit_count, updated_at
            FROM mcts_actions
            WHERE state = ? AND action = ?
        """, (state, action))
        
        if not result:
            return {
                'reward': 0.0,
                'visit_count': 0,
                'last_updated': None
            }
            
        return {
            'reward': result[0]['avg_reward'],
            'visit_count': result[0]['visit_count'],
            'last_updated': result[0]['updated_at']
        } 