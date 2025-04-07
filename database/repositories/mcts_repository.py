from typing import Dict, List, Optional, Union
from datetime import datetime
from ..manager import DatabaseManager
from .base_repository import BaseRepository

class MCTSRepository(BaseRepository):
    """Repository for storing MCTS model data."""
    
    def __init__(self, db_manager):
        """Initialize the MCTS repository."""
        super().__init__(db_manager)
        self.table_name = "mcts"
        
        # Create mcts_simulations table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS mcts_simulations (
                state TEXT NOT NULL,
                action TEXT NOT NULL,
                reward REAL NOT NULL,
                visit_count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (state, action)
            )
        """)
        
    def record_simulation(self, state: str, action: str, reward: float, 
                         visit_count: int = 1) -> None:
        """
        Record a simulation result for a state-action pair.
        
        Args:
            state: Current game state
            action: Action taken
            reward: Reward received
            visit_count: Number of times this state-action pair was visited
        """
        self.db.execute_query("""
            INSERT INTO mcts_simulations (state, action, reward, visit_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(state, action) DO UPDATE SET
                reward = (reward * visit_count + ?) / (visit_count + 1),
                visit_count = visit_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (state, action, reward, visit_count, reward))
        
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
        
    def get_state_actions(self, state: str) -> Dict[str, Dict]:
        """
        Get all actions and their statistics for a state.
        
        Args:
            state: Game state
            
        Returns:
            Dict[str, Dict]: Dictionary of actions and their statistics
        """
        results = self.db.execute_query("""
            SELECT action, reward, visit_count
            FROM mcts_simulations
            WHERE state = ?
        """, (state,))
        
        return {
            row['action']: {
                'reward': row['reward'],
                'visit_count': row['visit_count']
            }
            for row in results
        }
        
    def get_best_action(self, state: Union[str, List[str]]) -> Optional[str]:
        """
        Get the best action for a state based on average reward.
        
        Args:
            state: Game state (string or list of letters)
            
        Returns:
            Optional[str]: Best action if found, None otherwise
        """
        # Convert list to string if needed
        state_str = ''.join(state) if isinstance(state, list) else state
        
        result = self.db.execute_query("""
            SELECT action
            FROM mcts_simulations
            WHERE state = ?
            ORDER BY reward DESC
            LIMIT 1
        """, (state_str,))
        
        return result[0]['action'] if result else None
        
    def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Remove entries that haven't been updated in the specified number of days.
        
        Args:
            days: Number of days after which to remove entries
            
        Returns:
            int: Number of entries removed
        """
        # First get the count of rows that will be deleted
        result = self.db.execute_query("""
            SELECT COUNT(*) as count
            FROM mcts_simulations
            WHERE updated_at < datetime('now', ?)
        """, (f"-{days} days",))
        count = result[0]['count'] if result else 0
        
        # Then delete the rows
        self.db.execute_query("""
            DELETE FROM mcts_simulations
            WHERE updated_at < datetime('now', ?)
        """, (f"-{days} days",))
        
        return count
        
    def get_learning_stats(self) -> Dict:
        """
        Get overall statistics about the MCTS model.
        
        Returns:
            Dict containing:
                - total_states: Total unique states
                - total_actions: Total unique actions
                - average_reward: Average reward across all simulations
                - most_visited_state: State with most visits
        """
        result = self.db.execute_query("""
            WITH state_stats AS (
                SELECT 
                    state,
                    SUM(visit_count) as total_visits
                FROM mcts_simulations
                GROUP BY state
            )
            SELECT 
                COUNT(DISTINCT state) as total_states,
                COUNT(DISTINCT action) as total_actions,
                AVG(reward) as average_reward,
                (SELECT state FROM state_stats ORDER BY total_visits DESC LIMIT 1) as most_visited_state
            FROM mcts_simulations
        """)
        
        return result[0] if result else {
            'total_states': 0,
            'total_actions': 0,
            'average_reward': 0.0,
            'most_visited_state': None
        } 