from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from ..manager import DatabaseManager
from .base_repository import BaseRepository
from collections import defaultdict

class MarkovRepository(BaseRepository):
    """Repository for managing Markov chain transitions."""
    
    def __init__(self, db_manager: DatabaseManager, game_id: Optional[int] = None):
        """Initialize the Markov chain repository.
        
        Args:
            db_manager: Database manager instance
            game_id: Optional game ID. If not provided, must be set before using methods that require it.
        """
        super().__init__(db_manager, 'markov_transitions')
        self.game_id = game_id
        
    def set_game_id(self, game_id: int) -> None:
        """Set the game ID for this repository instance."""
        self.game_id = game_id
        
    def _check_game_id(self) -> None:
        """Check if game_id is set, raise RuntimeError if not."""
        if self.game_id is None:
            raise RuntimeError("game_id must be set before using this method")
        
    def record_transition(self, current_state: str, next_state: str, count: int = 1, visit_count: int = 1) -> None:
        """
        Record a state transition.
        
        Args:
            current_state: Current state
            next_state: Next state
            count: Number of times this transition occurred (default: 1)
            visit_count: Number of times this state has been visited (default: 1)
        """
        self._check_game_id()
        self.db_manager.execute_query("""
            INSERT INTO markov_transitions (game_id, current_state, next_state, count, total_transitions, visit_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, current_state, next_state) DO UPDATE SET
                count = count + ?,
                total_transitions = total_transitions + ?,
                visit_count = visit_count + ?,
                updated_at = CURRENT_TIMESTAMP
        """, (self.game_id, current_state, next_state, count, count, visit_count, count, count, visit_count))
        
    def get_transition_probability(self, current_state: str, next_state: str) -> float:
        """
        Get the probability of transitioning from current_state to next_state.
        
        Args:
            current_state: Current state
            next_state: Next state
            
        Returns:
            Probability of the transition
        """
        self._check_game_id()
        result = self.db_manager.execute_query("""
            SELECT 
                CAST(count AS FLOAT) / (
                    SELECT SUM(count) 
                    FROM markov_transitions 
                    WHERE game_id = ? AND current_state = ?
                ) as probability
            FROM markov_transitions
            WHERE game_id = ? AND current_state = ? AND next_state = ?
        """, (self.game_id, current_state, self.game_id, current_state, next_state))
        
        return result[0]['probability'] if result else 0.0
        
    def get_next_states(self, current_state: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get possible next states and their probabilities.
        
        Args:
            current_state: Current state
            limit: Maximum number of next states to return
            
        Returns:
            List of dictionaries containing next states and probabilities
        """
        self._check_game_id()
        query = """
            WITH total AS (
                SELECT SUM(count) as total
                FROM markov_transitions
                WHERE game_id = ? AND current_state = ?
            )
            SELECT 
                next_state,
                count,
                CAST(count AS FLOAT) / (SELECT total FROM total) as probability,
                visit_count
            FROM markov_transitions
            WHERE game_id = ? AND current_state = ?
            ORDER BY count DESC
        """
        params = [self.game_id, current_state, self.game_id, current_state]
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            
        return self.db_manager.execute_query(query, tuple(params))
        
    def get_state_stats(self, state: str) -> Dict[str, Any]:
        """
        Get statistics for a state.
        
        Args:
            state: State to get statistics for
            
        Returns:
            Dictionary containing state statistics
        """
        self._check_game_id()
        result = self.db_manager.execute_query("""
            WITH stats AS (
                SELECT 
                    COUNT(*) as transition_count,
                    SUM(count) as total_transitions,
                    COUNT(DISTINCT next_state) as unique_next_states,
                    AVG(CAST(count AS FLOAT) / NULLIF(total_transitions, 0)) as avg_probability,
                    MAX(CAST(count AS FLOAT) / NULLIF(total_transitions, 0)) as max_probability,
                    SUM(visit_count) as total_visits,
                    MIN(updated_at) as first_seen,
                    MAX(updated_at) as last_seen
                FROM markov_transitions
                WHERE game_id = ? AND current_state = ?
            ),
            most_common AS (
                SELECT next_state
                FROM markov_transitions
                WHERE game_id = ? AND current_state = ?
                ORDER BY count DESC
                LIMIT 1
            ),
            entropy AS (
                SELECT 
                    -SUM(
                        (CAST(count AS FLOAT) / total) * 
                        LOG(CAST(count AS FLOAT) / total)
                    ) as entropy
                FROM (
                    SELECT 
                        count,
                        SUM(count) OVER () as total
                    FROM markov_transitions
                    WHERE game_id = ? AND current_state = ?
                )
            )
            SELECT 
                stats.*,
                most_common.next_state as most_common_next,
                entropy.entropy
            FROM stats
            LEFT JOIN most_common
            LEFT JOIN entropy
        """, (self.game_id, state, self.game_id, state, self.game_id, state))
        
        return result[0] if result else None
        
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics for the Markov chain.
        
        Returns:
            Dictionary containing chain statistics
        """
        self._check_game_id()
        result = self.db_manager.execute_query("""
            WITH state_entropy AS (
                SELECT 
                    current_state,
                    -SUM(
                        (CAST(count AS FLOAT) / total) * 
                        LOG(CAST(count AS FLOAT) / total)
                    ) as entropy
                FROM (
                    SELECT 
                        current_state,
                        count,
                        SUM(count) OVER (PARTITION BY current_state) as total
                    FROM markov_transitions
                    WHERE game_id = ?
                )
                GROUP BY current_state
            ),
            chain_stats AS (
                SELECT 
                    COUNT(DISTINCT current_state) as total_states,
                    SUM(count) as total_transitions,
                    COUNT(*) as total_counts,
                    AVG(CAST(count AS FLOAT) / NULLIF(total_transitions, 0)) as avg_probability,
                    MAX(CAST(count AS FLOAT) / NULLIF(total_transitions, 0)) as max_probability,
                    SUM(visit_count) as total_visits,
                    MIN(updated_at) as first_seen,
                    MAX(updated_at) as last_seen
                FROM markov_transitions
                WHERE game_id = ?
            ),
            most_uncertain AS (
                SELECT current_state as most_uncertain_state
                FROM state_entropy
                ORDER BY entropy DESC
                LIMIT 1
            ),
            most_certain AS (
                SELECT current_state as most_certain_state
                FROM state_entropy
                WHERE entropy > 0
                ORDER BY entropy ASC
                LIMIT 1
            )
            SELECT 
                chain_stats.*,
                most_uncertain.most_uncertain_state,
                most_certain.most_certain_state,
                (SELECT AVG(entropy) FROM state_entropy) as avg_entropy
            FROM chain_stats
            LEFT JOIN most_uncertain
            LEFT JOIN most_certain
        """, (self.game_id, self.game_id))
        
        return result[0] if result else None
        
    def bulk_update_transitions(self, transitions: List[Tuple[str, str, int]]) -> None:
        """
        Bulk update transition counts.
        
        Args:
            transitions: List of (current_state, next_state, count) tuples
        """
        self._check_game_id()
        params = [(self.game_id, current_state, next_state, count, count, count, count, count)
                 for current_state, next_state, count in transitions]
        
        self.db_manager.execute_many("""
            INSERT INTO markov_transitions (game_id, current_state, next_state, count, total_transitions, visit_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(game_id, current_state, next_state) DO UPDATE SET
                count = count + ?,
                total_transitions = total_transitions + ?,
                visit_count = visit_count + 1,
                updated_at = CURRENT_TIMESTAMP
        """, params)
        
    def cleanup_old_transitions(self, days: int = 30) -> int:
        """
        Remove transitions older than the specified number of days.
        
        Args:
            days: Number of days to keep transitions for
            
        Returns:
            Number of transitions removed
        """
        self._check_game_id()
        result = self.db_manager.execute_query("""
            WITH deleted AS (
                DELETE FROM markov_transitions
                WHERE game_id = ? AND updated_at < datetime('now', ?)
                RETURNING *
            )
            SELECT COUNT(*) as count FROM deleted
        """, (self.game_id, f'-{days} days'))
        
        return result[0]['count'] if result else 0
        
    def get_transitions(self) -> Dict[str, Dict[str, float]]:
        """
        Get all transitions and their probabilities.
        
        Returns:
            Dictionary mapping current states to dictionaries of next states and probabilities
        """
        self._check_game_id()
        transitions = defaultdict(dict)
        
        results = self.db_manager.execute_query("""
            WITH totals AS (
                SELECT current_state, SUM(count) as total
                FROM markov_transitions
                WHERE game_id = ?
                GROUP BY current_state
            )
            SELECT 
                m.current_state,
                m.next_state,
                CAST(m.count AS FLOAT) / t.total as probability
            FROM markov_transitions m
            JOIN totals t ON m.current_state = t.current_state
            WHERE m.game_id = ?
        """, (self.game_id, self.game_id))
        
        for row in results:
            transitions[row['current_state']][row['next_state']] = row['probability']
            
        return dict(transitions)
        
    def bulk_record_transitions(self, transitions: dict) -> None:
        """
        Record multiple transitions at once.
        
        Args:
            transitions: Dictionary mapping current states to dictionaries of next states and counts
        """
        self._check_game_id()
        params = []
        for current_state, next_states in transitions.items():
            for next_state, count in next_states.items():
                params.append((self.game_id, current_state, next_state, count, count, 1))
                
        self.db_manager.execute_many("""
            INSERT INTO markov_transitions (game_id, current_state, next_state, count, total_transitions, visit_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, current_state, next_state) DO UPDATE SET
                count = count + excluded.count,
                total_transitions = total_transitions + excluded.total_transitions,
                visit_count = visit_count + excluded.visit_count,
                updated_at = CURRENT_TIMESTAMP
        """, params)
        
    def get_state_probabilities(self, state: str) -> dict:
        """
        Get transition probabilities for a state.
        
        Args:
            state: State to get probabilities for
            
        Returns:
            Dictionary mapping next states to probabilities
        """
        self._check_game_id()
        results = self.db_manager.execute_query("""
            WITH total AS (
                SELECT SUM(count) as total
                FROM markov_transitions
                WHERE game_id = ? AND current_state = ?
            )
            SELECT 
                next_state,
                CAST(count AS FLOAT) / (SELECT total FROM total) as probability
            FROM markov_transitions
            WHERE game_id = ? AND current_state = ?
        """, (self.game_id, state, self.game_id, state))
        
        return {row['next_state']: row['probability'] for row in results}
        
    def save_transitions(self, transitions: Dict[str, Dict[str, float]]) -> None:
        """
        Save the current state of the Markov chain.
        
        Args:
            transitions: Dictionary mapping current states to dictionaries of next states and their probabilities
                {current_state: {next_state: probability}}
        """
        self._check_game_id()
        
        # Convert probabilities to counts (multiply by 1000 to preserve decimal places)
        transition_counts = []
        for current_state, next_states in transitions.items():
            total_count = sum(int(prob * 1000) for prob in next_states.values())
            for next_state, prob in next_states.items():
                count = int(prob * 1000)  # Convert probability to count
                transition_counts.append((current_state, next_state, count, total_count))
        
        # Bulk insert/update transitions
        self.db_manager.execute_many("""
            INSERT INTO markov_transitions (
                game_id, current_state, next_state, count, 
                total_transitions, visit_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(game_id, current_state, next_state) DO UPDATE SET
                count = ?,
                total_transitions = ?,
                updated_at = CURRENT_TIMESTAMP
        """, [(self.game_id, curr, next_, count, total, count, total) 
              for curr, next_, count, total in transition_counts]) 