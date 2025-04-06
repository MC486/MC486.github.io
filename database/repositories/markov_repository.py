from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..manager import DatabaseManager

class MarkovRepository:
    """Repository for managing Markov chain transitions and probabilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def record_transition(self, current_state: str, next_state: str, count: int = 1) -> None:
        """
        Record a transition between states and update its count.
        
        Args:
            current_state: The current state (e.g., current letter sequence)
            next_state: The next state (e.g., next letter)
            count: Number of times this transition occurred (default: 1)
        """
        self.db.execute_query("""
            INSERT INTO markov_transitions (current_state, next_state, count)
            VALUES (?, ?, ?)
            ON CONFLICT(current_state, next_state) DO UPDATE SET
                count = count + ?,
                updated_at = CURRENT_TIMESTAMP
        """, (current_state, next_state, count, count))
        
    def get_transition_probability(self, current_state: str, next_state: str) -> float:
        """
        Get the probability of transitioning from current_state to next_state.
        
        Args:
            current_state: The current state
            next_state: The next state
            
        Returns:
            Probability of the transition (0.0 to 1.0)
        """
        result = self.db.execute_query("""
            SELECT 
                t.count as transition_count,
                SUM(t2.count) as total_count
            FROM markov_transitions t
            LEFT JOIN markov_transitions t2 ON t.current_state = t2.current_state
            WHERE t.current_state = ? AND t.next_state = ?
            GROUP BY t.current_state, t.next_state, t.count
        """, (current_state, next_state))
        
        if not result:
            return 0.0
            
        transition_count = result[0]['transition_count']
        total_count = result[0]['total_count']
        
        return transition_count / total_count if total_count > 0 else 0.0
        
    def get_next_states(self, current_state: str, limit: int = 5) -> List[Dict]:
        """
        Get the most likely next states from the current state.
        
        Args:
            current_state: The current state
            limit: Maximum number of next states to return
            
        Returns:
            List of dictionaries containing next_state and probability
        """
        return self.db.execute_query("""
            SELECT 
                next_state,
                count * 1.0 / SUM(count) OVER (PARTITION BY current_state) as probability
            FROM markov_transitions
            WHERE current_state = ?
            ORDER BY probability DESC
            LIMIT ?
        """, (current_state, limit))
        
    def get_state_stats(self, state: str) -> Dict:
        """
        Get statistics for a given state.
        
        Args:
            state: The state to get statistics for
            
        Returns:
            Dictionary containing:
                - total_transitions: Total number of transitions from this state
                - unique_next_states: Number of unique next states
                - most_common_next: Most common next state
                - entropy: Measure of uncertainty in transitions
        """
        result = self.db.execute_query("""
            WITH state_stats AS (
                SELECT 
                    COUNT(*) as total_transitions,
                    COUNT(DISTINCT next_state) as unique_next_states,
                    next_state as most_common_next,
                    SUM(count) as next_state_count
                FROM markov_transitions
                WHERE current_state = ?
                GROUP BY next_state
                ORDER BY next_state_count DESC
                LIMIT 1
            )
            SELECT 
                SUM(total_transitions) as total_transitions,
                MAX(unique_next_states) as unique_next_states,
                most_common_next,
                -SUM(
                    (count * 1.0 / SUM(count) OVER ()) * 
                    LOG(2, count * 1.0 / SUM(count) OVER ())
                ) as entropy
            FROM markov_transitions
            LEFT JOIN state_stats ON 1=1
            WHERE current_state = ?
            GROUP BY most_common_next
        """, (state, state))
        
        return result[0] if result else {
            'total_transitions': 0,
            'unique_next_states': 0,
            'most_common_next': None,
            'entropy': 0.0
        }
        
    def bulk_update_transitions(self, transitions: List[Tuple[str, str, int]]) -> None:
        """
        Update multiple transitions at once.
        
        Args:
            transitions: List of (current_state, next_state, count) tuples
        """
        self.db.execute_many("""
            INSERT INTO markov_transitions (current_state, next_state, count)
            VALUES (?, ?, ?)
            ON CONFLICT(current_state, next_state) DO UPDATE SET
                count = count + ?,
                updated_at = CURRENT_TIMESTAMP
        """, [(c, n, cnt, cnt) for c, n, cnt in transitions])
        
    def get_chain_stats(self) -> Dict:
        """
        Get overall statistics about the Markov chain.
        
        Returns:
            Dictionary containing:
                - total_states: Total number of unique states
                - total_transitions: Total number of transitions
                - average_entropy: Average entropy across all states
                - most_uncertain_state: State with highest entropy
                - most_certain_state: State with lowest entropy
        """
        result = self.db.execute_query("""
            WITH state_entropy AS (
                SELECT 
                    current_state,
                    -SUM(
                        (count * 1.0 / SUM(count) OVER (PARTITION BY current_state)) * 
                        LOG(2, count * 1.0 / SUM(count) OVER (PARTITION BY current_state))
                    ) as entropy
                FROM markov_transitions
                GROUP BY current_state
            )
            SELECT 
                COUNT(DISTINCT current_state) as total_states,
                SUM(count) as total_transitions,
                AVG(entropy) as average_entropy,
                (SELECT current_state FROM state_entropy ORDER BY entropy DESC LIMIT 1) as most_uncertain_state,
                (SELECT current_state FROM state_entropy ORDER BY entropy ASC LIMIT 1) as most_certain_state
            FROM markov_transitions
            LEFT JOIN state_entropy ON markov_transitions.current_state = state_entropy.current_state
        """)
        
        return result[0] if result else {
            'total_states': 0,
            'total_transitions': 0,
            'average_entropy': 0.0,
            'most_uncertain_state': None,
            'most_certain_state': None
        }
        
    def cleanup_old_transitions(self, days: int = 30) -> int:
        """
        Remove transitions that haven't been updated in the specified number of days.
        
        Args:
            days: Number of days after which to remove transitions
            
        Returns:
            Number of transitions removed
        """
        result = self.db.execute_query("""
            DELETE FROM markov_transitions
            WHERE updated_at < datetime('now', ?)
            SELECT changes()
        """, (f"-{days} days",))
        
        return result[0]['changes()'] if result else 0 