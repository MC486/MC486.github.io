from typing import Optional, List, Dict, Any
from .base_repository import BaseRepository
from ..manager import DatabaseManager
import logging

class GameRepository(BaseRepository):
    """Repository for managing game data."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the game repository."""
        super().__init__(db_manager)
        self.table_name = 'games'
        
        # Create games table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                max_attempts INTEGER NOT NULL,
                status TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create game_moves table if it doesn't exist
        self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS game_moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                word TEXT NOT NULL,
                is_valid INTEGER NOT NULL,
                feedback TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        """)
        
    def create_game(self, player_name: str, difficulty: str, max_attempts: int) -> int:
        """
        Create a new game.
        
        Args:
            player_name: Name of the player
            difficulty: Game difficulty level
            max_attempts: Maximum number of attempts allowed
            
        Returns:
            ID of the created game
        """
        return self.db.execute_query("""
            INSERT INTO games (player_name, difficulty, max_attempts, status)
            VALUES (?, ?, ?, 'in_progress')
            RETURNING id
        """, (player_name, difficulty, max_attempts))[0]['id']
        
    def update_game_status(self, game_id: int, status: str, score: Optional[int] = None) -> None:
        """
        Update the status of a game.
        
        Args:
            game_id: ID of the game to update
            status: New status
            score: Optional new score
        """
        query = """
            UPDATE games 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
        """
        params = [status]
        
        if score is not None:
            query += ", score = ?"
            params.append(score)
            
        if status in ['won', 'lost']:
            query += ", end_time = CURRENT_TIMESTAMP"
            
        query += " WHERE id = ?"
        params.append(game_id)
        
        self.db.execute_query(query, tuple(params))
        
    def record_move(self, game_id: int, word: str, is_valid: bool, feedback: Optional[str] = None) -> None:
        """
        Record a move in the game.
        
        Args:
            game_id: ID of the game
            word: Word attempted
            is_valid: Whether the word is valid
            feedback: Optional feedback about the move
        """
        self.db.execute_query("""
            INSERT INTO game_moves (game_id, word, is_valid, feedback)
            VALUES (?, ?, ?, ?)
        """, (game_id, word, int(is_valid), feedback))
        
    def get_game_stats(self, game_id: int) -> Dict:
        """
        Get statistics for a game.
        
        Args:
            game_id: ID of the game
            
        Returns:
            Dictionary containing game statistics
        """
        result = self.db.execute_query("""
            SELECT 
                g.*,
                COUNT(gm.id) as total_moves,
                SUM(CASE WHEN gm.is_valid = 1 THEN 1 ELSE 0 END) as valid_moves,
                SUM(CASE WHEN gm.is_valid = 0 THEN 1 ELSE 0 END) as invalid_moves,
                COALESCE(
                    CAST(
                        (julianday(COALESCE(g.end_time, CURRENT_TIMESTAMP)) - julianday(g.start_time)) * 24 * 60 
                        AS INTEGER
                    ),
                    0
                ) as duration_minutes
            FROM games g
            LEFT JOIN game_moves gm ON g.id = gm.game_id
            WHERE g.id = ?
            GROUP BY g.id
        """, (game_id,))
        
        return result[0] if result else {}
        
    def get_entry_count(self) -> int:
        """
        Get the total number of entries in the games table.
        
        Returns:
            The number of entries
        """
        query = "SELECT COUNT(*) FROM games"
        return self.db.get_scalar(query) or 0
        
    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        """
        Get game statistics for a player.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dictionary containing player statistics
        """
        result = self.db.execute_query("""
            SELECT 
                COUNT(*) as total_games,
                SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as games_won,
                SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as games_lost,
                AVG(score) as average_score,
                MAX(score) as high_score,
                AVG(
                    COALESCE(
                        CAST(
                            (julianday(COALESCE(end_time, CURRENT_TIMESTAMP)) - julianday(start_time)) * 24 * 60 
                            AS INTEGER
                        ),
                        0
                    )
                ) as avg_duration_minutes
            FROM games
            WHERE player_name = ?
        """, (player_name,))
        
        return result[0] if result else {
            'total_games': 0,
            'games_won': 0,
            'games_lost': 0,
            'average_score': 0,
            'high_score': 0,
            'avg_duration_minutes': 0
        }
        
    def get_difficulty_stats(self, difficulty: str) -> Dict[str, Any]:
        """
        Get statistics for a specific difficulty level.
        
        Args:
            difficulty: The difficulty level
            
        Returns:
            Dictionary containing difficulty statistics
        """
        stats = {
            'total_games': self.db.get_scalar("""
                SELECT COUNT(*) FROM games
                WHERE difficulty = ?
            """, (difficulty,)),
            'average_score': self.db.get_scalar("""
                SELECT AVG(score) FROM games
                WHERE difficulty = ? AND status = 'completed'
            """, (difficulty,)),
            'completion_rate': self.db.get_scalar("""
                SELECT CAST(COUNT(CASE WHEN status = 'completed' THEN 1 END) AS FLOAT) / COUNT(*)
                FROM games
                WHERE difficulty = ?
            """, (difficulty,))
        }
        return stats
        
    def get_recent_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent games.
        
        Args:
            limit: Maximum number of games to return
            
        Returns:
            List of recent game records
        """
        query = """
            SELECT g.*, COUNT(gm.id) as move_count
            FROM games g
            LEFT JOIN game_moves gm ON g.id = gm.game_id
            GROUP BY g.id
            ORDER BY g.start_time DESC
            LIMIT ?
        """
        return self.db.execute_query(query, (limit,))
        
    def get_game_summary(self, game_id: int) -> Dict[str, Any]:
        """
        Get a summary of a specific game.
        
        Args:
            game_id: The game ID
            
        Returns:
            Dictionary containing game summary
        """
        game = self.get_by_id(game_id)
        if not game:
            return None
            
        moves = self.get_game_moves(game_id)
        summary = {
            'game_info': game,
            'total_moves': len(moves),
            'valid_moves': sum(1 for m in moves if m['is_valid']),
            'invalid_moves': sum(1 for m in moves if not m['is_valid']),
            'moves': moves
        }
        return summary
        
    def cleanup_old_entries(self, days: int = 30) -> None:
        """
        Clean up old game entries and their associated moves.
        
        Args:
            days: Number of days to keep entries for (default: 30)
        """
        # Delete old game moves first (due to foreign key constraints)
        self.db.execute_query("""
            DELETE FROM game_moves
            WHERE game_id IN (
                SELECT id FROM games
                WHERE start_time < datetime('now', ?)
            )
        """, (f'-{days} days',))
        
        # Then delete old games
        self.db.execute_query("""
            DELETE FROM games
            WHERE start_time < datetime('now', ?)
        """, (f'-{days} days',))
        
    def get_game_moves(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Get all moves for a specific game.
        
        Args:
            game_id: The game ID
            
        Returns:
            List of move records
        """
        query = """
            SELECT * FROM game_moves
            WHERE game_id = ?
            ORDER BY timestamp ASC
        """
        return self.db.execute_query(query, (game_id,))
        
    def update_game_end_time(self, game_id: int) -> None:
        """
        Update the end time of a game.
        
        Args:
            game_id: ID of the game
        """
        self.db.execute_query("""
            UPDATE games
            SET end_time = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (game_id,))
        
    def get_size_bytes(self) -> int:
        """
        Get the size of the games and game_moves tables in bytes.
        
        Returns:
            Size in bytes
        """
        try:
            result = self.db.execute_query("""
                SELECT SUM(pgsize) as total_size
                FROM dbstat
                WHERE name IN ('games', 'game_moves')
            """)
            return result[0]['total_size'] if result else 0
        except Exception as e:
            logging.error(f"Error getting table size: {e}")
            return 0 