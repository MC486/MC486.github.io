from typing import Optional, List, Dict, Any
from ..base_repository import BaseRepository
from ..manager import DatabaseManager

class GameRepository(BaseRepository):
    """Repository for managing game sessions and statistics."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize the game repository."""
        super().__init__(db_manager, 'games')
        
    def create_game(self, player_name: str, difficulty: str, max_attempts: int) -> int:
        """
        Create a new game session.
        
        Args:
            player_name: Name of the player
            difficulty: Game difficulty level
            max_attempts: Maximum number of attempts allowed
            
        Returns:
            The ID of the created game
        """
        game_data = {
            'player_name': player_name,
            'difficulty': difficulty,
            'max_attempts': max_attempts,
            'status': 'in_progress',
            'start_time': 'CURRENT_TIMESTAMP'
        }
        return self.create(game_data)
        
    def update_game_status(self, game_id: int, status: str, score: Optional[int] = None) -> bool:
        """
        Update the status of a game.
        
        Args:
            game_id: The game ID
            status: New status ('completed', 'abandoned', etc.)
            score: Optional final score
            
        Returns:
            True if the update was successful
        """
        update_data = {
            'status': status,
            'end_time': 'CURRENT_TIMESTAMP'
        }
        if score is not None:
            update_data['score'] = score
        return self.update(game_id, update_data)
        
    def record_move(self, game_id: int, word: str, is_valid: bool, feedback: str) -> int:
        """
        Record a player's move in the game.
        
        Args:
            game_id: The game ID
            word: The word attempted
            is_valid: Whether the word was valid
            feedback: Feedback about the move
            
        Returns:
            The ID of the recorded move
        """
        move_data = {
            'game_id': game_id,
            'word': word,
            'is_valid': is_valid,
            'feedback': feedback,
            'timestamp': 'CURRENT_TIMESTAMP'
        }
        return self.db.create('game_moves', move_data)
        
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
        
    def get_player_stats(self, player_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific player.
        
        Args:
            player_name: Name of the player
            
        Returns:
            Dictionary containing player statistics
        """
        stats = {
            'total_games': self.db.get_scalar("""
                SELECT COUNT(*) FROM games
                WHERE player_name = ?
            """, (player_name,)),
            'completed_games': self.db.get_scalar("""
                SELECT COUNT(*) FROM games
                WHERE player_name = ? AND status = 'completed'
            """, (player_name,)),
            'average_score': self.db.get_scalar("""
                SELECT AVG(score) FROM games
                WHERE player_name = ? AND status = 'completed'
            """, (player_name,)),
            'highest_score': self.db.get_scalar("""
                SELECT MAX(score) FROM games
                WHERE player_name = ? AND status = 'completed'
            """, (player_name,))
        }
        return stats
        
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