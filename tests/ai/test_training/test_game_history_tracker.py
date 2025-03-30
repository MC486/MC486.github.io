import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.training.game_history_tracker import GameHistoryTracker

class TestGameHistoryTracker(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.tracker = GameHistoryTracker(self.event_manager)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(len(self.tracker.game_history), 0)
        self.assertIsNone(self.tracker.current_game)
        self.assertEqual(self.tracker.total_games, 0)

    def test_game_start_tracking(self):
        """Test tracking of game start"""
        event = GameEvent(
            type=EventType.GAME_START,
            data={
                "player_count": 2,
                "board_size": 4,
                "time_limit": 180
            }
        )
        
        self.tracker._handle_game_start(event)
        
        self.assertIsNotNone(self.tracker.current_game)
        self.assertEqual(self.tracker.current_game["settings"]["player_count"], 2)
        self.assertEqual(self.tracker.current_game["moves"], [])

    def test_move_tracking(self):
        """Test tracking of player moves"""
        # Setup current game
        self.tracker._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"player_count": 2}
        ))
        
        # Track a move
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "player": "ai",
                "available_letters": ['H', 'E', 'L', 'L', 'O'],
                "turn_number": 1
            }
        )
        
        self.tracker._handle_word_submission(event)
        
        self.assertEqual(len(self.tracker.current_game["moves"]), 1)
        self.assertEqual(self.tracker.current_game["moves"][0]["word"], "HELLO")

    def test_game_end_tracking(self):
        """Test tracking of game end"""
        # Setup and play a game
        self.tracker._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"player_count": 2}
        ))
        
        self.tracker._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "player": "ai",
                "turn_number": 1
            }
        ))
        
        event = GameEvent(
            type=EventType.GAME_END,
            data={
                "scores": {"ai": 10, "human": 8},
                "winner": "ai"
            }
        )
        
        self.tracker._handle_game_end(event)
        
        self.assertEqual(len(self.tracker.game_history), 1)
        self.assertIsNone(self.tracker.current_game)
        self.assertEqual(self.tracker.total_games, 1)

    def test_move_validation_tracking(self):
        """Test tracking of move validation"""
        # Setup current game
        self.tracker._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"player_count": 2}
        ))
        
        # Submit a word
        self.tracker._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "player": "ai",
                "turn_number": 1
            }
        ))
        
        # Validate the word
        event = GameEvent(
            type=EventType.WORD_VALIDATED,
            data={
                "word": "HELLO",
                "player": "ai",
                "is_valid": True,
                "score": 5
            }
        )
        
        self.tracker._handle_word_validation(event)
        
        last_move = self.tracker.current_game["moves"][-1]
        self.assertTrue(last_move["valid"])
        self.assertEqual(last_move["score"], 5)

    def test_export_history(self):
        """Test history export functionality"""
        # Play a complete game
        self.tracker._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"player_count": 2}
        ))
        
        self.tracker._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO", "player": "ai", "turn_number": 1}
        ))
        
        self.tracker._handle_game_end(GameEvent(
            type=EventType.GAME_END,
            data={"scores": {"ai": 5}, "winner": "ai"}
        ))
        
        history = self.tracker.export_history()
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 1)
        self.assertIn("settings", history[0])
        self.assertIn("moves", history[0])
        self.assertIn("results", history[0])

    def test_invalid_event_handling(self):
        """Test handling of invalid events"""
        # Word submission without game start
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO", "player": "ai"}
        )
        
        self.tracker._handle_word_submission(event)
        self.assertEqual(len(self.tracker.game_history), 0)

    def test_history_statistics(self):
        """Test calculation of history statistics"""
        # Play multiple games
        for _ in range(3):
            self.tracker._handle_game_start(GameEvent(
                type=EventType.GAME_START,
                data={"player_count": 2}
            ))
            
            self.tracker._handle_word_submission(GameEvent(
                type=EventType.WORD_SUBMITTED,
                data={"word": "HELLO", "player": "ai", "turn_number": 1}
            ))
            
            self.tracker._handle_game_end(GameEvent(
                type=EventType.GAME_END,
                data={"scores": {"ai": 5}, "winner": "ai"}
            ))
        
        stats = self.tracker.get_statistics()
        self.assertEqual(stats["total_games"], 3)
        self.assertIn("average_score", stats)
        self.assertIn("win_rate", stats)

if __name__ == '__main__':
    unittest.main()