import unittest
from unittest.mock import Mock, patch
from ai.ai_player import AIPlayer
from core.game_events import GameEvent, EventType
from core.game_events_manager import game_events_manager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.strategy.ai_strategy import AIStrategy
from database.manager import DatabaseManager

class TestAIPlayer(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=game_events_manager)
        self.db_manager = Mock(spec=DatabaseManager)
        
        # Mock repositories
        self.word_repo = Mock()
        self.word_repo.get_word_stats.return_value = {}
        self.db_manager.get_word_usage_repository.return_value = self.word_repo
        self.db_manager.get_word_repository.return_value = Mock()
        self.db_manager.get_category_repository.return_value = Mock()
        
        # Mock strategy with correct method
        self.strategy = Mock()
        self.strategy.select_word.return_value = ""
        self.strategy.reset = Mock()
        
        self.ai_player = AIPlayer(self.event_manager, self.db_manager, self.strategy)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.ai_player.score, 0)
        self.assertEqual(len(self.ai_player.word_history), 0)
        self.assertIsNotNone(self.ai_player.strategy)
        
        # Verify repository initialization
        self.db_manager.get_word_usage_repository.assert_called_once()

    def test_make_move(self):
        """Test AI move generation"""
        available_letters = ['A', 'B', 'C', 'D', 'E']
        
        # Mock strategy response
        self.strategy.select_word.return_value = "BACK"
        
        word = self.ai_player.make_move(available_letters)
        
        self.assertEqual(word, "BACK")
        self.strategy.select_word.assert_called_once()
        
        # Verify repository usage
        self.word_repo.record_word_usage.assert_called_once()

    def test_handle_word_validation(self):
        """Test handling of word validation events"""
        event = GameEvent(
            type=EventType.WORD_VALIDATED,
            data={
                "word": "HELLO",
                "player": "ai",
                "is_valid": True,
                "score": 5
            }
        )
        
        self.ai_player._handle_word_validation(event)
        
        self.assertEqual(self.ai_player.score, 5)
        self.assertIn("HELLO", self.ai_player.word_history)
        
        # Verify repository updates
        self.word_repo.record_word_usage.assert_called_once()

    def test_handle_game_start(self):
        """Test game start event handling"""
        self.ai_player.score = 10
        self.ai_player.word_history = ["HELLO"]
        
        event = GameEvent(
            type=EventType.GAME_START,
            data={}
        )
        
        self.ai_player._handle_game_start(event)
        
        self.assertEqual(self.ai_player.score, 0)
        self.assertEqual(len(self.ai_player.word_history), 0)
        self.strategy.reset.assert_called_once()
        
        # Verify repository reset
        self.word_repo.cleanup_old_entries.assert_called_once()

    def test_handle_turn_start(self):
        """Test turn start event handling"""
        event = GameEvent(
            type=EventType.TURN_START,
            data={
                "player": "ai",
                "available_letters": ['A', 'B', 'C']
            }
        )
        
        with patch.object(self.ai_player, 'make_move') as mock_make_move:
            mock_make_move.return_value = "WORD"
            self.ai_player._handle_turn_start(event)
            
            mock_make_move.assert_called_once_with(['A', 'B', 'C'])
            self.event_manager.emit.assert_called_once_with(GameEvent(
                type=EventType.WORD_SUBMITTED,
                data={"word": "WORD", "player": "ai"}
            ))

    def test_invalid_moves(self):
        """Test handling of invalid moves"""
        # Empty letters
        word = self.ai_player.make_move([])
        self.assertEqual(word, '')
        
        # None letters
        word = self.ai_player.make_move(None)
        self.assertEqual(word, '')

    def test_duplicate_word_handling(self):
        """Test handling of duplicate words"""
        self.ai_player.word_history = ["HELLO"]
        
        # Mock strategy to suggest duplicate word then new word
        self.strategy.select_word.side_effect = ["HELLO", "WORLD"]
        
        word = self.ai_player.make_move(['H', 'E', 'L', 'O', 'W', 'R', 'D'])
        
        self.assertEqual(word, "WORLD")
        self.assertEqual(self.strategy.select_word.call_count, 2)
        
        # Verify repository usage
        self.word_repo.record_word_usage.assert_called_once()

    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        # Submit several words
        for word, valid in [("HELLO", True), ("WORLD", True), ("XXXX", False)]:
            event = GameEvent(
                type=EventType.WORD_VALIDATED,
                data={
                    "word": word,
                    "player": "ai",
                    "is_valid": valid,
                    "score": 5 if valid else 0
                }
            )
            self.ai_player._handle_word_validation(event)
        
        stats = self.ai_player.get_performance_stats()
        self.assertEqual(stats["total_words"], 3)
        self.assertEqual(stats["valid_words"], 2)
        self.assertEqual(stats["total_score"], 10)
        
        # Verify repository usage
        self.word_repo.get_word_stats.assert_called_once()

    def test_event_subscription(self):
        """Test proper event subscriptions"""
        expected_subscriptions = [
            EventType.GAME_START,
            EventType.TURN_START,
            EventType.WORD_VALIDATED
        ]
        
        for event_type in expected_subscriptions:
            self.event_manager.subscribe.assert_any_call(
                event_type,
                self.ai_player._get_event_handler(event_type)
            )

if __name__ == '__main__':
    unittest.main()