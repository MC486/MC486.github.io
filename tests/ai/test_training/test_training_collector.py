import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.training.training_collector import TrainingCollector

class TestTrainingCollector(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.collector = TrainingCollector(self.event_manager)

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(len(self.collector.training_data), 0)
        self.assertIsNone(self.collector.current_game_data)
        self.assertEqual(self.collector.total_samples, 0)

    def test_game_start_collection(self):
        """Test collection of game start data"""
        event = GameEvent(
            type=EventType.GAME_START,
            data={
                "available_letters": ['A', 'B', 'C', 'D', 'E'],
                "board_size": 4,
                "time_limit": 180
            }
        )
        
        self.collector._handle_game_start(event)
        
        self.assertIsNotNone(self.collector.current_game_data)
        self.assertEqual(len(self.collector.current_game_data["moves"]), 0)
        self.assertEqual(
            self.collector.current_game_data["initial_letters"],
            ['A', 'B', 'C', 'D', 'E']
        )

    def test_move_collection(self):
        """Test collection of move data"""
        # Setup game
        self.collector._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
        ))
        
        # Collect move
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "player": "ai",
                "model_used": "markov",
                "confidence": 0.8,
                "available_letters": ['H', 'E', 'L', 'L', 'O']
            }
        )
        
        self.collector._handle_word_submission(event)
        
        move_data = self.collector.current_game_data["moves"][0]
        self.assertEqual(move_data["word"], "HELLO")
        self.assertEqual(move_data["model"], "markov")
        self.assertEqual(move_data["confidence"], 0.8)

    def test_move_validation_collection(self):
        """Test collection of move validation data"""
        # Setup game and move
        self.collector._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
        ))
        
        self.collector._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "player": "ai",
                "model_used": "markov"
            }
        ))
        
        # Validate move
        event = GameEvent(
            type=EventType.WORD_VALIDATED,
            data={
                "word": "HELLO",
                "player": "ai",
                "is_valid": True,
                "score": 5
            }
        )
        
        self.collector._handle_word_validation(event)
        
        move_data = self.collector.current_game_data["moves"][0]
        self.assertTrue(move_data["valid"])
        self.assertEqual(move_data["score"], 5)

    def test_game_end_collection(self):
        """Test collection of game end data"""
        # Setup and play game
        self.collector._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
        ))
        
        self.collector._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO", "player": "ai", "model_used": "markov"}
        ))
        
        event = GameEvent(
            type=EventType.GAME_END,
            data={
                "scores": {"ai": 5, "human": 3},
                "winner": "ai",
                "total_turns": 10
            }
        )
        
        self.collector._handle_game_end(event)
        
        self.assertEqual(len(self.collector.training_data), 1)
        self.assertIsNone(self.collector.current_game_data)
        self.assertEqual(self.collector.total_samples, 1)

    def test_export_training_data(self):
        """Test export of training data"""
        # Generate some training data
        self.collector._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
        ))
        
        self.collector._handle_word_submission(GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={"word": "HELLO", "player": "ai", "model_used": "markov"}
        ))
        
        self.collector._handle_game_end(GameEvent(
            type=EventType.GAME_END,
            data={"scores": {"ai": 5}, "winner": "ai"}
        ))
        
        data = self.collector.export_training_data()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIn("moves", data[0])
        self.assertIn("results", data[0])

    def test_model_performance_tracking(self):
        """Test tracking of model performance"""
        # Play multiple games with different models
        models = ["markov", "naive_bayes", "mcts"]
        for model in models:
            self.collector._handle_game_start(GameEvent(
                type=EventType.GAME_START,
                data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
            ))
            
            self.collector._handle_word_submission(GameEvent(
                type=EventType.WORD_SUBMITTED,
                data={"word": "HELLO", "player": "ai", "model_used": model}
            ))
            
            self.collector._handle_word_validation(GameEvent(
                type=EventType.WORD_VALIDATED,
                data={"word": "HELLO", "is_valid": True, "score": 5}
            ))
            
            self.collector._handle_game_end(GameEvent(
                type=EventType.GAME_END,
                data={"scores": {"ai": 5}, "winner": "ai"}
            ))
        
        stats = self.collector.get_model_statistics()
        for model in models:
            self.assertIn(model, stats)
            self.assertIn("success_rate", stats[model])
            self.assertIn("average_score", stats[model])

    def test_data_cleanup(self):
        """Test cleanup of training data"""
        # Generate some data
        self.collector._handle_game_start(GameEvent(
            type=EventType.GAME_START,
            data={"available_letters": ['H', 'E', 'L', 'L', 'O']}
        ))
        
        self.collector.cleanup_training_data()
        
        self.assertEqual(len(self.collector.training_data), 0)
        self.assertEqual(self.collector.total_samples, 0)

if __name__ == '__main__':
    unittest.main()