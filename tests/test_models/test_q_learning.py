import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.q_learning import QLearningAgent

class TestQLearning(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.q_learning = QLearningAgent(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.1
        )
        self.valid_words = {'STAR', 'RATE', 'TEAR', 'ART'}

    def test_initialization(self):
        """Test proper initialization of Q-Learning agent"""
        self.assertEqual(self.q_learning.learning_rate, 0.1)
        self.assertEqual(self.q_learning.discount_factor, 0.9)
        self.assertEqual(self.q_learning.epsilon, 0.1)
        self.assertEqual(len(self.q_learning.q_table), 0)

    def test_q_table_update(self):
        """Test Q-value updates after rewards (from existing tests)"""
        # Setup initial state and action
        state = self.q_learning._get_state_key({'S', 'T', 'A', 'R'}, 1)
        action = "STAR"
        
        # Initial Q-value
        self.q_learning.q_table[state] = {action: 0.0}
        self.q_learning.current_state = state
        self.q_learning.last_action = action
        
        # Update with reward
        self.q_learning.update(
            reward=10.0,
            next_available_letters={'T', 'A', 'R'},
            next_turn_number=2
        )
        
        # Check Q-value was updated
        self.assertGreater(self.q_learning.q_table[state][action], 0.0)

    def test_action_selection(self):
        """Test action selection with epsilon-greedy policy"""
        available_letters = {'S', 'T', 'A', 'R'}
        
        # Force exploitation (epsilon = 0)
        self.q_learning.epsilon = 0
        action = self.q_learning.select_action(
            available_letters=available_letters,
            valid_words=self.valid_words,
            turn_number=1
        )
        self.assertIn(action, self.valid_words)

    def test_q_learning_rewards_and_penalties(self):
        """Test reward handling (from existing tests)"""
        state = self.q_learning._get_state_key({'S', 'T', 'A', 'R'}, 1)
        action = "STAR"
        
        # Setup initial state
        self.q_learning.current_state = state
        self.q_learning.last_action = action
        self.q_learning.q_table[state] = {action: 1.0}
        
        # Test positive reward
        self.q_learning.update(reward=10.0, next_available_letters={'T', 'A', 'R'}, next_turn_number=2)
        positive_q = self.q_learning.q_table[state][action]
        
        # Test negative reward
        self.q_learning.update(reward=-5.0, next_available_letters={'T', 'A', 'R'}, next_turn_number=2)
        negative_q = self.q_learning.q_table[state][action]
        
        self.assertGreater(positive_q, negative_q)

    def test_q_learning_exploration_decay(self):
        """Test epsilon decay for exploration (from existing tests)"""
        initial_epsilon = self.q_learning.epsilon
        
        # Simulate multiple game starts
        for _ in range(5):
            self.q_learning._handle_game_start(GameEvent(
                type=EventType.GAME_START,
                data={}
            ))
        
        self.assertLess(self.q_learning.epsilon, initial_epsilon)

    def test_state_representation(self):
        """Test state key generation"""
        state_key = self.q_learning._get_state_key({'A', 'B', 'C'}, 1)
        self.assertTrue(isinstance(state_key, str))
        self.assertIn("ABC", state_key)
        self.assertIn("1", state_key)

    def test_valid_actions_generation(self):
        """Test valid action generation"""
        available_letters = {'S', 'T', 'A', 'R'}
        valid_actions = self.q_learning._get_valid_actions(
            available_letters,
            self.valid_words
        )
        
        self.assertTrue(all(set(word).issubset(available_letters) for word in valid_actions))
        self.assertTrue(all(word in self.valid_words for word in valid_actions))

    def test_event_handling(self):
        """Test event system integration"""
        # Test word submission handling
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "STAR",
                "score": 10,
                "next_available_letters": ['T', 'A', 'R'],
                "turn_number": 1
            }
        )
        
        self.q_learning._handle_word_submission(event)
        self.assertGreater(self.q_learning.total_reward, 0)

    def test_model_statistics(self):
        """Test statistics reporting"""
        # Generate some data
        state = self.q_learning._get_state_key({'S', 'T', 'A', 'R'}, 1)
        self.q_learning.q_table[state] = {"STAR": 1.0, "ART": 0.5}
        
        stats = self.q_learning.get_stats()
        self.assertIn("total_states", stats)
        self.assertIn("total_actions", stats)
        self.assertIn("total_reward", stats)
        self.assertIn("current_epsilon", stats)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty letters
        action = self.q_learning.select_action(set(), self.valid_words, 1)
        self.assertEqual(action, "")
        
        # Invalid turn number
        action = self.q_learning.select_action({'A', 'B'}, self.valid_words, -1)
        self.assertEqual(action, "")
        
        # Empty valid words
        action = self.q_learning.select_action({'A', 'B'}, set(), 1)
        self.assertEqual(action, "")

if __name__ == '__main__':
    unittest.main()