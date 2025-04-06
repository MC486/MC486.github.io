import unittest
from unittest.mock import Mock, patch
import pytest
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.q_learning import QLearningAgent

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.agent = QLearningAgent(self.event_manager, self.word_analyzer)
    
    def test_initialization(self):
        """Test proper initialization of the agent"""
        self.assertIsNone(self.agent.current_state)
        self.assertIsNone(self.agent.last_action)
        
        # Verify event subscriptions
        self.event_manager.subscribe.assert_any_call(
            EventType.WORD_SUBMITTED, self.agent._handle_word_submission
        )
        self.event_manager.subscribe.assert_any_call(
            EventType.GAME_START, self.agent._handle_game_start
        )
        self.event_manager.subscribe.assert_any_call(
            EventType.TURN_START, self.agent._handle_turn_start
        )
    
    def test_state_representation(self):
        """Test state representation generation"""
        letters = {'a', 'b', 'c'}
        state = self.agent._get_state_representation(letters)
        self.assertEqual(state, 'abc')  # Should be sorted
    
    def test_q_value_update(self):
        """Test Q-value updates"""
        state = 'abc'
        action = 'cab'
        reward = 0.5
        
        # Initial Q-value should be 0
        self.assertEqual(self.agent.q_table[state][action], 0)
        
        # Update Q-value
        self.agent.current_state = state
        self.agent._update_q_value(state, action, reward)
        
        # Q-value should be updated
        self.assertGreater(self.agent.q_table[state][action], 0)
    
    def test_action_selection(self):
        """Test action selection with exploration and exploitation"""
        available_letters = {'c', 'a', 't'}
        valid_words = {'cat', 'act'}
        
        # Test exploitation (force it by setting exploration_rate to 0)
        self.agent.exploration_rate = 0
        action = self.agent.select_action(available_letters, valid_words)
        self.assertIn(action, valid_words)
        
        # Test exploration (force it by setting exploration_rate to 1)
        self.agent.exploration_rate = 1
        action = self.agent.select_action(available_letters, valid_words)
        self.assertIn(action, valid_words)
    
    def test_word_submission_handling(self):
        """Test Q-value updates on word submission"""
        # Setup initial state
        self.agent.current_state = 'abc'
        self.agent.last_action = 'cab'
        
        # Submit a word with score
        event = Mock(spec=GameEvent)
        event.data = {"word": "cab", "score": 10}
        self.agent._handle_word_submission(event)
        
        # Q-value should be updated
        self.assertGreater(
            self.agent.q_table[self.agent.current_state][self.agent.last_action],
            0
        )

if __name__ == '__main__':
    unittest.main()