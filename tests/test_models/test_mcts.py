import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.mcts import MCTS, MCTSNode
from database.manager import DatabaseManager

class TestMCTS(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.db_manager = Mock(spec=DatabaseManager)
        self.valid_words = {'STAR', 'START', 'STARE', 'RATE', 'TEAR'}
        
        # Mock repository methods
        self.repository = Mock()
        self.db_manager.get_mcts_repository.return_value = self.repository
        self.repository.get_best_action.return_value = None
        self.repository.get_state_actions.return_value = {
            'A': {'avg_reward': 0.5},
            'R': {'avg_reward': 0.6},
            'T': {'avg_reward': 0.4}
        }
        self.repository.get_state_stats.return_value = {
            'A': {'avg_reward': 0.5},
            'R': {'avg_reward': 0.6},
            'T': {'avg_reward': 0.4}
        }
        self.repository.save_state.return_value = None
        
        self.mcts = MCTS(
            valid_words=self.valid_words,
            max_depth=4,
            num_simulations=20,
            db_manager=self.db_manager
        )

    def test_mcts_returns_valid_output(self):
        """Test basic MCTS output (maintaining original test)"""
        # Use letters that don't immediately form a valid word
        shared_letters = ['E', 'D', 'G']
        private_letters = ['E']
        
        # Clear the valid words set and add only one word that requires MCTS to find
        self.valid_words.clear()
        self.valid_words.add('EDGE')
        
        # Configure repository to guide MCTS
        self.repository.get_state_stats.return_value = {
            'E': {'avg_reward': 0.8},
            'D': {'avg_reward': 0.7},
            'G': {'avg_reward': 0.6}
        }
        
        result = self.mcts.run(shared_letters, private_letters)
        self.assertTrue(isinstance(result, str))
        self.assertLessEqual(len(result), 5)
        
        # Verify repository usage
        self.repository.get_best_action.assert_called_once()
        self.repository.get_state_stats.assert_called()

    def test_node_initialization(self):
        """Test MCTSNode initialization and properties"""
        node = MCTSNode(state="STA")
        self.assertEqual(node.state, "STA")
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.win_count, 0)
        self.assertIsNone(node.parent)
        self.assertEqual(len(node.children), 0)

    def test_node_expansion(self):
        """Test node expansion with available letters"""
        node = MCTSNode(state="ST")
        node.expand(['A', 'R', 'E'])
        
        self.assertTrue(len(node.children) > 0)
        child_states = [child.state for child in node.children]
        self.assertTrue("STA" in child_states)

    def test_uct_score_calculation(self):
        """Test UCT score calculation"""
        parent = MCTSNode(state="")
        child = MCTSNode(state="S", parent=parent)
        parent.children.append(child)
        
        # Test unvisited node
        self.assertEqual(child.get_uct_score(), float('inf'))
        
        # Test visited node
        child.visit_count = 10
        child.win_count = 5
        parent.visit_count = 20
        score = child.get_uct_score()
        self.assertTrue(isinstance(score, float))
        self.assertTrue(0 <= score <= 2)  # Reasonable UCT score range

    def test_simulation_results(self):
        """Test simulation outcomes"""
        shared_letters = ['S', 'T', 'A', 'R']
        private_letters = ['E']
        
        # Configure word analyzer mock
        self.word_analyzer.get_word_score.return_value = 1.0
        
        result = self.mcts.run(shared_letters, private_letters)
        self.assertTrue(result in self.valid_words)
        
        # Verify repository updates
        self.repository.save_state.assert_called()

    def test_event_emission(self):
        """Test event system integration"""
        shared_letters = ['S', 'T', 'A', 'R']
        private_letters = ['E']
        
        self.mcts.run(shared_letters, private_letters)
        
        # Verify events were emitted
        self.event_manager.emit.assert_called()
        calls = self.event_manager.emit.call_args_list
        
        # Check for analysis start event
        self.assertTrue(
            any(call.args[0].type == EventType.AI_ANALYSIS_START 
                for call in calls)
        )

    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        shared_letters = ['S', 'T', 'A', 'R']
        private_letters = ['E']
        
        self.mcts.run(shared_letters, private_letters)
        stats = self.mcts.get_stats()
        
        self.assertIn('total_simulations', stats)
        self.assertIn('total_nodes', stats)
        self.assertIn('average_depth', stats)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty letter lists
        result = self.mcts.run([], [])
        self.assertEqual(result, None)
        
        # Invalid word length
        result = self.mcts.run(['A', 'B'], ['C'], word_length=0)
        self.assertEqual(result, None)

    def test_tree_exploration(self):
        """Test tree exploration behavior"""
        root = MCTSNode(state="")
        root.expand(['S', 'T', 'A'])
        
        # Simulate some visits and wins
        for child in root.children:
            child.visit_count = 5
            child.win_count = 2
        
        selected = self.mcts._select(root)
        self.assertIsNotNone(selected)
        self.assertTrue(isinstance(selected, MCTSNode))

if __name__ == '__main__':
    unittest.main()