import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.mcts import MCTS, MCTSNode

class TestMCTS(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.valid_words = {'STAR', 'START', 'STARE', 'RATE', 'TEAR'}
        self.mcts = MCTS(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer,
            valid_words=self.valid_words,
            max_depth=4,
            simulations=20
        )

    def test_mcts_returns_valid_output(self):
        """Test basic MCTS output (maintaining original test)"""
        shared_letters = ['A', 'T', 'E', 'R']
        private_letters = ['S', 'H', 'O', 'U']
        result = self.mcts.run(shared_letters, private_letters, word_length=5)
        self.assertTrue(isinstance(result, str))
        self.assertLessEqual(len(result), 5)

    def test_node_initialization(self):
        """Test MCTSNode initialization and properties"""
        node = MCTSNode(state="STA")
        self.assertEqual(node.state, "STA")
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.wins, 0)
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
        self.assertEqual(child.uct_score(), float('inf'))
        
        # Test visited node
        child.visits = 10
        child.wins = 5
        parent.visits = 20
        score = child.uct_score()
        self.assertTrue(isinstance(score, float))
        self.assertTrue(0 <= score <= 2)  # Reasonable UCT score range

    def test_simulation_results(self):
        """Test simulation outcomes"""
        shared_letters = ['S', 'T', 'A', 'R']
        private_letters = ['E']
        
        # Configure word analyzer mock
        self.word_analyzer.get_word_score.return_value = 1.0
        
        result = self.mcts.run(shared_letters, private_letters, word_length=5)
        self.assertTrue(result in self.valid_words)

    def test_event_emission(self):
        """Test event system integration"""
        shared_letters = ['S', 'T', 'A', 'R']
        private_letters = ['E']
        
        self.mcts.run(shared_letters, private_letters, word_length=4)
        
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
        
        self.mcts.run(shared_letters, private_letters, word_length=4)
        stats = self.mcts.get_stats()
        
        self.assertIn('total_simulations', stats)
        self.assertIn('total_nodes', stats)
        self.assertIn('average_depth', stats)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty letter lists
        result = self.mcts.run([], [], word_length=4)
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
            child.visits = 5
            child.wins = 2
        
        selected = self.mcts.select(root)
        self.assertIsNotNone(selected)
        self.assertTrue(isinstance(selected, MCTSNode))

if __name__ == '__main__':
    unittest.main()