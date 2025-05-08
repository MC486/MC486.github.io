"""
Monte Carlo Tree Search implementation for word game.
"""
import math
import random
import time
from typing import Optional, Dict, List, Set, Any
import logging
from math import log
from database.repositories.mcts_repository import MCTSRepository
from core.game_events_manager import GameEventManager
from core.game_events import GameEvent, EventType

logger = logging.getLogger(__name__)

class MCTSNode:
    """Node class for Monte Carlo Tree Search."""
    
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None):
        """
        Initialize a node with a state and optional parent.
        
        Args:
            state: The current word state
            parent: Parent node (None for root)
        """
        self.state = state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions: List[str] = []
        self.simulation_results: List[float] = []
        
    def expand(self, available_actions: List[str]) -> None:
        """
        Expand the node with available actions.
        
        Args:
            available_actions: List of available letters to try
            
        Raises:
            ValueError: If no actions are available
        """
        try:
            if not available_actions:
                raise ValueError("No actions available for expansion")
                
            self.untried_actions = list(available_actions)
            for action in available_actions:
                child = MCTSNode(self.state + action, parent=self)
                self.children.append(child)
        except Exception as e:
            logger.error(f"Error expanding node: {str(e)}")
            raise
            
    def update(self, reward: float) -> None:
        """
        Update node statistics.
        
        Args:
            reward: Reward value from simulation
        """
        try:
            self.visit_count += 1
            self.total_reward += reward
            self.simulation_results.append(reward)
        except Exception as e:
            logger.error(f"Error updating node: {str(e)}")
            
    def get_uct_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCT score for node selection.
        
        Args:
            exploration_constant: Weight for exploration term
            
        Returns:
            float: UCT score for node selection
        """
        try:
            if self.visit_count == 0:
                return float('inf')
            if not self.parent:
                return self.total_reward / self.visit_count
                
            exploitation = self.total_reward / self.visit_count
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visit_count) / self.visit_count
            )
            return exploitation + exploration
        except Exception as e:
            logger.error(f"Error calculating UCT score: {str(e)}")
            return 0.0
            
    def best_child(self) -> Optional['MCTSNode']:
        """
        Select best child based on UCT score.
        
        Returns:
            Optional[MCTSNode]: Best child node or None if no children
        """
        try:
            if not self.children:
                return None
            return max(self.children, key=lambda c: c.get_uct_score())
        except Exception as e:
            logger.error(f"Error selecting best child: {str(e)}")
            return None
            
    def is_terminal(self) -> bool:
        """
        Check if node is terminal.
        
        Returns:
            bool: True if node is terminal
        """
        return len(self.untried_actions) == 0 and len(self.children) == 0
        
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried.
        
        Returns:
            bool: True if node is fully expanded
        """
        return len(self.untried_actions) == 0
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get node statistics.
        
        Returns:
            Dict[str, Any]: Node statistics
        """
        return {
            'visit_count': self.visit_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.visit_count if self.visit_count > 0 else 0,
            'child_count': len(self.children),
            'untried_actions': len(self.untried_actions),
            'simulation_results': self.simulation_results
        }


class MCTS:
    """Monte Carlo Tree Search implementation for word game."""
    
    def __init__(self, 
                 valid_words: Set[str], 
                 max_depth: int = 4, 
                 num_simulations: int = 20, 
                 db_manager: Any = None,
                 min_length: int = 3):
        """
        Initialize MCTS with game parameters.
        
        Args:
            valid_words: Set of valid words
            max_depth: Maximum search depth
            num_simulations: Number of simulations per move
            db_manager: Database manager for persistence
            min_length: Minimum word length
        """
        self.valid_words = valid_words
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.min_length = min_length
        self.repository = MCTSRepository(db_manager) if db_manager else None
        self.event_manager = GameEventManager()
        self.simulation_strategies = ['random', 'greedy', 'balanced']
        self.stats = {
            'total_simulations': 0,
            'total_wins': 0,
            'avg_depth': 0.0,
            'best_word': None,
            'best_score': 0.0,
            'simulation_time': 0.0,
            'node_count': 0,
            'avg_branching_factor': 0.0,
            'strategy_success': {s: 0 for s in self.simulation_strategies}
        }
        
    def run(self, shared_letters: List[str], private_letters: List[str]) -> Optional[str]:
        """
        Run MCTS to find the best word.
        
        Args:
            shared_letters: List of shared letters
            private_letters: List of private letters
            
        Returns:
            Optional[str]: Best word found or None if no valid word
        """
        if not shared_letters and not private_letters:
            return None
            
        logger.info(f"MCTS starting with shared letters: {shared_letters}, private letters: {private_letters}")
        
        start_time = time.time()
        
        # Initialize root node
        root = MCTSNode(state="", parent=None)
        root.untried_actions = shared_letters + private_letters
        
        # Run simulations
        best_word = None
        best_score = float('-inf')
        
        try:
            for i in range(self.num_simulations):
                # Selection
                node = self._select(root)
                
                # Expansion
                child = self._expand(node)
                if child:
                    # Simulation
                    reward = self._simulate(child)
                    
                    # Backpropagation
                    self._backpropagate(child, reward)
                    
                    # Update best word if needed
                    if reward > best_score:
                        best_score = reward
                        best_word = child.state
                        
                logger.debug(f"Simulation {i+1}/{self.num_simulations}, Current best: {best_word}")
                
                # Update stats
                self._update_stats(i, child, reward, best_word, best_score)
                
                # Emit event for monitoring
                self._emit_simulation_event(i, best_word, best_score)
                
        except Exception as e:
            logger.error(f"Error in MCTS simulation: {str(e)}")
            
        # Update final statistics
        self.stats['simulation_time'] = time.time() - start_time
        self.stats['node_count'] = self._get_node_count(root)
        self.stats['avg_branching_factor'] = self._calculate_avg_branching_factor(root)
        
        return best_word
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the MCTS process.
        
        Returns:
            Dict[str, Any]: MCTS statistics
        """
        return self.stats

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node for expansion using UCT.
        
        Args:
            node: Current node
            
        Returns:
            MCTSNode: Selected node
        """
        try:
            while not node.is_terminal() and len(node.state) < self.max_depth:
                # If any child has not been visited, select it
                unvisited = [c for c in node.children if c.visit_count == 0]
                if unvisited:
                    return random.choice(unvisited)
                    
                # Otherwise use UCT to select
                node = node.best_child()
                if not node:
                    break
            return node
        except Exception as e:
            logger.error(f"Error in node selection: {str(e)}")
            return node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand a node with new children if possible.
        
        Args:
            node: Node to expand
            
        Returns:
            Optional[MCTSNode]: New child node or None if expansion not possible
        """
        try:
            if len(node.state) >= self.max_depth:
                return None
                
            # Only expand with letters that haven't been used yet
            used_letters = set(node.state)
            available_letters = [l for l in node.untried_actions if l not in used_letters]
            
            if available_letters:
                # Create children for each available letter
                for letter in available_letters:
                    new_state = node.state + letter
                    if new_state not in [c.state for c in node.children]:  # Avoid duplicate states
                        new_child = MCTSNode(state=new_state, parent=node)
                        node.children.append(new_child)
                        node.untried_actions.remove(letter)
                # Return a random child for simulation
                return random.choice(node.children)
            return None
        except Exception as e:
            logger.error(f"Error expanding node: {str(e)}")
            return None

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate a word completion and return a reward.
        
        Args:
            node: Node to simulate from
            
        Returns:
            float: Simulation reward
        """
        try:
            # Try each simulation strategy
            for strategy in self.simulation_strategies:
                result = self._simulate_with_strategy(node, strategy)
                if result > 0:
                    self.stats['strategy_success'][strategy] += 1
                    return result
            return 0
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            return 0

    def _simulate_with_strategy(self, node: MCTSNode, strategy: str) -> float:
        """
        Simulate using a specific strategy.
        
        Args:
            node: Node to simulate from
            strategy: Simulation strategy to use
            
        Returns:
            float: Simulation reward
        """
        try:
            current_state = node.state
            used_letters = set(current_state)
            available_letters = [l for l in node.untried_actions if l not in used_letters]
            
            if not available_letters:
                return 0
                
            if strategy == 'random':
                return self._simulate_random(current_state, available_letters)
            elif strategy == 'greedy':
                return self._simulate_greedy(current_state, available_letters)
            else:  # balanced
                return self._simulate_balanced(current_state, available_letters)
        except Exception as e:
            logger.error(f"Error in {strategy} simulation: {str(e)}")
            return 0

    def _simulate_random(self, state: str, available_letters: List[str]) -> float:
        """Random simulation strategy."""
        for _ in range(3):
            letter = random.choice(available_letters)
            new_state = state + letter
            if new_state in self.valid_words and len(new_state) >= self.min_length:
                return len(new_state) * 2
        return 0

    def _simulate_greedy(self, state: str, available_letters: List[str]) -> float:
        """Greedy simulation strategy."""
        best_reward = 0
        for letter in available_letters:
            new_state = state + letter
            if new_state in self.valid_words and len(new_state) >= self.min_length:
                reward = len(new_state) * 2
                if reward > best_reward:
                    best_reward = reward
        return best_reward

    def _simulate_balanced(self, state: str, available_letters: List[str]) -> float:
        """Balanced simulation strategy."""
        # Try to balance exploration and exploitation
        if random.random() < 0.3:  # 30% chance to explore
            return self._simulate_random(state, available_letters)
        return self._simulate_greedy(state, available_letters)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate the simulation reward up the tree.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate
        """
        try:
            while node:
                node.update(reward)
                node = node.parent
        except Exception as e:
            logger.error(f"Error in backpropagation: {str(e)}")

    def _update_stats(self, 
                     simulation_num: int, 
                     node: Optional[MCTSNode], 
                     reward: float, 
                     best_word: Optional[str], 
                     best_score: float) -> None:
        """
        Update simulation statistics.
        
        Args:
            simulation_num: Current simulation number
            node: Current node
            reward: Current reward
            best_word: Current best word
            best_score: Current best score
        """
        self.stats['total_simulations'] += 1
        if reward > 0:
            self.stats['total_wins'] += 1
        if node:
            self.stats['avg_depth'] = (self.stats['avg_depth'] * simulation_num + len(node.state)) / (simulation_num + 1)
        if best_word:
            self.stats['best_word'] = best_word
            self.stats['best_score'] = best_score

    def _emit_simulation_event(self, simulation_num: int, best_word: Optional[str], best_score: float) -> None:
        """
        Emit simulation event.
        
        Args:
            simulation_num: Current simulation number
            best_word: Current best word
            best_score: Current best score
        """
        if self.event_manager:
            self.event_manager.emit(GameEvent(
                type=EventType.MCTS_SIMULATION,
                data={
                    'simulation': simulation_num + 1,
                    'best_word': best_word,
                    'best_score': best_score
                }
            ))

    def _get_node_count(self, root: MCTSNode) -> int:
        """
        Count total nodes in the tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            int: Total node count
        """
        count = 1
        for child in root.children:
            count += self._get_node_count(child)
        return count

    def _calculate_avg_branching_factor(self, root: MCTSNode) -> float:
        """
        Calculate average branching factor of the tree.
        
        Args:
            root: Root node of the tree
            
        Returns:
            float: Average branching factor
        """
        total_branches = 0
        total_nodes = 0
        
        def traverse(node: MCTSNode):
            nonlocal total_branches, total_nodes
            if node.children:
                total_branches += len(node.children)
                total_nodes += 1
            for child in node.children:
                traverse(child)
                
        traverse(root)
        return total_branches / total_nodes if total_nodes > 0 else 0

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model's learning.
        
        Returns:
            Dict[str, Any]: Learning statistics
        """
        if self.repository:
            return self.repository.get_learning_stats()
        return {
            'total_states': 0,
            'total_actions': 0,
            'average_reward': 0.0,
            'most_visited_state': None
        }
        
    def cleanup(self, days: int = 30) -> int:
        """
        Clean up old entries.
        
        Args:
            days: Number of days to keep
            
        Returns:
            int: Number of entries cleaned
        """
        if self.repository:
            return self.repository.cleanup_old_entries(days)
        return 0 