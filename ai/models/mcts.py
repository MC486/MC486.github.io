# ai/mcts_model.py

import math
import random
from typing import Optional, Dict, List, Set
import logging
from math import log
from database.repositories.mcts_repository import MCTSRepository


class MCTSNode:
    """
    A node in the MCTS tree. Each node represents a partial word sequence.
    """
    def __init__(self, state: str = "", available_letters: Set[str] = None, parent: Optional['MCTSNode'] = None):
        """
        Initializes a node with a state (partial word), parent, children, visits, and wins.
        """
        self.state = state  # The current partial word represented by this node.
        self.available_letters = available_letters or set()
        self.parent = parent # The parent node in the tree.
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visit_count = 0
        self.win_count = 0

    def get_ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for node selection."""
        if self.visit_count == 0:
            return float('inf')
        if not self.parent:
            return float('-inf')
            
        exploitation = self.win_count / self.visit_count
        exploration = exploration_constant * (log(self.parent.visit_count) / self.visit_count) ** 0.5
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf (has no children).
        """
        return len(self.children) == 0

    def expand(self, letters: List[str]) -> None:
        """Expand node with new children."""
        for letter in letters:
            new_state = self.state + letter
            self.children[new_state] = MCTSNode(
                state=new_state,
                available_letters=self.available_letters,
                parent=self
            )


class MCTS:
    """
    Monte Carlo Tree Search implementation for word guessing.
    """
    def __init__(self, valid_words: Set[str], max_depth: int = 5, 
                 num_simulations: int = 1000, db_manager = None):
        """
        Initializes the MCTS with valid words, maximum search depth, and number of simulations.
        """
        self.valid_words = valid_words # Set of valid words to check against.
        self.max_depth = max_depth # Maximum depth of the search tree.
        self.num_simulations = num_simulations # Number of simulations to perform.
        self.min_length = 3  # Minimum word length requirement
        self.repository = MCTSRepository(db_manager) if db_manager else None

    def run(self, shared_letters: List[str], private_letters: List[str]) -> Optional[str]:
        """Run MCTS to find the best word."""
        logging.info(f"MCTS starting with shared letters: {shared_letters}, private letters: {private_letters}")
        
        # Initialize root node
        root = MCTSNode(state="", available_letters=set(shared_letters + private_letters))
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(root)
            child = self._expand(node)
            if child:
                reward = self._simulate(child)
                self._backpropagate(child, reward)
            logging.debug(f"Simulation {_ + 1}/{self.num_simulations}, Current best: {self._get_best_word(root)}")
            
        # Get best word
        best_word = self._get_best_word(root)
        if best_word and best_word in self.valid_words and len(best_word) >= self.min_length:
            return best_word
        return None

    def _get_best_word(self, root: MCTSNode) -> Optional[str]:
        """Get the best word from the root node."""
        if not root.children:
            return None
            
        # Filter out nodes with no visits
        valid_children = {k: v for k, v in root.children.items() if v.visit_count > 0}
        if not valid_children:
            return None
            
        best_child = max(valid_children.items(), key=lambda x: x[1].win_count / x[1].visit_count)
        return best_child[0]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node for expansion using UCB1."""
        while not node.is_leaf() and len(node.state) < self.max_depth:
            # If any child has not been visited, select it
            unvisited = [child for child in node.children.values() if child.visit_count == 0]
            if unvisited:
                return random.choice(unvisited)
                
            # Otherwise use UCB1 to select
            node = max(node.children.values(), key=lambda n: n.get_ucb1())
        return node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand a node with new children if possible."""
        if len(node.state) >= self.max_depth:
            return None
            
        # Only expand with letters that haven't been used yet
        used_letters = set(node.state)
        available_letters = [l for l in node.available_letters if l not in used_letters]
        
        if available_letters:
            # Create children for each available letter
            for letter in available_letters:
                new_state = node.state + letter
                if new_state not in node.children:  # Avoid duplicate states
                    node.children[new_state] = MCTSNode(
                        state=new_state,
                        available_letters=node.available_letters,
                        parent=node
                    )
            # Return a random child for simulation
            return random.choice(list(node.children.values()))
        return None

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a word completion and return a reward."""
        # If we've reached a valid word, return its score
        if node.state in self.valid_words and len(node.state) >= self.min_length:
            return len(node.state) * 2  # Basic scoring: 2 points per letter
            
        # If we've reached max depth or have no more letters, return 0
        if len(node.state) >= self.max_depth:
            return 0
            
        # Track used letters to avoid duplicates
        used_letters = set(node.state)
        remaining_letters = [l for l in node.available_letters if l not in used_letters]
        
        if not remaining_letters:
            return 0
            
        # Try random moves
        for _ in range(3):  # Try up to 3 random moves
            letter = random.choice(remaining_letters)
            new_state = node.state + letter
            
            # If this creates a valid word, return its score
            if new_state in self.valid_words and len(new_state) >= self.min_length:
                return len(new_state) * 2
                
        # No valid word found
        return 0

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagates the simulation reward up the tree.
        """
        while node:
            node.visit_count += 1 # Increment visit count.
            node.win_count += reward # Add reward to wins.
            node = node.parent # Move to parent node.

    def get_learning_stats(self) -> Dict:
        """Get statistics about the model's learning."""
        if self.repository:
            return self.repository.get_learning_stats()
        return {
            'total_states': 0,
            'total_actions': 0,
            'average_reward': 0.0,
            'most_visited_state': None
        }
        
    def cleanup(self, days: int = 30) -> int:
        """Clean up old entries."""
        if self.repository:
            return self.repository.cleanup_old_entries(days)
        return 0