# ai/mcts_model.py

import math
import random
from typing import Optional, Dict, List


class MCTSNode:
    """
    A node in the MCTS tree. Each node represents a partial word sequence.
    """
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None):
        """
        Initializes a node with a state (partial word), parent, children, visits, and wins.
        """
        self.state = state  # The current partial word represented by this node.
        self.parent = parent # The parent node in the tree.
        self.children: List[MCTSNode] = [] # List of child nodes.
        self.visits = 0 # Number of times this node has been visited.
        self.wins = 0 # Number of successful simulations from this node.

    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf (has no children).
        """
        return len(self.children) == 0

    def uct_score(self, exploration_param: float = 1.41) -> float:
        """
        Calculates the UCT (Upper Confidence Bound applied to Trees) score for node selection.
        This balances exploration and exploitation.
        """
        if self.visits == 0:
            return float('inf') # Prioritize unvisited nodes.
        return (self.wins / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, next_letters: List[str]):
        """
        Expands the node by creating child nodes for each possible next letter.
        """
        for letter in next_letters:
            new_state = self.state + letter # Create new partial word.
            if all(child.state != new_state for child in self.children): # Avoid duplicate states.
                self.children.append(MCTSNode(new_state, parent=self))


class MCTS:
    """
    Monte Carlo Tree Search implementation for word guessing.
    """
    def __init__(self, valid_words: set, max_depth: int = 5, simulations: int = 100):
        """
        Initializes the MCTS with valid words, maximum search depth, and number of simulations.
        """
        self.valid_words = valid_words # Set of valid words to check against.
        self.max_depth = max_depth # Maximum depth of the search tree.
        self.simulations = simulations # Number of simulations to perform.

    def run(self, shared_letters: List[str], private_letters: List[str], word_length: int) -> Optional[str]:
        """
        Runs the MCTS algorithm to find the best word guess.
        """
        root = MCTSNode("") # Start with an empty word.

        for _ in range(self.simulations):
            node = self.select(root) # Select a node to expand.
            self.expand(node, shared_letters + private_letters) # Expand the selected node.
            leaf = random.choice(node.children) if node.children else node # Choose a leaf node for simulation.
            reward = self.simulate(leaf.state, shared_letters, private_letters, word_length) # Simulate a word completion and get reward.
            self.backpropagate(leaf, reward) # Update node statistics.

        # Pick best child (most visits), which is the most explored and hopefully successful path.
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        return best_child.state if best_child else None

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a node using the UCT score.
        """
        while not node.is_leaf() and node.children:
            node = max(node.children, key=lambda c: c.uct_score())
        return node

    def expand(self, node: MCTSNode, letters: List[str]):
        """
        Expands the node if it's not at maximum depth.
        """
        if len(node.state) < self.max_depth:
            node.expand(letters)

    def simulate(self, partial_word: str, shared: List[str], private: List[str], length: int) -> float:
        """
        Simulates a word completion and returns a reward (1 if valid, 0 otherwise).
        """
        remaining_letters = shared + private # Combine available letters.
        random.shuffle(remaining_letters) # Shuffle to add randomness.
        fill = ''.join(random.choices(remaining_letters, k=max(0, length - len(partial_word)))) # Fill in remaining letters.
        full_word = partial_word + fill # Complete the word.
        return 1.0 if full_word in self.valid_words else 0.0 # Return 1 if the word is valid, 0 otherwise.

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagates the simulation reward up the tree.
        """
        while node:
            node.visits += 1 # Increment visit count.
            node.wins += reward # Add reward to wins.
            node = node.parent # Move to parent node.