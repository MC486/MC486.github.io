# ai/mcts_model.py

import math
import random
from typing import Optional, Dict, List
import logging


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
        self.min_length = 3  # Minimum word length requirement

    def run(self, shared_letters: List[str], private_letters: List[str], word_length: int) -> Optional[str]:
        """
        Runs the MCTS algorithm to find the best word guess.
        """
        if word_length < self.min_length:
            word_length = self.min_length
            
        root = MCTSNode("") # Start with an empty word.
        logging.info(f"MCTS starting with shared letters: {shared_letters}, private letters: {private_letters}")

        for sim in range(self.simulations):
            node = self.select(root) # Select a node to expand.
            self.expand(node, shared_letters + private_letters) # Expand the selected node.
            leaf = random.choice(node.children) if node.children else node # Choose a leaf node for simulation.
            reward = self.simulate(leaf.state, shared_letters, private_letters, word_length) # Simulate a word completion and get reward.
            self.backpropagate(leaf, reward) # Update node statistics.
            
            if sim % 20 == 0:  # Log progress every 20 simulations
                logging.debug(f"Simulation {sim + 1}/{self.simulations}, Current best: {self._get_best_word(root)}")

        # Pick best child (most visits), which is the most explored and hopefully successful path.
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        if best_child and len(best_child.state) >= self.min_length:
            best_word = best_child.state
            logging.info(f"MCTS found best word: {best_word} (visits: {best_child.visits}, wins: {best_child.wins})")
            return best_word
        logging.warning("MCTS failed to find a valid word")
        return None

    def _get_best_word(self, root: MCTSNode) -> Optional[str]:
        """Helper method to get the current best word from the tree."""
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
            # Only expand with letters that haven't been used yet
            used_letters = set(node.state)
            available_letters = [l for l in letters if l not in used_letters]
            if available_letters:
                node.expand(available_letters)
                logging.debug(f"Expanded node '{node.state}' with letters: {available_letters}")

    def simulate(self, partial_word: str, shared: List[str], private: List[str], length: int) -> float:
        """
        Simulates a word completion and returns a reward based on word quality.
        Considers word length, letter distribution, and word validity.
        """
        if len(partial_word) < self.min_length:
            return 0.0
            
        # Track used letters to avoid duplicates
        used_letters = set(partial_word)
        remaining_letters = [l for l in shared + private if l not in used_letters]
        
        if not remaining_letters:
            return 0.0
            
        # Try to complete the word with remaining letters
        current_word = partial_word
        while len(current_word) < length and remaining_letters:
            # Try to find a valid next letter
            valid_next = []
            for letter in remaining_letters:
                next_word = current_word + letter
                # Only consider letters that lead to valid words
                if next_word in self.valid_words:
                    valid_next.append(letter)
                    
            if not valid_next:
                # Check if current word is valid
                if current_word in self.valid_words and len(current_word) >= self.min_length:
                    # Calculate reward based on word quality
                    reward = self._calculate_word_reward(current_word, shared, private)
                    logging.debug(f"Simulation found valid word: {current_word} (reward: {reward:.3f})")
                    return reward
                break
                
            # Choose next letter based on quality of resulting words
            best_letter = None
            best_reward = -1
            for letter in valid_next:
                next_word = current_word + letter
                if next_word in self.valid_words:
                    reward = self._calculate_word_reward(next_word, shared, private)
                    if reward > best_reward:
                        best_reward = reward
                        best_letter = letter
                        
            if best_letter is None:
                best_letter = random.choice(valid_next)
                
            current_word += best_letter
            remaining_letters.remove(best_letter)
            
        # Check if we have a valid word
        if current_word in self.valid_words and len(current_word) >= self.min_length:
            reward = self._calculate_word_reward(current_word, shared, private)
            logging.debug(f"Simulation found valid word: {current_word} (reward: {reward:.3f})")
            return reward
            
        return 0.0
        
    def _calculate_word_reward(self, word: str, shared: List[str], private: List[str]) -> float:
        """
        Calculate reward for a word based on its quality.
        Considers word length, letter distribution, and word rarity.
        """
        # Base reward for valid word
        reward = 0.5
        
        # Bonus for optimal length (4-6 letters)
        if 4 <= len(word) <= 6:
            reward += 0.2
            
        # Bonus for using shared letters
        shared_used = sum(1 for letter in word if letter in shared)
        reward += 0.1 * (shared_used / len(word))
        
        # Bonus for using private letters
        private_used = sum(1 for letter in word if letter in private)
        reward += 0.2 * (private_used / len(word))
        
        # Cap reward at 1.0
        return min(reward, 1.0)

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagates the simulation reward up the tree.
        """
        while node:
            node.visits += 1 # Increment visit count.
            node.wins += reward # Add reward to wins.
            node = node.parent # Move to parent node.