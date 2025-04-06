# ai/mcts_model.py

import math
import random
from typing import Optional, Dict, List, Set
import logging
from ..repositories.mcts_repository import MCTSRepository


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
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visit_count = 0
        self.win_count = 0

    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf (has no children).
        """
        return len(self.children) == 0

    def get_uct_score(self, exploration_weight: float = 1.41) -> float:
        """
        Calculates the UCT (Upper Confidence Bound applied to Trees) score for node selection.
        This balances exploration and exploitation.
        """
        if self.visit_count == 0:
            return float('inf') # Prioritize unvisited nodes.
        exploitation = self.win_count / self.visit_count
        exploration = exploration_weight * (2 * math.log(self.parent.visit_count) / self.visit_count) ** 0.5
        return exploitation + exploration

    def expand(self, next_letters: List[str]):
        """
        Expands the node by creating child nodes for each possible next letter.
        """
        for letter in next_letters:
            new_state = self.state + letter # Create new partial word.
            if all(child.state != new_state for child in self.children): # Avoid duplicate states.
                self.children[letter] = MCTSNode(new_state, parent=self)


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

    def run(self, shared_letters: str, private_letters: str) -> str:
        """
        Runs the MCTS algorithm to find the best word guess.
        """
        root = MCTSNode(shared_letters)
        logging.info(f"MCTS starting with shared letters: {shared_letters}, private letters: {private_letters}")

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not node.is_leaf() and len(node.state) < self.max_depth:
                node = self._expand(node, private_letters)
            reward = self._simulate(node, private_letters)
            self._backpropagate(node, reward)
            
            if _ % 20 == 0:  # Log progress every 20 simulations
                logging.debug(f"Simulation {_ + 1}/{self.num_simulations}, Current best: {self._get_best_word(root)}")

        # Get best action from repository if available
        if self.repository:
            best_action = self.repository.get_best_action(shared_letters)
            if best_action:
                logging.info(f"MCTS found best word: {best_action}")
                return best_action
                
        # Fallback to selecting best child
        best_child = max(root.children.items(), 
                          key=lambda x: x[1].win_count / x[1].visit_count)
        if best_child[1].state not in self.valid_words or len(best_child[1].state) < self.min_length:
            logging.warning("MCTS failed to find a valid word")
            return None
        best_word = best_child[0]
        logging.info(f"MCTS found best word: {best_word} (visits: {best_child[1].visit_count}, wins: {best_child[1].win_count})")
        return best_word

    def _get_best_word(self, root: MCTSNode) -> Optional[str]:
        """Helper method to get the current best word from the tree."""
        best_child = max(root.children.items(), key=lambda x: x[1].win_count / x[1].visit_count)
        return best_child[0] if best_child[1].state in self.valid_words and len(best_child[1].state) >= self.min_length else None

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a node using the UCT score.
        """
        while not node.is_leaf():
            node = max(node.children.values(), key=lambda x: x.get_uct_score())
        return node

    def _expand(self, node: MCTSNode, private_letters: str) -> MCTSNode:
        """
        Expands the node if it's not at maximum depth.
        """
        if len(node.state) < self.max_depth:
            # Only expand with letters that haven't been used yet
            used_letters = set(node.state)
            available_letters = [l for l in private_letters if l not in used_letters]
            if available_letters:
                node.expand(available_letters)
                logging.debug(f"Expanded node '{node.state}' with letters: {available_letters}")
        return node

    def _simulate(self, node: MCTSNode, private_letters: str) -> float:
        """
        Simulates a word completion and returns a reward based on word quality.
        Considers word length, letter distribution, and word validity.
        """
        if len(node.state) < self.min_length:
            return 0.0
            
        # Track used letters to avoid duplicates
        used_letters = set(node.state)
        remaining_letters = [l for l in private_letters if l not in used_letters]
        
        if not remaining_letters:
            return 0.0
            
        # Try to complete the word with remaining letters
        current_word = node.state
        while len(current_word) < self.max_depth and remaining_letters:
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
                    reward = self._calculate_word_reward(current_word)
                    logging.debug(f"Simulation found valid word: {current_word} (reward: {reward:.3f})")
                    return reward
                break
                
            # Choose next letter based on quality of resulting words
            best_letter = None
            best_reward = -1
            for letter in valid_next:
                next_word = current_word + letter
                if next_word in self.valid_words:
                    reward = self._calculate_word_reward(next_word)
                    if reward > best_reward:
                        best_reward = reward
                        best_letter = letter
                        
            if best_letter is None:
                best_letter = random.choice(valid_next)
                
            current_word += best_letter
            remaining_letters.remove(best_letter)
            
        # Check if we have a valid word
        if current_word in self.valid_words and len(current_word) >= self.min_length:
            reward = self._calculate_word_reward(current_word)
            logging.debug(f"Simulation found valid word: {current_word} (reward: {reward:.3f})")
            return reward
            
        return 0.0
        
    def _calculate_word_reward(self, word: str) -> float:
        """
        Calculate reward for a word based on its quality.
        Considers word length, letter distribution, and word rarity.
        """
        if word not in self.valid_words:
            return 0.0
            
        # Base reward for valid word
        reward = 0.5
        
        # Bonus for optimal length (4-6 letters)
        if 4 <= len(word) <= 6:
            reward += 0.2
            
        # Bonus for using shared letters
        shared_used = sum(1 for letter in word if letter in shared_letters)
        reward += 0.1 * (shared_used / len(word))
        
        # Bonus for using private letters
        private_used = sum(1 for letter in word if letter in private_letters)
        reward += 0.2 * (private_used / len(word))
        
        # Cap reward at 1.0
        return min(reward, 1.0)

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