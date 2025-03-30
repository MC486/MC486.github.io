import math
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: Any, parent: Optional['Node'] = None, action: Optional[str] = None):
        """
        Initialize a node in the MCTS tree.
        
        Args:
            state: The game state at this node
            parent: Parent node (None for root)
            action: Action that led to this node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[Node] = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.get_untried_actions()
        self.is_terminal = self.is_terminal_state()
        
    def get_untried_actions(self) -> List[str]:
        """
        Get list of untried actions from current state.
        
        Returns:
            List[str]: List of available actions
        """
        try:
            return self.state.get_available_actions()
        except Exception as e:
            logger.error(f"Error getting untried actions: {str(e)}")
            return []
        
    def add_child(self, action: str, state: Any) -> 'Node':
        """
        Add a child node to the current node.
        
        Args:
            action: Action taken
            state: Resulting state
            
        Returns:
            Node: The new child node
        """
        child = Node(state=state, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children.append(child)
        return child
        
    def update(self, result: float) -> None:
        """
        Update node statistics.
        
        Args:
            result: Result of the simulation (0.0 to 1.0)
        """
        self.visits += 1
        self.wins += result
        
    def fully_expanded(self) -> bool:
        """
        Check if node is fully expanded.
        
        Returns:
            bool: True if all actions have been tried
        """
        return len(self.untried_actions) == 0
        
    def best_child(self, c_param: float = 1.414) -> Optional['Node']:
        """
        Select best child using UCT formula.
        
        Args:
            c_param: Exploration parameter
            
        Returns:
            Optional[Node]: Best child node or None if no children
        """
        if not self.children:
            return None
            
        try:
            choices = [(c.wins / c.visits) + c_param * math.sqrt((2 * math.log(self.visits) / c.visits)) 
                      for c in self.children]
            return self.children[np.argmax(choices)]
        except Exception as e:
            logger.error(f"Error selecting best child: {str(e)}")
            return None
            
    def is_terminal_state(self) -> bool:
        """
        Check if this is a terminal state.
        
        Returns:
            bool: True if state is terminal
        """
        try:
            return self.state.is_terminal()
        except Exception as e:
            logger.error(f"Error checking terminal state: {str(e)}")
            return True

class MCTS:
    def __init__(self, exploration_weight: float = 1.414):
        """
        Initialize MCTS with exploration parameter.
        
        Args:
            exploration_weight: Weight for exploration in UCT formula
        """
        self.exploration_weight = exploration_weight
        self.root = None
        
    def choose_action(self, root_state: Any, num_simulations: int = 1000) -> Optional[str]:
        """
        Choose the best action using MCTS.
        
        Args:
            root_state: The current game state
            num_simulations: Number of simulations to run
            
        Returns:
            Optional[str]: Best action or None if no valid actions
        """
        if not root_state:
            logger.error("Invalid root state")
            return None
            
        self.root = Node(state=root_state)
        
        try:
            for _ in range(num_simulations):
                node = self.root
                state = root_state.copy()
                
                # Selection
                while node.fully_expanded() and node.children and not node.is_terminal:
                    node = node.best_child(self.exploration_weight)
                    if not node:
                        break
                    state = node.state
                    
                # Expansion
                if node.untried_actions and not node.is_terminal:
                    action = random.choice(node.untried_actions)
                    state = self.simulate_action(state, action)
                    node = node.add_child(action, state)
                    
                # Simulation
                while not self.is_terminal(state):
                    action = random.choice(self.get_possible_actions(state))
                    state = self.simulate_action(state, action)
                    
                # Backpropagation
                result = self.get_result(state)
                while node is not None:
                    node.update(result)
                    node = node.parent
                    
            # Choose the action that leads to the most visited child
            if self.root.children:
                return max(self.root.children, key=lambda c: c.visits).action
            return None
            
        except Exception as e:
            logger.error(f"Error in MCTS simulation: {str(e)}")
            return None
        
    def simulate_action(self, state: Any, action: str) -> Any:
        """
        Simulate an action on the current state.
        
        Args:
            state: Current game state
            action: Action to simulate
            
        Returns:
            Any: Resulting state
        """
        try:
            return state.simulate_action(action)
        except Exception as e:
            logger.error(f"Error simulating action: {str(e)}")
            return state
        
    def is_terminal(self, state: Any) -> bool:
        """
        Check if the state is terminal.
        
        Args:
            state: Game state to check
            
        Returns:
            bool: True if state is terminal
        """
        try:
            return state.is_terminal()
        except Exception as e:
            logger.error(f"Error checking terminal state: {str(e)}")
            return True
        
    def get_possible_actions(self, state: Any) -> List[str]:
        """
        Get list of possible actions from current state.
        
        Args:
            state: Current game state
            
        Returns:
            List[str]: List of possible actions
        """
        try:
            return state.get_available_actions()
        except Exception as e:
            logger.error(f"Error getting possible actions: {str(e)}")
            return []
        
    def get_result(self, state: Any) -> float:
        """
        Get the result of a terminal state.
        
        Args:
            state: Terminal game state
            
        Returns:
            float: Result value (0.0 to 1.0)
        """
        try:
            return state.get_result()
        except Exception as e:
            logger.error(f"Error getting result: {str(e)}")
            return 0.0
            
    def save(self, filepath: str) -> None:
        """
        Save the MCTS tree to disk.
        
        Args:
            filepath: Path to save the tree
        """
        import pickle
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.root, f)
        except Exception as e:
            logger.error(f"Error saving MCTS tree: {str(e)}")
            
    def load(self, filepath: str) -> None:
        """
        Load the MCTS tree from disk.
        
        Args:
            filepath: Path to load the tree from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid data
        """
        import pickle
        try:
            with open(filepath, 'rb') as f:
                self.root = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tree file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid tree file: {str(e)}") 