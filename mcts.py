"""
Monte Carlo Tree Search Implementation
This module implements the MCTS algorithm for strategic decision-making in the word game.
It uses simulation and backpropagation to build a game tree and select optimal moves.
"""

from typing import List, Tuple, Optional, Dict
import random
import math
from game_state import GameState

class MCTSNode:
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None):
        # Game state at this node
        self.game_state = game_state
        
        # Parent node in the tree
        self.parent = parent
        
        # Child nodes
        self.children: List['MCTSNode'] = []
        
        # Statistics for UCB1 calculation
        self.visits = 0
        self.wins = 0
        
        # Move that led to this node
        self.move: Optional[Tuple[int, int, str]] = None

class MCTS:
    def __init__(self):
        # Configuration parameters
        self.exploration_weight = 1.414  # UCB1 exploration parameter
        self.simulation_count = 1000     # Number of simulations per move
        self.max_depth = 50              # Maximum simulation depth
        
        # Cache for game state evaluations
        self.evaluation_cache: Dict[str, float] = {}

    def get_moves(self, game_state: GameState) -> List[Tuple[int, int, str]]:
        """
        Get best moves using MCTS
        Steps:
        1. Build game tree through simulations
        2. Select most promising moves
        3. Return top moves
        """
        root = MCTSNode(game_state)
        
        # Perform simulations
        for _ in range(self.simulation_count):
            self._simulate(root)
        
        # Select best moves
        moves = []
        for child in root.children:
            if child.visits > 0:
                moves.append((child.move, child.wins / child.visits))
        
        # Sort by win rate
        moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in moves[:3]]  # Return top 3 moves

    def _simulate(self, node: MCTSNode):
        """
        Perform a single MCTS simulation
        Steps:
        1. Selection: Traverse tree using UCB1
        2. Expansion: Add new node if game not over
        3. Simulation: Play random moves to end
        4. Backpropagation: Update statistics
        """
        # Selection and Expansion
        leaf = self._select(node)
        
        # Simulation
        result = self._rollout(leaf.game_state)
        
        # Backpropagation
        self._backpropagate(leaf, result)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select node using UCB1
        Implementation:
        1. If node not fully expanded, expand
        2. Otherwise, select child with highest UCB1
        3. Repeat until leaf node
        """
        while not node.game_state.game_over:
            if len(node.children) < len(self._get_legal_moves(node.game_state)):
                return self._expand(node)
            else:
                node = self._select_child(node)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand tree by adding new child node
        Steps:
        1. Get untried moves
        2. Select random untried move
        3. Create new child node
        """
        legal_moves = self._get_legal_moves(node.game_state)
        tried_moves = {child.move for child in node.children}
        untried_moves = [move for move in legal_moves if move not in tried_moves]
        
        if not untried_moves:
            return node
            
        move = random.choice(untried_moves)
        new_state = self._apply_move(node.game_state, move)
        child = MCTSNode(new_state, node)
        child.move = move
        node.children.append(child)
        return child

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using UCB1 formula
        UCB1 = wins/visits + c * sqrt(ln(parent_visits)/visits)
        """
        log_parent_visits = math.log(node.visits)
        
        def ucb1(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            exploit = child.wins / child.visits
            explore = self.exploration_weight * math.sqrt(log_parent_visits / child.visits)
            return exploit + explore
        
        return max(node.children, key=ucb1)

    def _rollout(self, game_state: GameState) -> float:
        """
        Simulate random play until game end
        Returns:
        - 1.0 for win
        - 0.5 for draw
        - 0.0 for loss
        """
        state = game_state.copy()
        depth = 0
        
        while not state.game_over and depth < self.max_depth:
            moves = self._get_legal_moves(state)
            if not moves:
                break
            move = random.choice(moves)
            state.apply_move(move)
            depth += 1
        
        if state.winner:
            return 1.0 if state.winner else 0.0
        return 0.5

    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Update statistics along path to root
        Implementation:
        1. Update visits and wins
        2. Propagate to parent
        3. Adjust result for opponent's perspective
        """
        while node:
            node.visits += 1
            node.wins += result
            result = 1 - result  # Switch perspective
            node = node.parent

    def _get_legal_moves(self, game_state: GameState) -> List[Tuple[int, int, str]]:
        """Get all legal moves for current game state"""
        # Implementation depends on game rules
        pass

    def _apply_move(self, game_state: GameState, move: Tuple[int, int, str]) -> GameState:
        """Create new game state after applying move"""
        new_state = game_state.copy()
        new_state.apply_move(move)
        return new_state 