"""
AI Controller Implementation
This module manages the AI's decision-making process, combining multiple AI strategies
(MCTS, Q-Learning, and Naive Bayes) to determine optimal moves. It serves as the
interface between the game state and the AI algorithms.
"""

from typing import List, Tuple, Optional
from game_state import GameState
from mcts import MCTS
from q_learning import QLearning
from naive_bayes import NaiveBayes

class AIController:
    def __init__(self):
        # Initialize AI components
        self.mcts = MCTS()
        self.q_learning = QLearning()
        self.naive_bayes = NaiveBayes()
        
        # Track game history for learning
        self.game_history = []
        
        # Configuration for AI behavior
        self.use_mcts = True
        self.use_q_learning = True
        self.use_naive_bayes = True
        
        # Weights for combining AI strategies
        self.strategy_weights = {
            'mcts': 0.4,
            'q_learning': 0.4,
            'naive_bayes': 0.2
        }

    def get_move(self, game_state: GameState) -> Tuple[int, int, str]:
        """
        Determine the best move using combined AI strategies
        Steps:
        1. Get moves from each AI component
        2. Score and rank moves
        3. Combine scores using weights
        4. Select best move
        """
        # Get potential moves from each strategy
        mcts_moves = self.mcts.get_moves(game_state) if self.use_mcts else []
        q_learning_moves = self.q_learning.get_moves(game_state) if self.use_q_learning else []
        naive_bayes_moves = self.naive_bayes.get_moves(game_state) if self.use_naive_bayes else []
        
        # Combine and score all moves
        scored_moves = self._score_moves(
            mcts_moves, q_learning_moves, naive_bayes_moves
        )
        
        # Select best move
        best_move = max(scored_moves, key=lambda x: x[1])
        return best_move[0]

    def _score_moves(self, 
                    mcts_moves: List[Tuple[int, int, str]],
                    q_learning_moves: List[Tuple[int, int, str]],
                    naive_bayes_moves: List[Tuple[int, int, str]]) -> List[Tuple[Tuple[int, int, str], float]]:
        """
        Score and combine moves from different strategies
        Implementation:
        1. Normalize scores from each strategy
        2. Apply strategy weights
        3. Combine scores for each move
        """
        scored_moves = []
        
        # Score MCTS moves
        for move in mcts_moves:
            mcts_score = self.mcts.evaluate_move(move)
            q_score = self.q_learning.evaluate_move(move) if self.use_q_learning else 0
            nb_score = self.naive_bayes.evaluate_move(move) if self.use_naive_bayes else 0
            
            # Combine scores using weights
            total_score = (
                mcts_score * self.strategy_weights['mcts'] +
                q_score * self.strategy_weights['q_learning'] +
                nb_score * self.strategy_weights['naive_bayes']
            )
            
            scored_moves.append((move, total_score))
        
        return scored_moves

    def update_strategy_weights(self, 
                              mcts_weight: float,
                              q_learning_weight: float,
                              naive_bayes_weight: float):
        """
        Update the weights for combining AI strategies
        Ensures weights sum to 1.0
        """
        total = mcts_weight + q_learning_weight + naive_bayes_weight
        self.strategy_weights = {
            'mcts': mcts_weight / total,
            'q_learning': q_learning_weight / total,
            'naive_bayes': naive_bayes_weight / total
        }

    def learn_from_game(self, game_history: List[Tuple[int, int, str]]):
        """
        Update AI models based on game history
        Steps:
        1. Update Q-Learning model
        2. Update Naive Bayes model
        3. Store game in history
        """
        self.game_history.append(game_history)
        
        # Update Q-Learning
        if self.use_q_learning:
            self.q_learning.update(game_history)
        
        # Update Naive Bayes
        if self.use_naive_bayes:
            self.naive_bayes.update(game_history)
        
        # Limit history size
        if len(self.game_history) > 100:
            self.game_history.pop(0) 