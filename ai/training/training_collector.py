from typing import Dict, List, Set, Tuple
from datetime import datetime
import random
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from .game_history_tracker import GameHistoryTracker, GameRecord, TurnData

class TrainingDataCollector:
    """
    Collects and prepares training data for AI models from game history.
    Processes game records into formats suitable for different AI training approaches.
    """
    def __init__(self, event_manager: GameEventManager, history_tracker: GameHistoryTracker):
        self.event_manager = event_manager
        self.history_tracker = history_tracker
        self.training_samples: List[Dict] = []
        
        # Training data statistics
        self.total_games_processed = 0
        self.total_turns_processed = 0
        
        # Subscribe to relevant events
        self.event_manager.subscribe(EventType.GAME_END, self._handle_game_end)

    def prepare_training_data(self) -> Dict[str, List]:
        """
        Prepare training data in different formats for various AI models.
        
        Returns:
            Dictionary containing different training data formats:
            - sequence_data: for Markov Chain
            - feature_data: for Naive Bayes
            - state_action_pairs: for Q-Learning
            - game_trees: for Monte Carlo
        """
        game_history = self.history_tracker.get_game_history()
        
        self.event_manager.emit(GameEvent(
            type=EventType.AI_ANALYSIS_START,
            data={"message": "Preparing training data"},
            debug_data={"games_count": len(game_history)}
        ))
        
        return {
            "sequence_data": self._prepare_sequence_data(game_history),
            "feature_data": self._prepare_feature_data(game_history),
            "state_action_pairs": self._prepare_rl_data(game_history),
            "game_trees": self._prepare_mcts_data(game_history)
        }

    def _prepare_sequence_data(self, games: List[GameRecord]) -> List[Dict]:
        """Prepare letter sequence data for Markov Chain training"""
        sequences = []
        for game in games:
            for turn in game.turns:
                sequences.append({
                    "word": turn.word,
                    "letters_used": list(turn.letters_used),
                    "score": turn.score
                })
        return sequences

    def _prepare_feature_data(self, games: List[GameRecord]) -> List[Dict]:
        """Prepare feature-based data for Naive Bayes training"""
        features = []
        for game in games:
            for turn in game.turns:
                features.append({
                    "word_length": len(turn.word),
                    "vowel_count": sum(1 for c in turn.word if c in 'AEIOU'),
                    "shared_letters_used": len(turn.letters_used.intersection(turn.shared_letters)),
                    "score": turn.score
                })
        return features

    def _prepare_rl_data(self, games: List[GameRecord]) -> List[Dict]:
        """Prepare state-action pairs for Q-Learning training"""
        rl_data = []
        for game in games:
            for i, turn in enumerate(game.turns):
                state = {
                    "available_letters": turn.shared_letters + turn.private_letters,
                    "turn_number": turn.turn_number,
                    "game_progress": turn.turn_number / len(game.turns)
                }
                action = turn.word
                reward = turn.score
                
                rl_data.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": None if i == len(game.turns) - 1 else {
                        "available_letters": game.turns[i + 1].shared_letters + game.turns[i + 1].private_letters,
                        "turn_number": game.turns[i + 1].turn_number,
                        "game_progress": game.turns[i + 1].turn_number / len(game.turns)
                    }
                })
        return rl_data

    def _prepare_mcts_data(self, games: List[GameRecord]) -> List[Dict]:
        """Prepare game tree data for Monte Carlo Tree Search training"""
        tree_data = []
        for game in games:
            game_tree = {
                "initial_state": {
                    "shared_letters": game.turns[0].shared_letters,
                    "private_letters": game.turns[0].private_letters
                },
                "moves": [{
                    "word": turn.word,
                    "score": turn.score,
                    "letters_after": turn.shared_letters + turn.private_letters
                } for turn in game.turns]
            }
            tree_data.append(game_tree)
        return tree_data

    def _handle_game_end(self, event: GameEvent) -> None:
        """Process completed game data"""
        self.total_games_processed += 1
        self.total_turns_processed += event.data.get("total_turns", 0)
        
        self.event_manager.emit(GameEvent(
            type=EventType.MODEL_STATE_UPDATE,
            data={"message": "Training data updated"},
            debug_data={
                "total_games": self.total_games_processed,
                "total_turns": self.total_turns_processed
            }
        ))

    def get_training_stats(self) -> Dict:
        """Get statistics about collected training data"""
        return {
            "total_games": self.total_games_processed,
            "total_turns": self.total_turns_processed,
            "average_turns_per_game": (
                self.total_turns_processed / self.total_games_processed 
                if self.total_games_processed > 0 else 0
            )
        }