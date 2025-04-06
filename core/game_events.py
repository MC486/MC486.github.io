from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

class EventType(Enum):
    """Enumeration of all possible game events"""
    # Game flow events
    GAME_START = "game_start"
    GAME_END = "game_end"
    GAME_QUIT = "game_quit"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    
    # Player action events
    WORD_SUBMITTED = "word_submitted"
    WORD_VALIDATED = "word_validated"
    SCORE_UPDATED = "score_updated"
    BOGGLE_REQUESTED = "boggle_requested"
    INVALID_WORD = "invalid_word"
    LETTERS_REDRAWN = "letters_redrawn"
    
    # AI events
    AI_TURN_START = "ai_turn_start"
    AI_WORD_SELECTED = "ai_word_selected"
    AI_ACTION_STARTED = "ai_action_started"
    AI_ACTION_COMPLETED = "ai_action_completed"
    AI_ACTION_FAILED = "ai_action_failed"
    
    # Analysis events
    AI_ANALYSIS_START = "ai_analysis_start"
    AI_ANALYSIS_COMPLETE = "ai_analysis_complete"
    AI_CANDIDATES_GENERATED = "ai_candidates_generated"
    AI_SCORING_UPDATE = "ai_scoring_update"
    AI_DECISION_MADE = "ai_decision_made"
    MODEL_STATE_UPDATE = "model_state_update"
    
    # Game settings events
    DIFFICULTY_CHANGED = "difficulty_changed"

@dataclass
class GameEvent:
    """
    Represents a game event with its type, associated data, and metadata.
    
    Attributes:
        type (EventType): The type of event
        data (Dict[str, Any]): Event-specific data
        timestamp (datetime): When the event occurred
        debug_data (Dict[str, Any]): Optional analysis/debug information
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()
    debug_data: Dict[str, Any] = None

    def __post_init__(self):
        """Validate event data after initialization"""
        if self.debug_data is None:
            self.debug_data = {}