from typing import Dict, List, Callable, Optional
import logging
from collections import defaultdict
from game_events import GameEvent, EventType

class GameEventManager:
    """
    Manages game events using the Observer pattern. Handles event subscription,
    emission, and debugging/analysis output.
    
    The GameEventManager is a singleton to ensure consistent event handling
    across the entire game system.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Skip initialization if already initialized
        if hasattr(self, 'initialized'):
            return
            
        # Dictionary of event types to list of callback functions
        self.listeners: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Analysis output handlers
        self.dev_output = None  # VS Code output
        self.user_output = None  # Game UI output
        self.history_output = None  # Log file output
        
        # Event history for debugging and analysis
        self.event_history: List[GameEvent] = []
        self.max_history_size = 1000
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.initialized = True

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe a callback function to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event occurs
        """
        self.listeners[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Remove a callback function from an event type's subscribers.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self.listeners:
            self.listeners[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from {event_type.value}")

    def emit(self, event: GameEvent) -> None:
        """
        Emit an event to all subscribed listeners.
        
        Args:
            event: The GameEvent to emit
        """
        # Store event in history
        self._update_history(event)
        
        # Process analysis data if present
        self._handle_analysis(event)
        
        # Notify all listeners
        for callback in self.listeners[event.type]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in event listener: {e}")
                # Continue processing other listeners despite error

    def _update_history(self, event: GameEvent) -> None:
        """
        Update event history, maintaining maximum size.
        
        Args:
            event: Event to add to history
        """
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

    def _handle_analysis(self, event: GameEvent) -> None:
        """
        Process and route analysis data to appropriate outputs.
        
        Args:
            event: Event containing analysis data
        """
        if not event.debug_data:
            return
            
        # Format analysis data for different outputs
        if self.dev_output:
            self.dev_output.process_analysis(event)
        if self.user_output:
            self.user_output.process_analysis(event)
        if self.history_output:
            self.history_output.process_analysis(event)

    def clear_listeners(self) -> None:
        """Remove all event listeners."""
        self.listeners.clear()
        self.logger.debug("All event listeners cleared")

    def get_recent_events(self, event_type: Optional[EventType] = None) -> List[GameEvent]:
        """
        Get recent events, optionally filtered by type.
        
        Args:
            event_type: Optional filter for specific event type
            
        Returns:
            List of recent GameEvents
        """
        if event_type is None:
            return self.event_history.copy()
        return [e for e in self.event_history if e.type == event_type]