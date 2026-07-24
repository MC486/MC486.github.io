"""A tiny synchronous publish/subscribe event bus.

Borrowed (in spirit) from the word-game engine: components communicate through
events rather than direct calls, which keeps the backtest loop decoupled from
logging, metrics collection, and persistence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, DefaultDict, Dict, List


class EventType(Enum):
    BAR_CLOSED = "bar_closed"          # a new bar's data is available
    ORDER_FILLED = "order_filled"      # a rebalance/trade executed
    EQUITY_UPDATED = "equity_updated"  # portfolio marked-to-market
    BACKTEST_ENDED = "backtest_ended"


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


Listener = Callable[[Event], None]


class EventBus:
    """Minimal synchronous event bus."""

    def __init__(self) -> None:
        self._listeners: DefaultDict[EventType, List[Listener]] = defaultdict(list)

    def subscribe(self, event_type: EventType, listener: Listener) -> None:
        self._listeners[event_type].append(listener)

    def emit(self, event: Event) -> None:
        for listener in self._listeners.get(event.type, []):
            listener(event)
