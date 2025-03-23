# tests/test_input_handler.py
# Unit test for player input handling.

import builtins
import pytest
from engine.input_handler import get_player_input

def test_get_player_input(monkeypatch):
    """
    Tests user input normalization.
    """
    monkeypatch.setattr(builtins, 'input', lambda _: "  Boggle  ")
    assert get_player_input() == "boggle"

    monkeypatch.setattr(builtins, 'input', lambda _: " Quit ")
    assert get_player_input() == "quit"

    monkeypatch.setattr(builtins, 'input', lambda _: "unicorn")
    assert get_player_input() == "unicorn"
