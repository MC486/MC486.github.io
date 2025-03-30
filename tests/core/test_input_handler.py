# tests/test_input_handler.py
# Unit test for player input handling.

import builtins
import pytest
from engine.input_handler import get_player_input

def test_get_player_input(monkeypatch):
    """
    Tests user input normalization (lowercase and stripped whitespace).
    """
    # monkeypatch.setattr: Temporarily replaces an attribute of an object/module during the test.
    # builtins: The module containing built-in functions like input().
    # 'input': The attribute (the input() function) we want to replace.
    # lambda _: "  Boggle  ": A lambda function that simulates user input, returning "  Boggle  ".
    #    The lambda function accepts any input (represented by '_') but ignores it.
    monkeypatch.setattr(builtins, 'input', lambda _: "  Boggle  ")
    assert get_player_input() == "boggle" # Assert that the input is converted to lowercase and whitespace is removed.

    # Same pattern as above, but simulating different user input.
    monkeypatch.setattr(builtins, 'input', lambda _: " Quit ")
    assert get_player_input() == "quit" # Assert that it's correctly normalized to "quit".

    # Simulating a regular word input.
    monkeypatch.setattr(builtins, 'input', lambda _: "unicorn")
    assert get_player_input() == "unicorn" # Assert that it's returned as is (lowercase).