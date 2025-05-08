# tests/test_input_handler.py
# Unit test for player input handling.

import pytest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType
from core.game_events_manager import GameEventManager
from engine.input_handler import InputHandler
from engine.game_state import GameState

@pytest.fixture
def mock_dependencies():
    """
    Creates mock dependencies for tests.
    """
    event_manager = Mock(spec=GameEventManager)
    game_state = Mock()
    game_state.shared_letters = ['A', 'T', 'R', 'S']
    game_state.boggle_letters = ['E', 'L', 'O', 'P', 'U', 'N']

    word_repo = Mock()
    word_repo.get_word_stats = Mock(return_value={})
    word_repo.record_word_usage = Mock()
    word_repo.get_entry_count = Mock(return_value=0)
    
    category_repo = Mock()
    category_repo.get_categories = Mock(return_value=[])
    category_repo.get_category_stats = Mock(return_value={})
    
    return {
        'event_manager': event_manager,
        'game_state': game_state,
        'word_repo': word_repo,
        'category_repo': category_repo
    }

@pytest.fixture
def input_handler(mock_dependencies):
    """
    Creates an input handler instance with mock dependencies.
    """
    return InputHandler(
        event_manager=mock_dependencies['event_manager'],
        word_repo=mock_dependencies['word_repo'],
        category_repo=mock_dependencies['category_repo']
    )

def test_input_normalization(input_handler, mock_dependencies):
    """
    Tests that input is properly normalized (lowercase and stripped whitespace).
    """
    with patch('builtins.input', return_value='  Boggle  '):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'BOGGLE'
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']})
        )

    with patch('builtins.input', return_value=' Quit '):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'QUIT'
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
        )

    with patch('builtins.input', return_value='  Unicorn  '), \
         patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=True):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'unicorn'
        mock_dependencies['event_manager'].emit.assert_not_called()

def test_quit_command(input_handler, mock_dependencies):
    """
    Tests that the quit command is handled correctly.
    """
    with patch('builtins.input', return_value='quit'):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'QUIT'
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.GAME_QUIT, data={'reason': 'player_request'})
        )

def test_boggle_command(input_handler, mock_dependencies):
    """
    Tests that the boggle command is handled correctly.
    """
    with patch('builtins.input', return_value='boggle'):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'BOGGLE'
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.BOGGLE_REQUESTED, data={'current_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']})
        )

def test_valid_word(input_handler, mock_dependencies):
    """
    Tests that a valid word is accepted.
    """
    with patch('builtins.input', return_value='plate'), \
         patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=True):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'plate'
        mock_dependencies['event_manager'].emit.assert_not_called()

def test_invalid_word(input_handler, mock_dependencies):
    """
    Tests that an invalid word is rejected and appropriate event is emitted.
    """
    with patch('builtins.input', return_value='zebra'), \
         patch('core.validation.word_validator.WordValidator.validate_word_with_letters', return_value=False):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'zebra'  # The word is returned but will be rejected by game state
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.INVALID_WORD, data={
                'word': 'zebra',
                'available_letters': ['A', 'T', 'R', 'S', 'E', 'L', 'O', 'P', 'U', 'N']
            })
        )

def test_non_alphabetic_input(input_handler, mock_dependencies):
    """
    Tests that non-alphabetic input is rejected.
    """
    with patch('builtins.input', return_value='123'):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == '123'  # The input is returned but will be rejected by game state
        mock_dependencies['event_manager'].emit.assert_not_called()

def test_keyboard_interrupt(input_handler, mock_dependencies):
    """
    Tests that keyboard interrupt is handled correctly.
    """
    with patch('builtins.input', side_effect=KeyboardInterrupt):
        result = input_handler.get_player_word(mock_dependencies['game_state'])
        assert result == 'QUIT'
        mock_dependencies['event_manager'].emit.assert_called_with(
            Mock(type=EventType.GAME_QUIT, data={'reason': 'keyboard_interrupt'})
        )