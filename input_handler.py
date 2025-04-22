"""
Input Handler Implementation
This module processes all user input, including mouse clicks, keyboard input,
and game commands. It translates raw input events into game actions and maintains
input state for continuous actions.
"""

import pygame
from typing import Optional, Tuple
from game_state import GameState

class InputHandler:
    def __init__(self):
        # Track mouse state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        
        # Track keyboard state
        self.keys_pressed = set()
        
        # Store current word being typed
        self.current_word = ""
        
        # Track selected board position
        self.selected_pos: Optional[Tuple[int, int]] = None

    def handle_event(self, event: pygame.event.Event, game_state: GameState):
        """
        Process a single input event
        Key event types handled:
        1. Mouse clicks for board interaction
        2. Keyboard input for word entry
        3. Special commands (undo, reset, etc.)
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event, game_state)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(event, game_state)
        elif event.type == pygame.KEYDOWN:
            self._handle_key_down(event, game_state)
        elif event.type == pygame.KEYUP:
            self._handle_key_up(event)

    def _handle_mouse_down(self, event: pygame.event.Event, game_state: GameState):
        """
        Process mouse button press
        Steps:
        1. Update mouse position
        2. Check for board cell selection
        3. Handle special UI elements
        """
        self.mouse_pos = event.pos
        self.mouse_clicked = True
        
        # Convert screen coordinates to board coordinates
        board_x = self.mouse_pos[0] // 50
        board_y = self.mouse_pos[1] // 50
        
        # Validate board position
        if (0 <= board_x < game_state.board_width and 
            0 <= board_y < game_state.board_height):
            self.selected_pos = (board_x, board_y)

    def _handle_mouse_up(self, event: pygame.event.Event, game_state: GameState):
        """
        Process mouse button release
        Implementation:
        1. Reset click state
        2. Finalize drag operations
        3. Complete selection
        """
        self.mouse_clicked = False
        self.mouse_pos = event.pos
        
        # If we have a complete word and position, make the move
        if self.selected_pos and self.current_word:
            x, y = self.selected_pos
            game_state.apply_move((x, y, self.current_word))
            self.current_word = ""
            self.selected_pos = None

    def _handle_key_down(self, event: pygame.event.Event, game_state: GameState):
        """
        Process keyboard input
        Key actions:
        1. Letter input for word building
        2. Backspace for word editing
        3. Enter to submit word
        4. Escape to cancel
        5. Special commands
        """
        # Add to current word if it's a letter
        if event.unicode.isalpha():
            self.current_word += event.unicode.upper()
        # Handle backspace
        elif event.key == pygame.K_BACKSPACE:
            self.current_word = self.current_word[:-1]
        # Submit word
        elif event.key == pygame.K_RETURN:
            if self.selected_pos and self.current_word:
                x, y = self.selected_pos
                game_state.apply_move((x, y, self.current_word))
                self.current_word = ""
                self.selected_pos = None
        # Cancel current action
        elif event.key == pygame.K_ESCAPE:
            self.current_word = ""
            self.selected_pos = None

    def _handle_key_up(self, event: pygame.event.Event):
        """
        Process key release
        Implementation:
        1. Update key state tracking
        2. Handle key repeat behavior
        """
        if event.key in self.keys_pressed:
            self.keys_pressed.remove(event.key)

    def update(self):
        """
        Update continuous input state
        Called every frame to:
        1. Update mouse position
        2. Handle key repeat
        3. Process continuous actions
        """
        self.mouse_pos = pygame.mouse.get_pos()
        
        # Handle key repeat for continuous actions
        keys = pygame.key.get_pressed()
        for key in self.keys_pressed:
            if keys[key]:
                # Handle continuous key actions
                pass 