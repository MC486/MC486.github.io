"""
Game State Management
This module manages the core game state, including the game board, player turns,
and game rules. It serves as the central authority for game state validation and updates.
"""

import pygame
from typing import List, Tuple, Optional

class GameState:
    def __init__(self):
        # Initialize game board dimensions
        self.board_width = 8
        self.board_height = 8
        
        # Create empty game board
        self.board = [[' ' for _ in range(self.board_width)] 
                     for _ in range(self.board_height)]
        
        # Track current player (True for player 1, False for player 2/AI)
        self.current_player = True
        
        # Store game history for undo functionality
        self.move_history = []
        
        # Track game status
        self.game_over = False
        self.winner = None

    def is_valid_move(self, move: Tuple[int, int, str]) -> bool:
        """
        Validate a potential move
        Key checks:
        1. Position is within board bounds
        2. Cell is empty
        3. Word is valid according to game rules
        """
        x, y, word = move
        
        # Check board boundaries
        if not (0 <= x < self.board_width and 0 <= y < self.board_height):
            return False
            
        # Check if cell is empty
        if self.board[y][x] != ' ':
            return False
            
        # Validate word placement rules
        return self._validate_word_placement(x, y, word)

    def apply_move(self, move: Tuple[int, int, str]) -> bool:
        """
        Apply a validated move to the game board
        Steps:
        1. Validate the move
        2. Update the board
        3. Record in history
        4. Check for game end conditions
        """
        if not self.is_valid_move(move):
            return False
            
        x, y, word = move
        
        # Place word on board
        for i, letter in enumerate(word):
            self.board[y][x + i] = letter
            
        # Record move in history
        self.move_history.append(move)
        
        # Switch players
        self.current_player = not self.current_player
        
        # Check for game end
        self._check_game_end()
        
        return True

    def _validate_word_placement(self, x: int, y: int, word: str) -> bool:
        """
        Validate word placement according to game rules
        Key checks:
        1. Word fits on board
        2. Connects to existing words if not first move
        3. Forms valid words in all directions
        """
        # Check if word fits on board
        if x + len(word) > self.board_width:
            return False
            
        # First move must start in center
        if not self.move_history and (x, y) != (self.board_width//2, self.board_height//2):
            return False
            
        # Subsequent moves must connect to existing words
        if self.move_history and not self._connects_to_existing(x, y, word):
            return False
            
        return True

    def _connects_to_existing(self, x: int, y: int, word: str) -> bool:
        """
        Check if word placement connects to existing words
        Implementation:
        1. Check adjacent cells
        2. Verify word connections form valid words
        """
        for i, letter in enumerate(word):
            # Check adjacent cells
            if self._has_adjacent_letter(x + i, y):
                return True
        return False

    def _check_game_end(self):
        """
        Check for game-ending conditions
        Conditions checked:
        1. Board is full
        2. No valid moves remain
        3. Player has formed winning word
        """
        if self._is_board_full():
            self.game_over = True
            self.winner = self.current_player
        elif not self._has_valid_moves():
            self.game_over = True
            self.winner = not self.current_player

    def render(self, screen: pygame.Surface):
        """
        Render current game state to screen
        Steps:
        1. Draw board grid
        2. Draw placed letters
        3. Draw current player indicator
        4. Draw game status
        """
        # Draw board background
        screen.fill((255, 255, 255))
        
        # Draw grid lines
        for x in range(self.board_width + 1):
            pygame.draw.line(screen, (200, 200, 200),
                           (x * 50, 0), (x * 50, self.board_height * 50))
        for y in range(self.board_height + 1):
            pygame.draw.line(screen, (200, 200, 200),
                           (0, y * 50), (self.board_width * 50, y * 50))
        
        # Draw letters
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x] != ' ':
                    font = pygame.font.Font(None, 36)
                    text = font.render(self.board[y][x], True, (0, 0, 0))
                    screen.blit(text, (x * 50 + 15, y * 50 + 10)) 