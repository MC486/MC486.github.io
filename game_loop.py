"""
Game Loop Implementation
This module orchestrates the main game lifecycle, managing the game state, user input,
and AI interactions. It follows a clear sequence of initialization, event handling,
and cleanup.
"""

import pygame
import sys
from game_state import GameState
from input_handler import InputHandler
from ai_controller import AIController

class GameLoop:
    def __init__(self):
        # Step 1: Initialize core game components
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Word Game AI")
        
        # Step 2: Create game state manager
        self.game_state = GameState()
        
        # Step 3: Set up input handling system
        self.input_handler = InputHandler()
        
        # Step 4: Initialize AI controller
        self.ai_controller = AIController()
        
        # Step 5: Set up game clock for consistent timing
        self.clock = pygame.time.Clock()
        self.running = False

    def run(self):
        # Step 6: Start main game loop
        self.running = True
        while self.running:
            # Step 7: Process all events in the queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Step 8: Handle user input through input handler
                self.input_handler.handle_event(event, self.game_state)

            # Step 9: Update game state based on current conditions
            self.game_state.update()
            
            # Step 10: Get AI move if it's AI's turn
            if self.game_state.is_ai_turn():
                ai_move = self.ai_controller.get_move(self.game_state)
                self.game_state.apply_move(ai_move)

            # Step 11: Render current game state
            self.render()
            
            # Step 12: Control game speed
            self.clock.tick(60)

        # Step 13: Clean up resources
        self.cleanup()

    def render(self):
        # Clear screen with background color
        self.screen.fill((255, 255, 255))
        
        # Render game board
        self.game_state.render(self.screen)
        
        # Update display
        pygame.display.flip()

    def cleanup(self):
        # Step 14: Properly shut down pygame
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Step 15: Create and run game instance
    game = GameLoop()
    game.run() 