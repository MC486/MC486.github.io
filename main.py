# main.py

"""
Main entry point for the Word Strategy AI game.
Handles player setup, game initialization, and loop management.
"""

from engine.game_loop import run_game_loop
from engine.game_state import GameState
import logging

# Setup logger
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

def main():
    print("Welcome to the Word Strategy AI Game!")
    player_name = input("Please enter your name: ").strip()
    if not player_name:
        print("Invalid name. Exiting.")
        return

    # Initialize game state with player name
    game_state = GameState(player_name=player_name)

    # Start the game loop
    run_game_loop(game_state)

if __name__ == "__main__":
    main()
