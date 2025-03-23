# engine/game_state.py
# Holds the current game state and manages game progress.

import logging
from core.letter_pool import generate_letter_pool  # Function to generate letter pools.
from core.word_scoring import score_word  # Function to calculate word scores.

logger = logging.getLogger(__name__)

class GameState:
    def __init__(self):
        """
        Initializes game state variables.
        """
        self.shared_letters = []  # Letters shared between player and AI.
        self.boggle_letters = []  # Letters available only to the player.
        self.player_score = 0  # Player's current score.
        self.used_words = set()  # Set of words already used by the player.
        self.word_usage_counts = {}  # Tracks how many times each word has been used.
        self.is_game_over = False  # Flag indicating if the game is over.
        self.player_name = None  # Player's name.

    def initialize_game(self):
        """
        Initializes player, letters, and game tracking.
        """
        self.player_name = input("Enter your name: ").strip()  # Get player's name from input.
        self.shared_letters, self.boggle_letters = generate_letter_pool()  # Generate shared and boggle letters.
        logger.info(f"Game started for player {self.player_name}.")  # Log game start.

    def redraw_boggle_letters(self):
        """
        Regenerates the player's boggle letters.
        """
        _, self.boggle_letters = generate_letter_pool()  # Generate new boggle letters.
        print("New boggle letters drawn.")

    def display_status(self):
        """
        Displays the current game state to the player.
        """
        print(f"\nShared Letters: {' '.join(self.shared_letters)}")  # Print shared letters.
        print(f"Boggle Letters: {' '.join(self.boggle_letters)}")  # Print boggle letters.
        print(f"Current Score: {self.player_score}")  # Print current score.

    def process_turn(self, word):
        """
        Validates and scores the player word, stores data, and checks for AI win.
        """
        if not word:
            print("No word entered.")
            return

        repeat_count = self.word_usage_counts.get(word, 0)
        if repeat_count > 0:
            print("You already used this word. Score will be reduced.")

        score = score_word(word, repeat_count)  # Score word using repeat count
        self.player_score += score
        self.used_words.add(word)
        self.word_usage_counts[word] = repeat_count + 1  # Update usage count

        print(f"Word '{word}' scored {score} points.")
        print(f"New Score: {self.player_score}")

        # Placeholder for AI guess - Replace this with your AI's guess.
        ai_guess = "PLACEHOLDER"
        if ai_guess == word:
            print(f"AI guessed your word! It was '{ai_guess}'.")
            self.is_game_over = True

    def display_game_over(self):
        """
        Ends the game and displays final score.
        """
        print(f"\nGame Over! Final Score for {self.player_name}: {self.player_score}")
        logger.info("Game ended.")  # Log game end.
