# engine/game_state.py
# Holds the current game state and manages game progress.

import logging
from core.letter_pool import generate_letter_pool  # Function to generate letter pools.
from core.word_scoring import score_word  # Function to calculate word scores.
from utils.word_list_loader import load_word_list  # Function to load valid words.

logger = logging.getLogger(__name__)

class GameState:
    def __init__(self):
        """
        Initializes game state variables.
        """
        self.shared_letters = []
        self.boggle_letters = []
        self.player_score = 0
        self.used_words = set()
        self.word_usage_counts = {}
        self.is_game_over = False
        self.player_name = None
        self.valid_words = set()  # Set of valid dictionary words

    def initialize_game(self):
        """
        Initializes player, letters, and game tracking.
        """
        self.player_name = input("🕺🏼Enter your name: ").strip()
        self.shared_letters, self.boggle_letters = generate_letter_pool()
        self.valid_words = load_word_list() # Load the valid word list from the dictionary.
        logger.info(f"🐦‍🔥 Game started for player {self.player_name}.")

    def redraw_boggle_letters(self):
        """
        Regenerates the player's boggle letters.
        """
        _, self.boggle_letters = generate_letter_pool()
        print("✨New boggle letters drawn.✨")

    def display_status(self):
        """
        Displays the current game state to the player.
        """
        print(f"\nShared Letters: {' '.join(self.shared_letters)}")
        print(f"Boggle Letters: {' '.join(self.boggle_letters)}")
        print(f"Current Score: {self.player_score} 🏆")

    def process_turn(self, word):
        """
        Validates and scores the player word, stores data, and checks for AI win.
        """
        if not word:
            print("🚫 No word entered.🚫")
            return

        if word not in self.valid_words:
            print(f"🤔'{word}' is not a recognized English word. Try again.🤔")
            return

        repeat_count = self.word_usage_counts.get(word, 0) # Get the number of times the word has been used.
        if repeat_count > 0:
            print("⚠️ You already used this word. Score will be reduced. ⚠️")

        score = score_word(word, repeat_count) # Score the word based on rarity and repeat count.
        self.player_score += score
        self.used_words.add(word)
        self.word_usage_counts[word] = repeat_count + 1 # Increment the word usage count.

        print(f"Word '{word}' scored {score} points. 🎉")
        print(f"New Score: {self.player_score} 📈")

        # Placeholder for AI guess
        ai_guess = "PLACEHOLDER"
        if ai_guess == word:
            print(f"🤖 AI guessed your word! It was '{ai_guess}'. 🤯")
            self.is_game_over = True

    def display_game_over(self):
        """
        Ends the game and displays final score.
        """
        print(f"\nGame Over! 🏁 Final Score for {self.player_name}: {self.player_score} 🏆")
        logger.info("Game ended.")