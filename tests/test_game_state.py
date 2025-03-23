# tests/test_game_state.py
# Unit tests for game state logic.

import unittest
from engine.game_state import GameState

class TestGameState(unittest.TestCase):
    def setUp(self):
        """
        Sets up the game state for each test.
        """
        self.state = GameState()
        self.state.shared_letters = ['A', 'T', 'R', 'S']
        self.state.boggle_letters = ['E', 'L', 'O', 'P', 'U', 'N']

    def test_word_scoring_and_storage(self):
        """
        Tests that a word is scored correctly and stored in used words.
        """
        self.state.process_turn("LOPE") # Process the turn with the word "LOPE".
        self.assertGreater(self.state.player_score, 0) # Check if the player's score is greater than 0.
        self.assertIn("LOPE", self.state.used_words) # Check if "LOPE" is in the used words set.

    def test_repeat_word_penalty(self):
        """
        Tests that repeated words are penalized, resulting in a lower score.
        """
        self.state.process_turn("LOPE") # First time using "LOPE".
        score_1 = self.state.player_score # Store the score after the first use.
        self.state.process_turn("LOPE") # Second time using "LOPE".
        score_2 = self.state.player_score # Store the score after the second use.
        self.assertLess(score_2, score_1 * 2)  # Check that the score increase is less than double, confirming diminishing returns.

    def test_word_usage_count_increment(self):
        """
        Tests that word usage counts are correctly incremented.
        """
        self.state.process_turn("LOPE") # First time using "LOPE".
        self.assertEqual(self.state.word_usage_counts["LOPE"], 1) # Check if the usage count is 1.
        self.state.process_turn("LOPE") # Second time using "LOPE".
        self.assertEqual(self.state.word_usage_counts["LOPE"], 2) # Check if the usage count is 2.

if __name__ == "__main__":
    unittest.main() # Run the unit tests.