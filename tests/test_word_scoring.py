# tests/test_word_scoring.py
# Unit tests for word scoring logic

import unittest
from core.word_scoring import score_word

class TestWordScoring(unittest.TestCase):
    def test_base_score(self):
        """
        Tests that a basic word is scored correctly (positive score).
        """
        score = score_word("banana")
        self.assertGreater(score, 0, "Base word should score positive points.")

    def test_repeat_score_penalty(self):
        """
        Tests that the repeat penalty reduces the score for subsequent uses of the same word.
        """
        first = score_word("banana", repeat_count=0) # First use, no penalty.
        second = score_word("banana", repeat_count=1) # Second use, penalty applied.
        third = score_word("banana", repeat_count=3)  # Fourth use, further penalty applied.

        self.assertGreater(first, second, "Second use should have reduced score.")
        self.assertGreater(second, third, "Further uses should reduce score further.")
        self.assertGreaterEqual(third, 1, "Minimum score should still be at least 1.")

    def test_edge_case_short_word(self):
        """
        Tests that even a single-letter word scores at least 1 point.
        """
        score = score_word("a")
        self.assertGreaterEqual(score, 1, "Even single-letter word should yield at least 1 point.")

    def test_score_word_basic(self):
        # Basic high-frequency word
        score_apple = score_word("apple")
        self.assertGreater(score_apple, 0)

        # Less common but valid word
        score_zebra = score_word("zebra")
        self.assertGreater(score_zebra, 0)
            

if __name__ == "__main__":
    unittest.main() # Run the unit tests.