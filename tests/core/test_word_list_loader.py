# tests/test_word_list_loader.py
# Unit test for word list loading utility.

import unittest
from utils.word_list_loader import load_word_list

class TestWordListLoader(unittest.TestCase):
    def test_word_list_is_loaded(self):
        """
        Tests that the word list is loaded and is not empty.
        """
        word_set = load_word_list() # Load the word list.
        self.assertIsInstance(word_set, set) # Check if the loaded object is a set.
        self.assertGreater(len(word_set), 0, "The word list should not be empty.") # Check if the word list has at least one word.

    def test_common_words_present(self):
        """
        Tests that common English words are present in the loaded word list.
        """
        word_set = load_word_list() # Load the word list.
        # Check for a few high-frequency English words
        for word in ["orange", "apple", "tree", "game", "player"]:
            self.assertIn(word, word_set, f"Expected word '{word}' not found in word list.") # Check if common words are in the loaded set.

if __name__ == "__main__":
    unittest.main() # Run the unit tests.