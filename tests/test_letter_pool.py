# tests/test_letter_pool.py
# Unit tests for letter pool generation logic

import unittest
from core.letter_pool import generate_letter_pool

class TestLetterPool(unittest.TestCase):
    def test_letter_pool_structure(self):
        """
        Tests that the letter pools are generated with the correct structure and constraints.
        """
        shared, boggle = generate_letter_pool() # Generate letter pools.

        self.assertEqual(len(shared), 4, "Shared pool must have 4 letters.") # Check if shared pool has 4 letters.
        self.assertTrue(any(l in 'AEIOU' for l in shared), "Shared pool must contain at least one vowel.") # Check if shared pool has at least one vowel.
        self.assertTrue(any(l not in 'AEIOU' for l in shared), "Shared pool must contain at least one consonant.") # Check if shared pool has at least one consonant.
        self.assertEqual(len(set(shared)), 4, "Shared letters must be distinct.") # Check if all shared letters are unique.

        self.assertIn(len(boggle), [6, 7], "Boggle pool must contain 6 or 7 letters.") # Check if boggle pool has 6 or 7 letters.

    def test_randomness_and_distribution(self):
        """
        Tests that the letter pools show variety and randomness across multiple samples.
        """
        all_samples = set() # Set to store all letters generated.
        for _ in range(50): # Generate 50 letter pools.
            shared, boggle = generate_letter_pool() # Generate letter pools.
            all_samples.update(shared + boggle) # Add the generated letters to the set.

        self.assertGreater(len(all_samples), 10, "Letter pool should have variety across samples.") # Check if there are at least 10 unique letters generated.

if __name__ == "__main__":
    unittest.main() # Run the unit tests.