# tests/test_mcts_model.py

import unittest
from ai.mcts_model import MCTS
from nltk.corpus import words as nltk_words


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.valid_words = set(nltk_words.words())
        self.model = MCTS(valid_words=self.valid_words, max_depth=4, simulations=20)

    def test_mcts_returns_valid_output(self):
        shared_letters = ['a', 't', 'e', 'r']
        private_letters = ['s', 'h', 'o', 'u']
        result = self.model.run(shared_letters, private_letters, word_length=5)
        self.assertTrue(isinstance(result, str))
        self.assertLessEqual(len(result), 5)
