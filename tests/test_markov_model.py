# tests/test_markov_model.py

import unittest
from ai.markov_model import MarkovModel


class TestMarkovModel(unittest.TestCase):
    def setUp(self):
        self.model = MarkovModel()
        self.training_words = ["hello", "hell", "helmet", "help"]

    def test_train_model(self):
        self.model.train(self.training_words)
        self.assertTrue(self.model.trained)
        self.assertIn('h', self.model.transitions)

    def test_generate_word(self):
        self.model.train(self.training_words)
        word = self.model.generate()
        self.assertIsInstance(word, str)
        self.assertGreater(len(word), 0)

    def test_generate_raises_if_not_trained(self):
        with self.assertRaises(ValueError):
            self.model.generate()
