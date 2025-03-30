import unittest
from unittest.mock import Mock, patch
from core.game_events import GameEvent, EventType, GameEventManager
from ai.word_analysis import WordFrequencyAnalyzer
from ai.models.naive_bayes import NaiveBayes

class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.event_manager = Mock(spec=GameEventManager)
        self.word_analyzer = Mock(spec=WordFrequencyAnalyzer)
        self.naive_bayes = NaiveBayes(
            event_manager=self.event_manager,
            word_analyzer=self.word_analyzer
        )

    def test_initialization(self):
        """Test proper initialization of NaiveBayes"""
        self.assertEqual(self.naive_bayes.total_words, 0)
        self.assertEqual(len(self.naive_bayes.length_probs), 0)
        self.assertEqual(len(self.naive_bayes.position_probs), 0)
        self.assertEqual(len(self.naive_bayes.pair_probs), 0)
        self.assertEqual(len(self.naive_bayes.score_priors), 0)

    def test_train_single_word(self):
        """Test training on a single word"""
        test_word = "HELLO"
        test_score = 10
        self.naive_bayes._train_single_word(test_word, test_score)

        # Check length probabilities
        self.assertIn(5, self.naive_bayes.length_probs)
        
        # Check position probabilities
        self.assertIn(0, self.naive_bayes.position_probs)  # First position
        self.assertIn('H', self.naive_bayes.position_probs[0])
        
        # Check pair probabilities
        self.assertIn('H', self.naive_bayes.pair_probs)
        self.assertIn('E', self.naive_bayes.pair_probs['H'])
        
        # Check score priors
        self.assertIn(test_score, self.naive_bayes.score_priors)

    def test_train_multiple_words(self):
        """Test training on multiple words with scores"""
        training_data = [
            ("HELLO", 10),
            ("HELP", 8),
            ("HEAP", 7)
        ]
        self.naive_bayes.train(training_data)

        # Check common patterns were learned
        self.assertIn('H', self.naive_bayes.position_probs[0])
        self.assertIn('E', self.naive_bayes.position_probs[1])

    def test_probability_calculation(self):
        """Test probability calculations"""
        self.naive_bayes.train([("HELLO", 10), ("HELP", 8)])
        self.naive_bayes._calculate_probabilities()

        # Check probabilities sum to 1 for each position
        for pos in self.naive_bayes.position_probs:
            prob_sum = sum(self.naive_bayes.position_probs[pos].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_word_probability_estimation(self):
        """Test word probability estimation"""
        self.naive_bayes.train([("HELLO", 10), ("HELP", 8), ("HEAP", 7)])
        
        # Test probability of seen word
        prob_hello = self.naive_bayes.estimate_word_probability("HELLO")
        self.assertGreater(prob_hello, 0.0)
        
        # Test probability of unseen word
        prob_unseen = self.naive_bayes.estimate_word_probability("XYZ")
        self.assertGreaterEqual(prob_unseen, 0.0)  # Should handle unseen words

    def test_score_prediction(self):
        """Test score prediction"""
        training_data = [("HELLO", 10), ("HELP", 8), ("HEAP", 7)]
        self.naive_bayes.train(training_data)
        
        predicted_score = self.naive_bayes.predict_score("HELLO")
        self.assertGreaterEqual(predicted_score, 0)

    def test_candidate_evaluation(self):
        """Test candidate word evaluation"""
        self.naive_bayes.train([("HELLO", 10), ("HELP", 8)])
        
        candidates = ["HELLO", "HELP", "NEW"]
        evaluations = self.naive_bayes.evaluate_candidates(candidates)
        
        self.assertEqual(len(evaluations), len(candidates))
        self.assertTrue(all(isinstance(score, float) for _, score in evaluations))
        self.assertTrue(all(score >= 0 for _, score in evaluations))

    def test_event_handling(self):
        """Test event handling"""
        event = GameEvent(
            type=EventType.WORD_SUBMITTED,
            data={
                "word": "HELLO",
                "score": 10
            }
        )
        
        self.naive_bayes._handle_word_submission(event)
        self.assertEqual(self.naive_bayes.total_words, 1)

    def test_model_statistics(self):
        """Test model statistics reporting"""
        self.naive_bayes.train([("HELLO", 10), ("HELP", 8)])
        
        stats = self.naive_bayes.get_model_stats()
        self.assertIn("total_words_trained", stats)
        self.assertIn("unique_lengths", stats)
        self.assertIn("unique_scores", stats)

    def test_laplace_smoothing(self):
        """Test Laplace smoothing for unseen events"""
        self.naive_bayes.train([("HELLO", 10)])
        
        # Test probability for unseen letter in seen position
        prob = self.naive_bayes.position_probs[0].get('X', 0)
        self.assertGreater(prob, 0)  # Should be smoothed, not zero

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Empty word
        self.naive_bayes._train_single_word("", 0)
        self.assertEqual(self.naive_bayes.total_words, 0)
        
        # None score
        self.naive_bayes._train_single_word("HELLO", None)
        self.assertEqual(len(self.naive_bayes.score_priors), 0)
        
        # Empty candidate list
        evaluations = self.naive_bayes.evaluate_candidates([])
        self.assertEqual(len(evaluations), 0)

if __name__ == '__main__':
    unittest.main()