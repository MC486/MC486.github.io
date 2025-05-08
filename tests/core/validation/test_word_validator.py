import unittest
import os
import tempfile
import logging
from unittest.mock import patch, Mock, mock_open
from core.validation.word_validator import WordValidator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

class TestWordValidator(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_words = "HELLO\nHELP\nHEAP\nWORLD\nWORD"
        
        # Mock NLTK
        self.nltk_words = ["hello", "help", "heap", "world", "word", "cat"]
        self.mock_nltk = Mock()
        self.mock_nltk.words = Mock(return_value=self.nltk_words)
        self.nltk_patcher = patch('nltk.corpus.words', self.mock_nltk)
        self.nltk_patcher.start()

        # Mock NLTK data finder
        self.nltk_data_patcher = patch('nltk.data.find', return_value=True)
        self.mock_nltk_data = self.nltk_data_patcher.start()
        
        # Mock word repository
        self.mock_word_repo = Mock()
        self.mock_word_repo.add_word = Mock()
        self.mock_word_repo.get_word_stats = Mock(return_value={
            'total_words': 5,
            'unique_words': 5,
            'avg_word_length': 4.6
        })

    def tearDown(self):
        """Cleanup after each test"""
        self.nltk_patcher.stop()
        self.nltk_data_patcher.stop()
        
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_initialization_with_nltk(self):
        """Test validator initialization with NLTK"""
        validator = WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
        
        # Should contain words from NLTK corpus (within length limits)
        self.assertTrue(validator.is_valid_word("HELLO"))
        self.assertTrue(validator.is_valid_word("WORLD"))
        self.assertTrue(validator.is_valid_word("CAT"))  # CAT is valid (3 letters)

    def test_initialization_with_custom_dictionary(self):
        """Test validator initialization with custom dictionary"""
        with patch("builtins.open", mock_open(read_data=self.sample_words)), \
             patch("os.path.exists", return_value=True):
            validator = WordValidator(
                word_repo=self.mock_word_repo,
                use_nltk=False,
                custom_dictionary_path="mock_path"
            )
            
        self.assertTrue(validator.is_valid_word("HELLO"))
        self.assertTrue(validator.is_valid_word("WORLD"))

    def test_nltk_download_handling(self):
        """Test NLTK download handling when corpus not found"""
        # Mock NLTK data finder to raise LookupError
        with patch('nltk.data.find', side_effect=LookupError), \
             patch('nltk.download') as mock_download:
            
            WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
            mock_download.assert_called_once_with('words', quiet=True)

    def test_is_valid_word(self):
        """Test word validation"""
        validator = WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
        
        # Test valid words
        self.assertTrue(validator.is_valid_word("HELLO"))
        self.assertTrue(validator.is_valid_word("WORLD"))
        
        # Test case insensitivity
        self.assertTrue(validator.is_valid_word("hello"))
        self.assertTrue(validator.is_valid_word("World"))
        
        # Test invalid words
        self.assertFalse(validator.is_valid_word("NOTFOUND"))
        self.assertFalse(validator.is_valid_word(""))
        self.assertFalse(validator.is_valid_word("A"))  # Too short
        self.assertFalse(validator.is_valid_word("AB"))  # Too short

    def test_validate_word_with_letters(self):
        """Test word validation with available letters"""
        validator = WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
        letters = ['H', 'E', 'L', 'L', 'O']  # List with duplicates
        
        # Print debug info
        print("\nNLTK words:", validator.nltk_words)
        print("Testing word:", "HELLO")
        print("Available letters:", letters)
        
        # Test valid word with available letters
        result = validator.validate_word_with_letters("HELLO", letters)
        print("Result:", result)
        self.assertTrue(result)
        
        # Test valid word with insufficient letters
        self.assertFalse(validator.validate_word_with_letters("HELP", ['H', 'E', 'L', 'O']))
        
        # Test invalid word with available letters
        self.assertFalse(validator.validate_word_with_letters("NOTFOUND", letters))
        
        # Test case insensitivity
        self.assertTrue(validator.validate_word_with_letters("hello", letters))
        
        # Test empty inputs
        self.assertFalse(validator.validate_word_with_letters("", letters))
        self.assertFalse(validator.validate_word_with_letters("HELLO", []))

    def test_get_word_suggestions(self):
        """Test word suggestion functionality"""
        validator = WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
        
        # Test valid prefix
        suggestions = validator.get_word_suggestions("HE")
        self.assertIn("HELLO", suggestions)
        self.assertIn("HELP", suggestions)
        self.assertIn("HEAP", suggestions)
        
        # Test max suggestions limit
        limited_suggestions = validator.get_word_suggestions("HE", max_suggestions=2)
        self.assertEqual(len(limited_suggestions), 2)
        
        # Test invalid prefix
        self.assertEqual(len(validator.get_word_suggestions("XY")), 0)
        
        # Test empty prefix
        self.assertEqual(len(validator.get_word_suggestions("")), 0)

    def test_get_dictionary_stats(self):
        """Test dictionary statistics"""
        validator = WordValidator(word_repo=self.mock_word_repo, use_nltk=True)
        stats = validator.get_dictionary_stats()
        
        self.assertIn("total_words", stats)
        self.assertIn("max_length", stats)
        self.assertIn("min_length", stats)
        self.assertIn("usage_stats", stats)
        
        # Check stats reflect NLTK words within valid length range
        valid_word_count = len([w for w in self.nltk_words if 3 <= len(w) <= 15])
        self.assertEqual(stats["total_words"], valid_word_count)
        
        # Check usage stats from word repo
        self.assertEqual(stats["usage_stats"]["total_words"], 5)
        self.assertEqual(stats["usage_stats"]["unique_words"], 5)
        self.assertEqual(stats["usage_stats"]["avg_word_length"], 4.6)

    def test_combined_dictionaries(self):
        """Test using both NLTK and custom dictionary"""
        custom_words = "CUSTOM\nWORDS\nNOTINNLTK"
        
        with patch("builtins.open", mock_open(read_data=custom_words)), \
             patch("os.path.exists", return_value=True):
            validator = WordValidator(
                word_repo=self.mock_word_repo,
                use_nltk=True,
                custom_dictionary_path="mock_path"
            )
            
        # Should contain words from both sources
        self.assertTrue(validator.is_valid_word("HELLO"))  # From NLTK
        self.assertTrue(validator.is_valid_word("CUSTOM"))  # From custom dictionary
        self.assertTrue(validator.is_valid_word("WORLD"))  # From NLTK
        self.assertTrue(validator.is_valid_word("NOTINNLTK"))  # From custom dictionary

if __name__ == '__main__':
    unittest.main()