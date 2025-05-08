import unittest
import os
import tempfile
import sys
from pathlib import Path
import sqlite3
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.word_repository import WordRepository

class TestWordRepository(unittest.TestCase):
    """Test the WordRepository class."""
    
    def setUp(self):
        """Setup test environment before each test"""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        
        # Initialize database manager with the temporary file
        self.db_manager = DatabaseManager(self.db_path)
        
        # Create WordRepository instance with required parameters
        self.repository = WordRepository(
            db_manager=self.db_manager
        )
        
        # Create a test category
        self.category_id = self.db_manager.execute_query("""
            INSERT INTO categories (name, description)
            VALUES (?, ?)
        """, ("test_category", "Test category"))[0]['id']
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_create_word(self):
        """Test creating a new word."""
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        
        id = self.repository.create(data)
        self.assertIsInstance(id, int)
        
        # Verify the word was created
        word = self.repository.get_by_id(id)
        self.assertIsNotNone(word)
        self.assertEqual(word['word'], 'test')
        self.assertEqual(word['frequency'], 10)
        
    def test_record_word_usage(self):
        """Test recording word usage."""
        # Test adding new word
        self.repository.record_word_usage("TEST", True)
        word = self.repository.get_by_word("TEST")
        self.assertIsNotNone(word)
        self.assertEqual(word["word"], "TEST")
        self.assertEqual(word["allowed"], True)
        self.assertEqual(word["num_played"], 1)
        
        # Test updating existing word
        self.repository.record_word_usage("TEST", False)
        word = self.repository.get_by_word("TEST")
        self.assertEqual(word["allowed"], False)
        self.assertEqual(word["num_played"], 2)
        
    def test_get_word_stats(self):
        """Test getting word usage statistics."""
        # Add some test words
        self.repository.record_word_usage("TEST1", True)
        self.repository.record_word_usage("TEST2", True)
        self.repository.record_word_usage("TEST3", False)
        self.repository.record_word_usage("TEST1", True)  # Increment count
        
        stats = self.repository.get_word_stats()
        self.assertEqual(stats["total_words"], 3)
        self.assertEqual(stats["valid_words"], 2)
        self.assertEqual(stats["invalid_words"], 1)
        self.assertEqual(stats["total_plays"], 4)
        self.assertAlmostEqual(stats["avg_plays_per_word"], 4/3)

    def test_get_by_word(self):
        """Test getting a word by its exact spelling."""
        # Create a test word
        word_data = {
            'word': 'testword',
            'category_id': self.category_id,
            'frequency': 10
        }
        self.repository.create(word_data)
        
        # Test exact match
        word = self.repository.get_by_word('testword')
        self.assertIsNotNone(word)
        self.assertEqual(word['word'], 'testword')
        
        # Test case sensitivity
        word = self.repository.get_by_word('TESTWORD')
        self.assertIsNone(word)
        
        # Test non-existent word
        word = self.repository.get_by_word('nonexistent')
        self.assertIsNone(word)
        
    def test_get_by_category(self):
        """Test getting words by category."""
        # Create test words in different categories
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'word3', 'category_id': 999, 'frequency': 30}  # Different category
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Test getting words from the test category
        category_words = self.repository.get_by_category(self.category_id)
        self.assertEqual(len(category_words), 2)
        self.assertEqual({w['word'] for w in category_words}, {'word1', 'word2'})
        
        # Test getting words from non-existent category
        category_words = self.repository.get_by_category(999)
        self.assertEqual(len(category_words), 1)
        self.assertEqual(category_words[0]['word'], 'word3')
        
    def test_get_by_frequency_range(self):
        """Test getting words within a frequency range."""
        # Create test words with different frequencies
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'word3', 'category_id': self.category_id, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Test range that includes all words
        words = self.repository.get_by_frequency_range(10, 30)
        self.assertEqual(len(words), 3)
        
        # Test range that includes some words
        words = self.repository.get_by_frequency_range(15, 25)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]['word'], 'word2')
        
        # Test range with no words
        words = self.repository.get_by_frequency_range(100, 200)
        self.assertEqual(len(words), 0)
        
    def test_get_top_words(self):
        """Test getting the most frequent words."""
        # Create test words with different frequencies
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': self.category_id, 'frequency': 30},
            {'word': 'word3', 'category_id': self.category_id, 'frequency': 20}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Test getting top 2 words
        top_words = self.repository.get_top_words(2)
        self.assertEqual(len(top_words), 2)
        self.assertEqual(top_words[0]['word'], 'word2')  # Highest frequency
        self.assertEqual(top_words[1]['word'], 'word3')  # Second highest
        
        # Test getting all words
        top_words = self.repository.get_top_words()
        self.assertEqual(len(top_words), 3)
        
    def test_increment_frequency(self):
        """Test incrementing word frequency."""
        # Create a test word
        word_data = {
            'word': 'testword',
            'category_id': self.category_id,
            'frequency': 10
        }
        self.repository.create(word_data)
        
        # Increment frequency
        self.repository.increment_frequency('testword')
        
        # Verify the increment
        word = self.repository.get_by_word('testword')
        self.assertEqual(word['frequency'], 11)
        
        # Test incrementing non-existent word
        self.repository.increment_frequency('nonexistent')
        # Should not raise an error
        
    def test_get_word_stats(self):
        """Test getting word statistics."""
        # Create test words with different frequencies and categories
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'word3', 'category_id': self.category_id, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Get statistics
        stats = self.repository.get_word_stats()
        
        # Verify statistics
        self.assertEqual(stats['total_words'], 3)
        self.assertEqual(stats['avg_frequency'], 20.0)
        self.assertEqual(stats['max_frequency'], 30)
        self.assertEqual(stats['min_frequency'], 10)
        self.assertEqual(len(stats['words_by_category']), 1)
        self.assertEqual(stats['words_by_category'][0]['count'], 3)

    def test_search_words(self):
        """Test searching for words using patterns."""
        # Create test words
        words = [
            {'word': 'test', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'testing', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'other', 'category_id': self.category_id, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Test prefix search
        results = self.repository.search_words('test%')
        self.assertEqual(len(results), 2)
        self.assertEqual({w['word'] for w in results}, {'test', 'testing'})
        
        # Test exact match
        results = self.repository.search_words('test')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'test')
        
        # Test no matches
        results = self.repository.search_words('nonexistent%')
        self.assertEqual(len(results), 0)
        
    def test_get_words_by_length(self):
        """Test getting words by length."""
        # Create test words of different lengths
        words = [
            {'word': 'a', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'ab', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'abc', 'category_id': self.category_id, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Test length 2
        results = self.repository.get_words_by_length(2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'ab')
        
        # Test length 3
        results = self.repository.get_words_by_length(3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'abc')
        
        # Test length with no matches
        results = self.repository.get_words_by_length(4)
        self.assertEqual(len(results), 0)
        
    def test_bulk_update_frequency(self):
        """Test updating frequencies for multiple words."""
        # Create test words
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': self.category_id, 'frequency': 20},
            {'word': 'word3', 'category_id': self.category_id, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Update frequencies
        new_frequencies = {
            'word1': 100,
            'word2': 200,
            'word3': 300
        }
        self.repository.bulk_update_frequency(new_frequencies)
        
        # Verify updates
        for word, freq in new_frequencies.items():
            record = self.repository.get_by_word(word)
            self.assertEqual(record['frequency'], freq)
            
    def test_get_words_without_category(self):
        """Test getting words without a category."""
        # Create test words with and without categories
        words = [
            {'word': 'word1', 'category_id': self.category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': None, 'frequency': 20},
            {'word': 'word3', 'category_id': None, 'frequency': 30}
        ]
        
        for word in words:
            self.repository.create(word)
            
        # Get uncategorized words
        results = self.repository.get_words_without_category()
        self.assertEqual(len(results), 2)
        self.assertEqual({w['word'] for w in results}, {'word2', 'word3'})

if __name__ == '__main__':
    unittest.main() 