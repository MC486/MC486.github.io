import unittest
import os
import tempfile
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.word_repository import WordRepository

class TestWordRepository(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database and test repository."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.create_tables()
        
        # Create test categories
        self.db_manager.execute("""
            INSERT INTO categories (name, description)
            VALUES (?, ?)
        """, ('test_category', 'Test category description'))
        
        self.category_id = self.db_manager.get_scalar("SELECT last_insert_rowid()")
        
        self.repo = WordRepository(self.db_manager)
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_get_by_word(self):
        """Test getting a word by its exact spelling."""
        # Create a test word
        word_data = {
            'word': 'testword',
            'category_id': self.category_id,
            'frequency': 10
        }
        self.repo.create(word_data)
        
        # Test exact match
        word = self.repo.get_by_word('testword')
        self.assertIsNotNone(word)
        self.assertEqual(word['word'], 'testword')
        
        # Test case sensitivity
        word = self.repo.get_by_word('TESTWORD')
        self.assertIsNone(word)
        
        # Test non-existent word
        word = self.repo.get_by_word('nonexistent')
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
            self.repo.create(word)
            
        # Test getting words from the test category
        category_words = self.repo.get_by_category(self.category_id)
        self.assertEqual(len(category_words), 2)
        self.assertEqual({w['word'] for w in category_words}, {'word1', 'word2'})
        
        # Test getting words from non-existent category
        category_words = self.repo.get_by_category(999)
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
            self.repo.create(word)
            
        # Test range that includes all words
        words = self.repo.get_by_frequency_range(10, 30)
        self.assertEqual(len(words), 3)
        
        # Test range that includes some words
        words = self.repo.get_by_frequency_range(15, 25)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]['word'], 'word2')
        
        # Test range with no words
        words = self.repo.get_by_frequency_range(100, 200)
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
            self.repo.create(word)
            
        # Test getting top 2 words
        top_words = self.repo.get_top_words(2)
        self.assertEqual(len(top_words), 2)
        self.assertEqual(top_words[0]['word'], 'word2')  # Highest frequency
        self.assertEqual(top_words[1]['word'], 'word3')  # Second highest
        
        # Test getting all words
        top_words = self.repo.get_top_words()
        self.assertEqual(len(top_words), 3)
        
    def test_increment_frequency(self):
        """Test incrementing word frequency."""
        # Create a test word
        word_data = {
            'word': 'testword',
            'category_id': self.category_id,
            'frequency': 10
        }
        self.repo.create(word_data)
        
        # Increment frequency
        self.repo.increment_frequency('testword')
        
        # Verify the increment
        word = self.repo.get_by_word('testword')
        self.assertEqual(word['frequency'], 11)
        
        # Test incrementing non-existent word
        self.repo.increment_frequency('nonexistent')
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
            self.repo.create(word)
            
        # Get statistics
        stats = self.repo.get_word_stats()
        
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
            self.repo.create(word)
            
        # Test prefix search
        results = self.repo.search_words('test%')
        self.assertEqual(len(results), 2)
        self.assertEqual({w['word'] for w in results}, {'test', 'testing'})
        
        # Test exact match
        results = self.repo.search_words('test')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'test')
        
        # Test no matches
        results = self.repo.search_words('nonexistent%')
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
            self.repo.create(word)
            
        # Test length 2
        results = self.repo.get_words_by_length(2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'ab')
        
        # Test length 3
        results = self.repo.get_words_by_length(3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['word'], 'abc')
        
        # Test length with no matches
        results = self.repo.get_words_by_length(4)
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
            self.repo.create(word)
            
        # Update frequencies
        new_frequencies = {
            'word1': 100,
            'word2': 200,
            'word3': 300
        }
        self.repo.bulk_update_frequency(new_frequencies)
        
        # Verify updates
        for word, freq in new_frequencies.items():
            record = self.repo.get_by_word(word)
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
            self.repo.create(word)
            
        # Get uncategorized words
        results = self.repo.get_words_without_category()
        self.assertEqual(len(results), 2)
        self.assertEqual({w['word'] for w in results}, {'word2', 'word3'})

if __name__ == '__main__':
    unittest.main() 