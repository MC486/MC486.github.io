import unittest
import os
import tempfile
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.category_repository import CategoryRepository
from database.repositories.word_repository import WordRepository

class TestCategoryRepository(unittest.TestCase):
    def setUp(self):
        """Set up a temporary database and test repositories."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        self.db_manager.create_tables()
        
        self.category_repo = CategoryRepository(self.db_manager)
        self.word_repo = WordRepository(self.db_manager)
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_get_by_name(self):
        """Test getting a category by name."""
        # Create a test category
        category_data = {
            'name': 'test_category',
            'description': 'Test category description'
        }
        self.category_repo.create(category_data)
        
        # Test exact match
        category = self.category_repo.get_by_name('test_category')
        self.assertIsNotNone(category)
        self.assertEqual(category['name'], 'test_category')
        
        # Test case sensitivity
        category = self.category_repo.get_by_name('TEST_CATEGORY')
        self.assertIsNone(category)
        
        # Test non-existent category
        category = self.category_repo.get_by_name('nonexistent')
        self.assertIsNone(category)
        
    def test_get_categories_with_word_count(self):
        """Test getting categories with word counts."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[0], 'frequency': 20},
            {'word': 'word3', 'category_id': category_ids[1], 'frequency': 30}
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Get categories with counts
        categories = self.category_repo.get_categories_with_word_count()
        self.assertEqual(len(categories), 2)
        
        # Verify counts
        counts = {cat['name']: cat['word_count'] for cat in categories}
        self.assertEqual(counts['cat1'], 2)
        self.assertEqual(counts['cat2'], 1)
        
    def test_get_popular_categories(self):
        """Test getting popular categories."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'},
            {'name': 'cat3', 'description': 'Category 3'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[0], 'frequency': 20},
            {'word': 'word3', 'category_id': category_ids[1], 'frequency': 30},
            {'word': 'word4', 'category_id': category_ids[1], 'frequency': 40},
            {'word': 'word5', 'category_id': category_ids[2], 'frequency': 50}
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Get top 2 categories
        popular = self.category_repo.get_popular_categories(2)
        self.assertEqual(len(popular), 2)
        self.assertEqual(popular[0]['name'], 'cat2')  # Most words
        self.assertEqual(popular[1]['name'], 'cat1')  # Second most
        
    def test_get_category_stats(self):
        """Test getting category statistics."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[0], 'frequency': 20},
            {'word': 'word3', 'category_id': category_ids[1], 'frequency': 30}
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Get stats
        stats = self.category_repo.get_category_stats()
        
        # Verify stats
        self.assertEqual(stats['total_categories'], 2)
        self.assertEqual(stats['categories_with_words'], 2)
        self.assertEqual(len(stats['words_per_category']), 2)
        self.assertEqual(stats['avg_words_per_category'], 1.5)
        
    def test_update_category_words(self):
        """Test updating category for multiple words."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[0], 'frequency': 20}
        ]
        word_ids = []
        for word in words:
            word_id = self.word_repo.create(word)
            word_ids.append(word_id)
            
        # Update category
        self.category_repo.update_category_words(category_ids[1], word_ids)
        
        # Verify updates
        for word_id in word_ids:
            word = self.word_repo.get_by_id(word_id)
            self.assertEqual(word['category_id'], category_ids[1])
            
    def test_delete_category(self):
        """Test deleting a category."""
        # Create a test category
        category_data = {
            'name': 'test_category',
            'description': 'Test category description'
        }
        category_id = self.category_repo.create(category_data)
        
        # Create a test word
        word_data = {
            'word': 'testword',
            'category_id': category_id,
            'frequency': 10
        }
        word_id = self.word_repo.create(word_data)
        
        # Delete the category
        self.category_repo.delete_category(category_id)
        
        # Verify category is deleted
        category = self.category_repo.get_by_id(category_id)
        self.assertIsNone(category)
        
        # Verify word has no category
        word = self.word_repo.get_by_id(word_id)
        self.assertIsNone(word['category_id'])
        
    def test_merge_categories(self):
        """Test merging categories."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[1], 'frequency': 20}
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Merge categories
        self.category_repo.merge_categories(category_ids[0], category_ids[1])
        
        # Verify source category is deleted
        category = self.category_repo.get_by_id(category_ids[0])
        self.assertIsNone(category)
        
        # Verify words are in target category
        words = self.word_repo.get_all()
        for word in words:
            self.assertEqual(word['category_id'], category_ids[1])
            
    def test_get_words_in_category(self):
        """Test getting words in a category."""
        # Create a test category
        category_data = {
            'name': 'test_category',
            'description': 'Test category description'
        }
        category_id = self.category_repo.create(category_data)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_id, 'frequency': 10},
            {'word': 'word2', 'category_id': category_id, 'frequency': 20},
            {'word': 'word3', 'category_id': None, 'frequency': 30}  # No category
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Get words in category
        category_words = self.category_repo.get_words_in_category(category_id)
        self.assertEqual(len(category_words), 2)
        self.assertEqual({w['word'] for w in category_words}, {'word1', 'word2'})
        
    def test_get_categories_by_word_count_range(self):
        """Test getting categories by word count range."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'},
            {'name': 'cat3', 'description': 'Category 3'}
        ]
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Create test words
        words = [
            {'word': 'word1', 'category_id': category_ids[0], 'frequency': 10},
            {'word': 'word2', 'category_id': category_ids[0], 'frequency': 20},
            {'word': 'word3', 'category_id': category_ids[1], 'frequency': 30},
            {'word': 'word4', 'category_id': category_ids[2], 'frequency': 40}
        ]
        
        for word in words:
            self.word_repo.create(word)
            
        # Get categories with 1-2 words
        categories = self.category_repo.get_categories_by_word_count_range(1, 2)
        self.assertEqual(len(categories), 2)
        self.assertEqual({cat['name'] for cat in categories}, {'cat1', 'cat2'})
        
    def test_bulk_create_categories(self):
        """Test creating multiple categories at once."""
        # Create test categories
        categories = [
            {'name': 'cat1', 'description': 'Category 1'},
            {'name': 'cat2', 'description': 'Category 2'},
            {'name': 'cat3', 'description': 'Category 3'}
        ]
        
        category_ids = self.category_repo.bulk_create_categories(categories)
        
        # Verify categories were created
        self.assertEqual(len(category_ids), 3)
        for i, category_id in enumerate(category_ids):
            category = self.category_repo.get_by_id(category_id)
            self.assertIsNotNone(category)
            self.assertEqual(category['name'], categories[i]['name'])

if __name__ == '__main__':
    unittest.main() 