import os
import tempfile
import sys
from pathlib import Path
import pytest
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from database.repositories.category_repository import CategoryRepository
from database.repositories.word_repository import WordRepository

@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    yield temp_db
    # Cleanup
    temp_db.close()
    os.unlink(temp_db.name)

@pytest.fixture
def db_manager(temp_db):
    """Create a test database manager."""
    return DatabaseManager(temp_db.name)

@pytest.fixture
def category_repo(db_manager):
    """Create a test category repository."""
    return CategoryRepository(db_manager)

@pytest.fixture
def word_repo(db_manager):
    """Create a WordRepository instance."""
    return WordRepository(db_manager)

def test_get_by_name(category_repo):
    """Test retrieving a category by name."""
    created = category_repo.create_category("test_category", "Test description")
    retrieved = category_repo.get_by_name("test_category")
    assert created['id'] == retrieved['id']
    assert retrieved['name'] == "test_category"
    assert retrieved['description'] == "Test description"
    assert retrieved['created_at'] is not None
    assert retrieved['updated_at'] is not None

def test_get_popular_categories(category_repo, word_repo):
    """Test getting popular categories."""
    # Create test categories
    categories = [
        {'name': 'cat1', 'description': 'Category 1'},
        {'name': 'cat2', 'description': 'Category 2'},
        {'name': 'cat3', 'description': 'Category 3'}
    ]
    category_ids = []
    for cat in categories:
        category = category_repo.create_category(cat['name'], cat['description'])
        category_ids.append(category['id'])
    
    # Add words to categories with different frequencies
    # cat1: 1 word with frequency 10
    # cat2: 2 words with frequency 10 each
    # cat3: 3 words with frequency 10 each
    for i, cat_id in enumerate(category_ids):
        for j in range(i + 1):
            word_repo.create({
                'word': f'word{i}_{j}',
                'category_id': cat_id,
                'frequency': 10,
                'allowed': True
            })
    
    # Get popular categories
    popular = category_repo.get_popular_categories(limit=2)
    assert len(popular) == 2
    
    # Verify order: cat3 (3 words) > cat2 (2 words)
    assert popular[0]['name'] == 'cat3'  # Most words
    assert popular[0]['word_count'] == 3
    assert popular[0]['total_frequency'] == 30
    
    assert popular[1]['name'] == 'cat2'  # Second most words
    assert popular[1]['word_count'] == 2
    assert popular[1]['total_frequency'] == 20

def test_get_category_stats(category_repo, word_repo):
    """Test getting category statistics."""
    # Create test categories
    categories = [
        {'name': 'cat1', 'description': 'Category 1'},
        {'name': 'cat2', 'description': 'Category 2'}
    ]
    category_ids = []
    for cat in categories:
        category = category_repo.create_category(cat['name'], cat['description'])
        category_ids.append(category['id'])

    # Add words to categories
    # Each category gets 3 words: word0 (3 chars), word1 (4 chars), word2 (5 chars)
    for cat_id in category_ids:
        for i in range(3):
            word_repo.create({
                'word': f'word{i}_{cat_id}',  # Make word names unique by including category ID
                'category_id': cat_id,
                'frequency': 10,
                'allowed': True
            })

    # Get stats for each category
    for cat_id in category_ids:
        stats = category_repo.get_category_stats(cat_id)
        assert stats is not None
        assert stats['word_count'] == 3
        assert stats['total_frequency'] == 30
        assert stats['average_frequency'] == 10
        assert stats['allowed_word_count'] == 3
        assert stats['disallowed_word_count'] == 0
        # Word lengths are now longer due to category ID suffix
        assert stats['min_word_length'] == len(f'word0_{cat_id}')
        assert stats['max_word_length'] == len(f'word2_{cat_id}')
        assert stats['average_word_length'] == (len(f'word0_{cat_id}') + len(f'word1_{cat_id}') + len(f'word2_{cat_id}')) / 3

def test_update_category_words(category_repo, word_repo):
    """Test updating words in a category."""
    # Create a test category
    category = category_repo.create_category('test', 'Test category')
    
    # Create some words without category
    words = [
        {'word': 'word1', 'frequency': 10, 'allowed': True},
        {'word': 'word2', 'frequency': 20, 'allowed': True}
    ]
    word_ids = [word_repo.create(word) for word in words]
    
    # Update category words
    category_repo.update_category_words(category['id'], word_ids)
    
    # Verify update
    category_words = word_repo.get_by_category(category['id'])
    assert len(category_words) == len(words)
    word_texts = [word['word'] for word in category_words]
    assert 'word1' in word_texts
    assert 'word2' in word_texts

def test_delete_category(category_repo):
    """Test deleting a category."""
    # Create a test category
    category = category_repo.create_category('test', 'Test category')
    category_id = category['id']
    
    # Delete the category
    deleted = category_repo.delete_category(category_id)
    assert deleted is True
    
    # Verify the category is gone
    assert category_repo.get_category_by_id(category_id) is None
    assert category_repo.get_category_by_name('test') is None
    
    # Try to delete non-existent category
    non_existent = category_repo.delete_category(999)
    assert non_existent is False
    
    # Create multiple categories and delete one
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Delete one category
    deleted = category_repo.delete_category(categories[1]['id'])
    assert deleted is True
    
    # Verify the correct category was deleted
    assert category_repo.get_category_by_id(categories[0]['id']) is not None
    assert category_repo.get_category_by_id(categories[1]['id']) is None
    assert category_repo.get_category_by_id(categories[2]['id']) is not None

def test_merge_categories(category_repo, word_repo):
    """Test merging categories."""
    # Create source and target categories
    source = category_repo.create_category('source', 'Source category')
    target = category_repo.create_category('target', 'Target category')
    
    # Add words to source category
    source_words = [
        word_repo.create({
            'word': f'word{i}',
            'category_id': source['id'],
            'frequency': 10,
            'allowed': True
        }) for i in range(3)
    ]
    
    # Add words to target category
    target_words = [
        word_repo.create({
            'word': f'target_word{i}',
            'category_id': target['id'],
            'frequency': 10,
            'allowed': True
        }) for i in range(2)
    ]
    
    # Merge categories
    merged = category_repo.merge_categories(source['id'], target['id'])
    assert merged is True
    
    # Verify source category is deleted
    assert category_repo.get_category_by_id(source['id']) is None
    
    # Verify words are moved to target category
    target_category_words = word_repo.get_by_category(target['id'])
    assert len(target_category_words) == len(source_words) + len(target_words)
    
    # Verify word frequencies are preserved
    for word in target_category_words:
        assert word['frequency'] == 10

def test_get_words_in_category(category_repo, word_repo):
    """Test getting words in a category."""
    # Create a test category
    category = category_repo.create_category('test', 'Test category')
    
    # Add words to category
    words = [
        word_repo.create({
            'word': f'word{i}',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        }) for i in range(5)
    ]
    
    # Get words in category
    category_words = category_repo.get_words_in_category(category['id'])
    assert len(category_words) == len(words)
    
    # Verify word properties
    for word in category_words:
        assert word['category_id'] == category['id']
        assert word['frequency'] == 10
        assert word['allowed'] is True

def test_get_categories_by_word_count_range(category_repo, word_repo):
    """Test getting categories by word count range."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words to categories with different counts
    # cat1: 1 word
    # cat2: 2 words
    # cat3: 3 words
    for i, category in enumerate(categories):
        for j in range(i + 1):
            word_repo.create({
                'word': f'word{i}_{j}',
                'category_id': category['id'],
                'frequency': 10,
                'allowed': True
            })
    
    # Test range 1-2 words
    result = category_repo.get_categories_by_word_count_range(1, 2)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 2-3 words
    result = category_repo.get_categories_by_word_count_range(2, 3)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_bulk_create_categories(category_repo):
    """Test bulk creating categories."""
    # Create multiple categories at once
    categories = [
        {'name': 'cat1', 'description': 'Category 1'},
        {'name': 'cat2', 'description': 'Category 2'},
        {'name': 'cat3', 'description': 'Category 3'}
    ]
    
    created = category_repo.bulk_create_categories(categories)
    assert len(created) == len(categories)
    
    # Verify each category was created
    for category in created:
        retrieved = category_repo.get_category_by_id(category['id'])
        assert retrieved is not None
        assert retrieved['name'] in ['cat1', 'cat2', 'cat3']
        assert retrieved['description'] in ['Category 1', 'Category 2', 'Category 3']

def test_category_operations(category_repo):
    """Test basic category operations."""
    # Create a category
    category = category_repo.create_category('test', 'Test category')
    assert category['name'] == 'test'
    assert category['description'] == 'Test category'
    assert category['id'] is not None
    
    # Get category by ID
    retrieved = category_repo.get_category_by_id(category['id'])
    assert retrieved['id'] == category['id']
    assert retrieved['name'] == 'test'
    assert retrieved['description'] == 'Test category'
    
    # Update category
    updated = category_repo.update_category(category['id'], 'updated', 'Updated category')
    assert updated['id'] == category['id']
    assert updated['name'] == 'updated'
    assert updated['description'] == 'Updated category'
    
    # Delete category
    deleted = category_repo.delete_category(category['id'])
    assert deleted is True
    assert category_repo.get_category_by_id(category['id']) is None

def test_cleanup_old_entries(category_repo, word_repo):
    """Test cleaning up old entries."""
    # Create a test category
    category = category_repo.create_category('test', 'Test category')

    # Add some words
    word_ids = []
    for i in range(3):
        word_id = word_repo.create({
            'word': f'word{i}',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
        word_ids.append(word_id)

    # Set some words as old
    old_date = datetime.now() - timedelta(days=31)
    for word_id in word_ids[:2]:
        word_repo.update(word_id, {'updated_at': old_date})

    # Clean up old entries
    deleted_count = category_repo.cleanup_old_entries(30)
    assert deleted_count == 1  # Category should be deleted since it has old words

def test_get_size_bytes(category_repo, word_repo):
    """Test getting database size in bytes."""
    # Create a test category
    category = category_repo.create_category('test', 'Test category')
    
    # Add some words
    words = [
        word_repo.create({
            'word': f'word{i}',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        }) for i in range(3)
    ]
    
    # Get size
    size = category_repo.get_size_bytes()
    assert size > 0

def test_get_entry_count(category_repo):
    """Test getting entry count."""
    # Create some categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Get count
    count = category_repo.get_entry_count()
    assert count == len(categories)

def test_get_categories_with_word_count(category_repo, word_repo):
    """Test getting categories with word count."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words to categories with different counts
    # cat1: 1 word
    # cat2: 2 words
    # cat3: 3 words
    for i, category in enumerate(categories):
        for j in range(i + 1):
            word_repo.create({
                'word': f'word{i}_{j}',
                'category_id': category['id'],
                'frequency': 10,
                'allowed': True
            })
    
    # Get categories with word count
    result = category_repo.get_categories_with_word_count()
    assert len(result) == len(categories)
    
    # Verify word counts
    for category in result:
        if category['name'] == 'cat1':
            assert category['word_count'] == 1
        elif category['name'] == 'cat2':
            assert category['word_count'] == 2
        elif category['name'] == 'cat3':
            assert category['word_count'] == 3

def test_get_categories(category_repo):
    """Test getting all categories."""
    # Create some categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Get all categories
    result = category_repo.get_categories()
    assert len(result) == len(categories)
    
    # Verify each category exists
    for category in categories:
        assert any(c['id'] == category['id'] for c in result)
        assert any(c['name'] == category['name'] for c in result)
        assert any(c['description'] == category['description'] for c in result)

def test_get_category_by_id(category_repo):
    """Test getting a category by ID."""
    # Create a category
    category = category_repo.create_category('test', 'Test category')
    
    # Get category by ID
    retrieved = category_repo.get_category_by_id(category['id'])
    assert retrieved['id'] == category['id']
    assert retrieved['name'] == 'test'
    assert retrieved['description'] == 'Test category'
    
    # Try to get non-existent category
    assert category_repo.get_category_by_id(999) is None

def test_get_category_by_name(category_repo):
    """Test getting a category by name."""
    # Create a category
    category = category_repo.create_category('test', 'Test category')
    
    # Get category by name
    retrieved = category_repo.get_category_by_name('test')
    assert retrieved['id'] == category['id']
    assert retrieved['name'] == 'test'
    assert retrieved['description'] == 'Test category'
    
    # Try to get non-existent category
    assert category_repo.get_category_by_name('nonexistent') is None

def test_update_category(category_repo):
    """Test updating a category."""
    # Create a category
    category = category_repo.create_category('test', 'Test category')
    
    # Update category
    updated = category_repo.update_category(category['id'], 'updated', 'Updated category')
    assert updated['id'] == category['id']
    assert updated['name'] == 'updated'
    assert updated['description'] == 'Updated category'
    
    # Verify update
    retrieved = category_repo.get_category_by_id(category['id'])
    assert retrieved['name'] == 'updated'
    assert retrieved['description'] == 'Updated category'
    
    # Try to update non-existent category
    assert category_repo.update_category(999, 'nonexistent', 'Nonexistent category') is None

def test_get_categories_by_frequency_range(category_repo, word_repo):
    """Test getting categories by word frequency range."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different frequencies
    # cat1: words with frequency 5
    # cat2: words with frequency 10
    # cat3: words with frequency 15
    for i, category in enumerate(categories):
        word_repo.create({
            'word': f'word{i}',
            'category_id': category['id'],
            'frequency': (i + 1) * 5,
            'allowed': True
        })
    
    # Test range 5-10
    result = category_repo.get_categories_by_frequency_range(5, 10)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 10-15
    result = category_repo.get_categories_by_frequency_range(10, 15)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_allowed_status(category_repo, word_repo):
    """Test getting categories by word allowed status."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different allowed statuses
    # cat1: all words allowed
    # cat2: some words allowed
    # cat3: no words allowed
    for i, category in enumerate(categories):
        word_repo.create({
            'word': f'word{i}_1',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
        word_repo.create({
            'word': f'word{i}_2',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': i != 2
        })
    
    # Test getting categories with all allowed words
    result = category_repo.get_categories_by_allowed_status(True)
    assert len(result) == 1
    assert result[0]['name'] == 'cat1'
    
    # Test getting categories with some allowed words
    result = category_repo.get_categories_by_allowed_status(False)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_search_term(category_repo):
    """Test getting categories by search term."""
    # Create test categories
    categories = [
        category_repo.create_category('test1', 'Category 1'),
        category_repo.create_category('test2', 'Category 2'),
        category_repo.create_category('other', 'Other category')
    ]
    
    # Search for 'test'
    result = category_repo.get_categories_by_search_term('test')
    assert len(result) == 2
    assert any(c['name'] == 'test1' for c in result)
    assert any(c['name'] == 'test2' for c in result)
    
    # Search for 'category'
    result = category_repo.get_categories_by_search_term('category')
    assert len(result) == 3
    
    # Search for non-existent term
    result = category_repo.get_categories_by_search_term('nonexistent')
    assert len(result) == 0

def test_get_categories_by_word_length(category_repo, word_repo):
    """Test getting categories by word length."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different lengths
    # cat1: 3-letter words
    # cat2: 4-letter words
    # cat3: 5-letter words
    for i, category in enumerate(categories):
        word_repo.create({
            'word': 'a' * (i + 3),
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test range 3-4 letters
    result = category_repo.get_categories_by_word_length(3, 4)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 4-5 letters
    result = category_repo.get_categories_by_word_length(4, 5)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_word_contains(category_repo, word_repo):
    """Test getting categories by word contains."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words containing different substrings
    # cat1: words containing 'test'
    # cat2: words containing 'word'
    # cat3: words containing 'other'
    for i, category in enumerate(categories):
        word_repo.create({
            'word': f'test{i}' if i == 0 else f'word{i}' if i == 1 else f'other{i}',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test getting categories with words containing 'test'
    result = category_repo.get_categories_by_word_contains('test')
    assert len(result) == 1
    assert result[0]['name'] == 'cat1'
    
    # Test getting categories with words containing 'word'
    result = category_repo.get_categories_by_word_contains('word')
    assert len(result) == 1
    assert result[0]['name'] == 'cat2'
    
    # Test getting categories with words containing 'other'
    result = category_repo.get_categories_by_word_contains('other')
    assert len(result) == 1
    assert result[0]['name'] == 'cat3'

def test_get_categories_by_word_regex(category_repo, word_repo):
    """Test getting categories by word regex."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words matching different patterns
    # cat1: words starting with 'test'
    # cat2: words ending with 'word'
    # cat3: words containing 'other'
    for i, category in enumerate(categories):
        word_repo.create({
            'word': f'test{i}' if i == 0 else f'{i}word' if i == 1 else f'other{i}',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test getting categories with words matching '^test'
    result = category_repo.get_categories_by_word_regex('^test')
    assert len(result) == 1
    assert result[0]['name'] == 'cat1'
    
    # Test getting categories with words matching 'word$'
    result = category_repo.get_categories_by_word_regex('word$')
    assert len(result) == 1
    assert result[0]['name'] == 'cat2'
    
    # Test getting categories with words matching 'other'
    result = category_repo.get_categories_by_word_regex('other')
    assert len(result) == 1
    assert result[0]['name'] == 'cat3'

def test_get_categories_by_word_palindrome(category_repo, word_repo):
    """Test getting categories by word palindrome."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different palindrome properties
    # cat1: palindrome words
    # cat2: non-palindrome words
    # cat3: mix of both
    for i, category in enumerate(categories):
        word_repo.create({
            'word': 'racecar' if i == 0 else 'test' if i == 1 else 'level' if i == 2 else 'word',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test getting categories with palindrome words
    result = category_repo.get_categories_by_word_palindrome(True)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat3' for c in result)
    
    # Test getting categories with non-palindrome words
    result = category_repo.get_categories_by_word_palindrome(False)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_word_scrabble_score(category_repo, word_repo):
    """Test getting categories by word Scrabble score."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different Scrabble scores
    # cat1: words with score 5
    # cat2: words with score 10
    # cat3: words with score 15
    for i, category in enumerate(categories):
        word_repo.create({
            'word': 'cat' if i == 0 else 'word' if i == 1 else 'scrabble',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test range 5-10 points
    result = category_repo.get_categories_by_word_scrabble_score(5, 10)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 10-15 points
    result = category_repo.get_categories_by_word_scrabble_score(10, 15)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_word_vowel_count(category_repo, word_repo):
    """Test getting categories by word vowel count."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different vowel counts
    # cat1: words with 1 vowel
    # cat2: words with 2 vowels
    # cat3: words with 3 vowels
    for i, category in enumerate(categories):
        word_repo.create({
            'word': 'cat' if i == 0 else 'word' if i == 1 else 'scrabble',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test range 1-2 vowels
    result = category_repo.get_categories_by_word_vowel_count(1, 2)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 2-3 vowels
    result = category_repo.get_categories_by_word_vowel_count(2, 3)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)

def test_get_categories_by_word_consonant_count(category_repo, word_repo):
    """Test getting categories by word consonant count."""
    # Create test categories
    categories = [
        category_repo.create_category('cat1', 'Category 1'),
        category_repo.create_category('cat2', 'Category 2'),
        category_repo.create_category('cat3', 'Category 3')
    ]
    
    # Add words with different consonant counts
    # cat1: words with 2 consonants
    # cat2: words with 3 consonants
    # cat3: words with 4 consonants
    for i, category in enumerate(categories):
        word_repo.create({
            'word': 'cat' if i == 0 else 'word' if i == 1 else 'scrabble',
            'category_id': category['id'],
            'frequency': 10,
            'allowed': True
        })
    
    # Test range 2-3 consonants
    result = category_repo.get_categories_by_word_consonant_count(2, 3)
    assert len(result) == 2
    assert any(c['name'] == 'cat1' for c in result)
    assert any(c['name'] == 'cat2' for c in result)
    
    # Test range 3-4 consonants
    result = category_repo.get_categories_by_word_consonant_count(3, 4)
    assert len(result) == 2
    assert any(c['name'] == 'cat2' for c in result)
    assert any(c['name'] == 'cat3' for c in result)