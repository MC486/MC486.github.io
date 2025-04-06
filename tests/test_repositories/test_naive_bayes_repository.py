import pytest
from database.repositories.naive_bayes_repository import NaiveBayesRepository
from database.manager import DatabaseManager

@pytest.fixture
def db_manager():
    manager = DatabaseManager(':memory:')
    manager.initialize_database()
    return manager

@pytest.fixture
def repository(db_manager):
    return NaiveBayesRepository(db_manager)

def test_record_word_probability(repository):
    # Test recording word probability
    repository.record_word_probability('test', 0.8)
    repository.record_word_probability('test', 0.9, 'prefix')
    
    # Verify probabilities were recorded
    assert repository.get_word_probability('test') == 0.8
    assert repository.get_word_probability('test', 'prefix') == 0.9
    
def test_get_pattern_probabilities(repository):
    # Record multiple words with same pattern
    repository.record_word_probability('test1', 0.7, 'prefix')
    repository.record_word_probability('test2', 0.8, 'prefix')
    repository.record_word_probability('test3', 0.6, 'suffix')
    
    # Get probabilities for prefix pattern
    probs = repository.get_pattern_probabilities('prefix')
    assert len(probs) == 2
    assert probs['test1'] == 0.7
    assert probs['test2'] == 0.8
    
def test_get_total_observations(repository):
    # Record multiple words
    repository.record_word_probability('test1', 0.7)
    repository.record_word_probability('test2', 0.8)
    repository.record_word_probability('test1', 0.9)  # Update existing word
    
    # Verify total observations
    assert repository.get_total_observations() == 3
    
def test_get_word_stats(repository):
    # Record word with multiple patterns
    repository.record_word_probability('test', 0.8)
    repository.record_word_probability('test', 0.7, 'prefix')
    repository.record_word_probability('test', 0.6, 'suffix')
    
    # Get word statistics
    stats = repository.get_word_stats('test')
    assert stats['total_probability'] == 0.8
    assert stats['pattern_probabilities']['prefix'] == 0.7
    assert stats['pattern_probabilities']['suffix'] == 0.6
    assert stats['visit_count'] == 3
    
def test_cleanup_old_entries(repository):
    # Record some words
    repository.record_word_probability('test1', 0.8)
    repository.record_word_probability('test2', 0.7)
    
    # Cleanup entries (should remove none since they're new)
    removed = repository.cleanup_old_entries(1)
    assert removed == 0
    
def test_get_learning_stats(repository):
    # Record words with different patterns
    repository.record_word_probability('test1', 0.8)
    repository.record_word_probability('test2', 0.7, 'prefix')
    repository.record_word_probability('test3', 0.6, 'prefix')
    repository.record_word_probability('test4', 0.5, 'suffix')
    
    # Get learning statistics
    stats = repository.get_learning_stats()
    assert stats['total_words'] == 4
    assert stats['total_patterns'] == 2
    assert 0.5 <= stats['average_probability'] <= 0.8
    assert stats['most_common_pattern'] == 'prefix' 