import pytest
from database.repositories.mcts_repository import MCTSRepository
from database.manager import DatabaseManager

@pytest.fixture
def db_manager():
    manager = DatabaseManager(':memory:')
    manager.initialize_database()
    return manager

@pytest.fixture
def repository(db_manager):
    return MCTSRepository(db_manager)

def test_record_simulation(repository):
    # Test recording simulation result
    repository.record_simulation('state1', 'action1', 0.8)
    repository.record_simulation('state1', 'action1', 0.9)  # Update existing
    
    # Verify statistics
    stats = repository.get_state_action_stats('state1', 'action1')
    assert stats['reward'] == 0.85  # Average of 0.8 and 0.9
    assert stats['visit_count'] == 2
    
def test_get_state_actions(repository):
    # Record multiple actions for same state
    repository.record_simulation('state1', 'action1', 0.8)
    repository.record_simulation('state1', 'action2', 0.7)
    repository.record_simulation('state2', 'action1', 0.6)
    
    # Get actions for state1
    actions = repository.get_state_actions('state1')
    assert len(actions) == 2
    assert actions['action1']['reward'] == 0.8
    assert actions['action2']['reward'] == 0.7
    
def test_get_best_action(repository):
    # Record multiple actions for same state
    repository.record_simulation('state1', 'action1', 0.8)
    repository.record_simulation('state1', 'action2', 0.9)
    repository.record_simulation('state1', 'action3', 0.7)
    
    # Get best action
    best_action = repository.get_best_action('state1')
    assert best_action == 'action2'
    
def test_cleanup_old_entries(repository):
    # Record some simulations
    repository.record_simulation('state1', 'action1', 0.8)
    repository.record_simulation('state2', 'action1', 0.7)
    
    # Cleanup entries (should remove none since they're new)
    removed = repository.cleanup_old_entries(1)
    assert removed == 0
    
def test_get_learning_stats(repository):
    # Record multiple simulations
    repository.record_simulation('state1', 'action1', 0.8)
    repository.record_simulation('state1', 'action2', 0.7)
    repository.record_simulation('state2', 'action1', 0.6)
    repository.record_simulation('state2', 'action2', 0.5)
    
    # Get learning statistics
    stats = repository.get_learning_stats()
    assert stats['total_states'] == 2
    assert stats['total_actions'] == 2
    assert 0.5 <= stats['average_reward'] <= 0.8
    assert stats['most_visited_state'] in ['state1', 'state2'] 