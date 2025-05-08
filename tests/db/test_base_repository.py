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
from database.repositories.base_repository import BaseRepository

class TestBaseRepository(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.db_path = self.temp_db.name
        
        # Initialize database manager with the temporary file
        self.db_manager = DatabaseManager(self.db_path)
        
        # Create BaseRepository instance with required parameters
        self.repository = BaseRepository(
            db_manager=self.db_manager,
            table_name="test_table"
        )
        
    def tearDown(self):
        """Clean up the temporary database."""
        self.temp_db.close()
        os.unlink(self.db_path)
        
    def test_create(self):
        """Test creating a new record."""
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        
        id = self.repository.create(data)
        self.assertIsInstance(id, int)
        
        # Verify the record was created
        record = self.repository.get_by_id(id)
        self.assertIsNotNone(record)
        self.assertEqual(record['word'], 'test')
        self.assertEqual(record['frequency'], 10)
        
    def test_get_by_id(self):
        """Test retrieving a record by ID."""
        # Create a test record
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        id = self.repository.create(data)
        
        # Retrieve it
        record = self.repository.get_by_id(id)
        self.assertIsNotNone(record)
        self.assertEqual(record['word'], 'test')
        
        # Test non-existent ID
        record = self.repository.get_by_id(999)
        self.assertIsNone(record)
        
    def test_get_all(self):
        """Test retrieving all records."""
        # Create multiple records
        records = [
            {'word': 'test1', 'category_id': 1, 'frequency': 10, 'allowed': True},
            {'word': 'test2', 'category_id': 2, 'frequency': 20, 'allowed': True},
            {'word': 'test3', 'category_id': 3, 'frequency': 30, 'allowed': True}
        ]
    
        for record in records:
            self.repository.create(record)
            
        # Retrieve all
        all_records = self.repository.get_all()
        self.assertEqual(len(all_records), 3)
        
    def test_update(self):
        """Test updating a record."""
        # Create a test record
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        id = self.repository.create(data)
        
        # Update it
        update_data = {'frequency': 20}
        success = self.repository.update(id, update_data)
        self.assertTrue(success)
        
        # Verify the update
        record = self.repository.get_by_id(id)
        self.assertEqual(record['frequency'], 20)
        
        # Test updating non-existent record
        success = self.repository.update(999, update_data)
        self.assertTrue(success)  # SQLite UPDATE doesn't fail for non-existent records
        
    def test_delete(self):
        """Test deleting a record."""
        # Create a test record
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        id = self.repository.create(data)
        
        # Delete it
        success = self.repository.delete(id)
        self.assertTrue(success)
        
        # Verify it's gone
        record = self.repository.get_by_id(id)
        self.assertIsNone(record)
        
        # Test deleting non-existent record
        success = self.repository.delete(999)
        self.assertTrue(success)  # SQLite DELETE doesn't fail for non-existent records
        
    def test_find(self):
        """Test finding records by conditions."""
        # Create test records
        records = [
            {'word': 'test1', 'category_id': 1, 'frequency': 10, 'allowed': True},
            {'word': 'test2', 'category_id': 1, 'frequency': 20, 'allowed': True},
            {'word': 'test3', 'category_id': 2, 'frequency': 30, 'allowed': True}
        ]
        
        for record in records:
            self.repository.create(record)
            
        # Find by category
        found = self.repository.find({'category_id': 1})
        self.assertEqual(len(found), 2)
        
        # Find by frequency
        found = self.repository.find({'frequency': 30})
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['word'], 'test3')
        
    def test_find_one(self):
        """Test finding a single record by conditions."""
        # Create test records
        data = {
            'word': 'test',
            'category_id': 1,
            'frequency': 10,
            'allowed': True
        }
        self.repository.create(data)
        
        # Find it
        record = self.repository.find_one({'word': 'test'})
        self.assertIsNotNone(record)
        self.assertEqual(record['frequency'], 10)
        
        # Test non-existent record
        record = self.repository.find_one({'word': 'nonexistent'})
        self.assertIsNone(record)
        
    def test_count(self):
        """Test counting records."""
        # Create test records
        records = [
            {'word': 'test1', 'category_id': 1, 'frequency': 10, 'allowed': True},
            {'word': 'test2', 'category_id': 1, 'frequency': 20, 'allowed': True},
            {'word': 'test3', 'category_id': 2, 'frequency': 30, 'allowed': True}
        ]
        
        for record in records:
            self.repository.create(record)
            
        # Count all
        total = self.repository.count()
        self.assertEqual(total, 3)
        
        # Count by condition
        count = self.repository.count({'category_id': 1})
        self.assertEqual(count, 2)
        
        # Count non-existent
        count = self.repository.count({'word': 'nonexistent'})
        self.assertEqual(count, 0)

if __name__ == '__main__':
    unittest.main() 