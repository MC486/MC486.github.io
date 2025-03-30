import json
import unittest
import os
import tempfile
from unittest.mock import patch, mock_open
from core.validation.trie_utils import TrieUtils
from core.validation.trie import Trie

class TestTrieUtils(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_words = {"HELLO", "HELP", "HEAP", "WORLD"}

    def tearDown(self):
        """Cleanup after each test"""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_load_word_list(self):
        """Test loading words from a file"""
        # Test with mock file
        mock_content = "HELLO\nHELP\nHEAP\n\nWORLD"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            words = TrieUtils.load_word_list("mock_path")
            self.assertEqual(words, self.sample_words)

        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            TrieUtils.load_word_list("nonexistent.txt")

    def test_build_trie_from_words(self):
        """Test building Trie from a set of words"""
        trie = TrieUtils.build_trie_from_words(self.sample_words)
        
        # Verify all words were added
        for word in self.sample_words:
            self.assertTrue(trie.search(word))
        
        # Verify word count
        self.assertEqual(trie.total_words, len(self.sample_words))

    def test_build_trie_from_file(self):
        """Test building Trie from a file"""
        mock_content = "HELLO\nHELP\nHEAP\n\nWORLD"
        with patch("builtins.open", mock_open(read_data=mock_content)):
            trie = TrieUtils.build_trie_from_file("mock_path")
            
            # Verify all words were added
            for word in self.sample_words:
                self.assertTrue(trie.search(word))

    def test_save_and_load_trie(self):
        """Test Trie serialization and deserialization"""
        # Create and save a trie
        original_trie = TrieUtils.build_trie_from_words(self.sample_words)
        save_path = os.path.join(self.temp_dir, "trie.pkl")
        TrieUtils.save_trie(original_trie, save_path)
        
        # Load and verify
        loaded_trie = TrieUtils.load_trie(save_path)
        self.assertEqual(loaded_trie.total_words, original_trie.total_words)
        for word in self.sample_words:
            self.assertTrue(loaded_trie.search(word))
        
        # Test loading non-existent file
        self.assertIsNone(TrieUtils.load_trie("nonexistent.pkl"))

    def test_get_memory_usage(self):
        """Test memory usage calculation"""
        trie = TrieUtils.build_trie_from_words(self.sample_words)
        stats = TrieUtils.get_memory_usage(trie)
        
        self.assertIn("total_nodes", stats)
        self.assertIn("total_characters", stats)
        self.assertIn("total_words", stats)
        self.assertIn("max_word_length", stats)
        self.assertIn("approximate_bytes", stats)
        
        self.assertEqual(stats["total_words"], len(self.sample_words))
        self.assertTrue(stats["approximate_bytes"] > 0)

    def test_optimize_trie(self):
        """Test Trie optimization"""
        # Create trie with duplicate words
        words = {"HELLO", "HELLO", "HELP"}
        original_trie = TrieUtils.build_trie_from_words(words)
        
        optimized_trie = TrieUtils.optimize_trie(original_trie)
        
        # Verify functionality is preserved
        self.assertEqual(optimized_trie.total_words, original_trie.total_words)
        for word in words:
            self.assertTrue(optimized_trie.search(word))
        
        # Verify memory usage is less or equal
        original_memory = TrieUtils.get_memory_usage(original_trie)["approximate_bytes"]
        optimized_memory = TrieUtils.get_memory_usage(optimized_trie)["approximate_bytes"]
        self.assertLessEqual(optimized_memory, original_memory)

    def test_export_statistics(self):
        """Test statistics export"""
        trie = TrieUtils.build_trie_from_words(self.sample_words)
        export_path = os.path.join(self.temp_dir, "stats.json")
        
        TrieUtils.export_statistics(trie, export_path)
        
        # Verify file exists and contains valid JSON
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'r') as f:
            stats = json.loads(f.read())
            
        self.assertIn("memory_usage", stats)
        self.assertIn("word_length_distribution", stats)
        self.assertIn("prefix_statistics", stats)

    def test_word_case_handling(self):
        """Test case insensitive word handling"""
        mixed_case_words = {"Hello", "WORLD", "Help"}
        trie = TrieUtils.build_trie_from_words(mixed_case_words)
        
        # All variations should be found
        self.assertTrue(trie.search("HELLO"))
        self.assertTrue(trie.search("hello"))
        self.assertTrue(trie.search("Hello"))

if __name__ == '__main__':
    unittest.main()