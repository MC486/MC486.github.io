import unittest
from core.validation.trie import Trie, TrieNode

class TestTrie(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.trie = Trie()

    def test_initialization(self):
        """Test proper initialization of Trie"""
        self.assertIsInstance(self.trie.root, TrieNode)
        self.assertEqual(self.trie.total_words, 0)
        self.assertEqual(self.trie.max_word_length, 0)

    def test_insert_and_search(self):
        """Test word insertion and search functionality"""
        words = ["HELLO", "HELP", "HEAP"]
        for word in words:
            self.trie.insert(word)
            self.assertTrue(self.trie.search(word))
        
        # Test case insensitivity
        self.assertTrue(self.trie.search("hello"))
        self.assertTrue(self.trie.search("HELLO"))
        
        # Test non-existent word
        self.assertFalse(self.trie.search("NOTFOUND"))
        
        # Test empty string
        self.assertFalse(self.trie.search(""))

    def test_prefix_operations(self):
        """Test prefix-related operations"""
        words = ["HELLO", "HELP", "HEAP"]
        for word in words:
            self.trie.insert(word)
        
        # Test starts_with
        self.assertTrue(self.trie.starts_with("HE"))
        self.assertTrue(self.trie.starts_with("HEL"))
        self.assertFalse(self.trie.starts_with("HAT"))
        
        # Test prefix count
        self.assertEqual(self.trie.get_prefix_count("HE"), 3)
        self.assertEqual(self.trie.get_prefix_count("HEL"), 2)
        self.assertEqual(self.trie.get_prefix_count("HELP"), 1)
        self.assertEqual(self.trie.get_prefix_count("NOT"), 0)

    def test_word_deletion(self):
        """Test word deletion functionality"""
        words = ["HELLO", "HELP", "HEAP"]
        for word in words:
            self.trie.insert(word)
        
        # Test successful deletion
        self.assertTrue(self.trie.delete("HELLO"))
        self.assertFalse(self.trie.search("HELLO"))
        self.assertTrue(self.trie.search("HELP"))
        
        # Test prefix counts after deletion
        self.assertEqual(self.trie.get_prefix_count("HE"), 2)
        
        # Test deletion of non-existent word
        self.assertFalse(self.trie.delete("NOTFOUND"))
        
        # Test deletion of empty string
        self.assertFalse(self.trie.delete(""))

    def test_get_words_with_prefix(self):
        """Test retrieval of words with prefix"""
        words = ["HELLO", "HELP", "HEAP", "HAT", "HOPE"]
        for word in words:
            self.trie.insert(word)
        
        # Test valid prefix
        he_words = self.trie.get_words_with_prefix("HE")
        self.assertEqual(len(he_words), 3)
        self.assertIn("HELLO", he_words)
        self.assertIn("HELP", he_words)
        self.assertIn("HEAP", he_words)
        
        # Test prefix with no matches
        no_words = self.trie.get_words_with_prefix("NOT")
        self.assertEqual(len(no_words), 0)
        
        # Test max_words limit
        limited_words = self.trie.get_words_with_prefix("H", max_words=2)
        self.assertEqual(len(limited_words), 2)

    def test_word_counts(self):
        """Test word counting functionality"""
        words = ["HELLO", "HELP", "HEAP"]
        for word in words:
            self.trie.insert(word)
        
        self.assertEqual(self.trie.total_words, 3)
        self.assertEqual(self.trie.max_word_length, 5)
        
        # Test duplicate insertion
        self.trie.insert("HELLO")
        node = self.trie._traverse("HELLO")
        self.assertEqual(node.word_count, 2)
        self.assertEqual(self.trie.total_words, 4)

    def test_case_handling(self):
        """Test case insensitive handling"""
        variations = ["Hello", "HELLO", "hello"]
        for word in variations:
            self.trie.insert(word)
        
        # All variations should be found
        for word in variations:
            self.assertTrue(self.trie.search(word))
        
        # Count should reflect they're the same word
        node = self.trie._traverse("HELLO")
        self.assertEqual(node.word_count, 3)

    def test_empty_and_invalid_inputs(self):
        """Test handling of empty and invalid inputs"""
        # Empty string
        self.trie.insert("")
        self.assertEqual(self.trie.total_words, 0)
        
        # None input
        self.trie.insert(None)
        self.assertEqual(self.trie.total_words, 0)
        
        # Special characters
        self.trie.insert("HELLO!")
        self.assertTrue(self.trie.search("HELLO!"))

if __name__ == '__main__':
    unittest.main()