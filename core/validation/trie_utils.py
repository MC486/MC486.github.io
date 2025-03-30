import os
import json
import pickle
from typing import List, Set, Optional
from .trie import Trie

class TrieUtils:
    @staticmethod
    def load_word_list(file_path: str) -> Set[str]:
        """Load words from a file into a set.
        
        Args:
            file_path: Path to the word list file
            
        Returns:
            Set of words from the file
        
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Word list file not found: {file_path}")
            
        words = set()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip().upper()
                if word:  # Skip empty lines
                    words.add(word)
        return words

    @staticmethod
    def build_trie_from_file(file_path: str) -> Trie:
        """Build a Trie from a word list file.
        
        Args:
            file_path: Path to the word list file
            
        Returns:
            Populated Trie instance
        """
        words = TrieUtils.load_word_list(file_path)
        return TrieUtils.build_trie_from_words(words)

    @staticmethod
    def build_trie_from_words(words: Set[str]) -> Trie:
        """Build a Trie from a set of words.
        
        Args:
            words: Set of words to add to the Trie
            
        Returns:
            Populated Trie instance
        """
        trie = Trie()
        for word in words:
            trie.insert(word)
        return trie

    @staticmethod
    def save_trie(trie: Trie, file_path: str) -> None:
        """Save a Trie to a file for later loading.
        
        Args:
            trie: Trie instance to save
            file_path: Path where to save the Trie
        """
        with open(file_path, 'wb') as file:
            pickle.dump(trie, file)

    @staticmethod
    def load_trie(file_path: str) -> Optional[Trie]:
        """Load a Trie from a file.
        
        Args:
            file_path: Path to the saved Trie file
            
        Returns:
            Loaded Trie instance or None if file doesn't exist
        """
        if not os.path.exists(file_path):
            return None
            
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def get_memory_usage(trie: Trie) -> dict:
        """Calculate approximate memory usage of the Trie.
        
        Args:
            trie: Trie instance to analyze
            
        Returns:
            Dictionary with memory statistics
        """
        def count_nodes(node) -> tuple[int, int]:
            nodes = 1  # Count current node
            chars = len(node.children)  # Count characters in current node
            
            for child in node.children.values():
                child_nodes, child_chars = count_nodes(child)
                nodes += child_nodes
                chars += child_chars
            
            return nodes, chars

        total_nodes, total_chars = count_nodes(trie.root)
        
        return {
            "total_nodes": total_nodes,
            "total_characters": total_chars,
            "total_words": trie.total_words,
            "max_word_length": trie.max_word_length,
            "approximate_bytes": (
                total_nodes * 24 +  # TrieNode object overhead
                total_chars * 28 +  # Dictionary entries
                trie.total_words * 4  # Word count integers
            )
        }

    @staticmethod
    def optimize_trie(trie: Trie) -> Trie:
        """Create an optimized copy of the Trie.
        
        Args:
            trie: Trie instance to optimize
            
        Returns:
            New, optimized Trie instance
        """
        # Get all words from the trie
        words = []
        def collect_words(node, prefix=""):
            if node.is_end:
                words.extend([prefix] * node.word_count)
            for char, child in node.children.items():
                collect_words(child, prefix + char)
        
        collect_words(trie.root)
        
        # Create new trie with sorted words for better memory layout
        new_trie = Trie()
        for word in sorted(words):
            new_trie.insert(word)
        
        return new_trie

    @staticmethod
    def export_statistics(trie: Trie, file_path: str) -> None:
        """Export Trie statistics to a JSON file.
        
        Args:
            trie: Trie instance to analyze
            file_path: Path where to save the statistics
        """
        stats = {
            "memory_usage": TrieUtils.get_memory_usage(trie),
            "word_length_distribution": {},
            "prefix_statistics": {}
        }
        
        # Collect word length distribution
        def collect_lengths(node, length=0):
            if node.is_end:
                stats["word_length_distribution"][length] = \
                    stats["word_length_distribution"].get(length, 0) + node.word_count
            for child in node.children.values():
                collect_lengths(child, length + 1)
        
        collect_lengths(trie.root)
        
        # Save statistics
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(stats, file, indent=2)