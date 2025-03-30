class TrieNode:
    """A node in the Trie data structure."""
    def __init__(self):
        self.children = {}  # Dictionary mapping characters to child nodes
        self.is_end = False  # Flag to mark end of a word
        self.word_count = 0  # Number of words ending at this node
        self.prefix_count = 0  # Number of words using this node as prefix

class Trie:
    """Trie data structure for efficient word storage and validation."""
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
        self.max_word_length = 0

    def insert(self, word: str) -> None:
        """Insert a word into the Trie.
        
        Args:
            word: The word to insert
        """
        if not word:
            return

        word = word.upper()
        node = self.root
        
        # Update prefix counts along the path
        for char in word:
            node.prefix_count += 1
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark word ending and update counts
        node.prefix_count += 1
        node.is_end = True
        node.word_count += 1
        self.total_words += 1
        self.max_word_length = max(self.max_word_length, len(word))

    def search(self, word: str) -> bool:
        """Search for a complete word in the Trie.
        
        Args:
            word: The word to search for
            
        Returns:
            bool: True if the word exists in the Trie
        """
        if not word:
            return False

        word = word.upper()
        node = self._traverse(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Check if any word in the Trie starts with the given prefix.
        
        Args:
            prefix: The prefix to search for
            
        Returns:
            bool: True if any word starts with the prefix
        """
        if not prefix:
            return True

        prefix = prefix.upper()
        node = self._traverse(prefix)
        return node is not None

    def get_prefix_count(self, prefix: str) -> int:
        """Get the number of words that start with the given prefix.
        
        Args:
            prefix: The prefix to count
            
        Returns:
            int: Number of words starting with the prefix
        """
        if not prefix:
            return self.total_words

        prefix = prefix.upper()
        node = self._traverse(prefix)
        return node.prefix_count if node else 0

    def _traverse(self, chars: str) -> TrieNode:
        """Traverse the Trie following the given characters.
        
        Args:
            chars: String of characters to follow
            
        Returns:
            TrieNode: The node at the end of the path, or None if path doesn't exist
        """
        node = self.root
        for char in chars:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def delete(self, word: str) -> bool:
        """Delete a word from the Trie.
        
        Args:
            word: The word to delete
            
        Returns:
            bool: True if the word was found and deleted
        """
        if not word:
            return False

        word = word.upper()
        node = self.root
        path = [(node, '')]  # Stack of (node, char) pairs
        
        # Traverse to the word's end node, storing the path
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
            path.append((node, char))
        
        # Word must exist to be deleted
        if not node.is_end:
            return False
        
        # Update word count and end flag
        node.is_end = False
        node.word_count -= 1
        self.total_words -= 1
        
        # Update prefix counts and remove empty nodes
        for node, char in reversed(path):
            node.prefix_count -= 1
            if node.prefix_count == 0 and path[-1][0] != node:
                parent = path[path.index((node, char)) - 1][0]
                del parent.children[char]

        return True

    def get_words_with_prefix(self, prefix: str, max_words: int = 100) -> list[str]:
        """Get all words that start with the given prefix.
        
        Args:
            prefix: The prefix to search for
            max_words: Maximum number of words to return
            
        Returns:
            list[str]: List of words starting with the prefix
        """
        if not prefix:
            return []

        prefix = prefix.upper()
        words = []
        node = self._traverse(prefix)
        
        if not node:
            return words
            
        def dfs(node: TrieNode, current_word: str):
            if len(words) >= max_words:
                return
                
            if node.is_end:
                words.append(current_word)
                
            for char, child in node.children.items():
                dfs(child, current_word + char)
        
        dfs(node, prefix)
        return words