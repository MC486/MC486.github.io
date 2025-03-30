import numpy as np
from collections import defaultdict
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class MarkovChain:
    def __init__(self, order: int = 2):
        """
        Initialize the Markov Chain with a specified order.
        
        Args:
            order (int): The order of the Markov Chain (default: 2)
            
        Raises:
            ValueError: If order is less than 1
        """
        if order < 1:
            raise ValueError("Order must be at least 1")
        self.order = order
        # Use nested defaultdict to automatically initialize transition counts
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.start_states = defaultdict(int)
        self.is_trained = False
        
    def train(self, word_list: List[str]) -> None:
        """
        Train the Markov Chain on a list of words.
        
        Args:
            word_list (List[str]): List of words to train on
            
        Raises:
            ValueError: If word_list is empty or contains invalid words
        """
        if not word_list:
            raise ValueError("Training data cannot be empty")
            
        # Filter out invalid words (non-alphabetic)
        valid_words = [word for word in word_list if isinstance(word, str) and word.isalpha()]
        if not valid_words:
            raise ValueError("No valid words found in training data")
            
        logger.info(f"Training Markov Chain on {len(valid_words)} words")
        
        for word in valid_words:
            if len(word) < self.order:
                continue
                
            # Track frequency of starting sequences
            start_state = word[:self.order].lower()
            self.start_states[start_state] += 1
            
            # Build transition matrix: count occurrences of each character following each state
            for i in range(len(word) - self.order):
                current_state = word[i:i+self.order].lower()
                next_char = word[i+self.order].lower()
                self.transitions[current_state][next_char] += 1
                
        self.is_trained = True
        logger.info("Markov Chain training completed")
                
    def generate_word(self, max_length: int = 10, min_length: int = 3) -> Optional[str]:
        """
        Generate a word using the trained Markov Chain.
        
        Args:
            max_length (int): Maximum length of the generated word
            min_length (int): Minimum length of the generated word
            
        Returns:
            Optional[str]: Generated word or None if generation fails
            
        Raises:
            RuntimeError: If the model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generating words")
            
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
            
        if not self.start_states:
            logger.warning("No start states available for word generation")
            return None
            
        # Convert start state frequencies to probabilities
        start_states = list(self.start_states.keys())
        start_probs = np.array(list(self.start_states.values()))
        start_probs = start_probs / start_probs.sum()
        
        try:
            # Select initial state based on frequency distribution
            current_state = np.random.choice(start_states, p=start_probs)
            word = current_state
            
            # Generate word character by character using transition probabilities
            while len(word) < max_length:
                if current_state not in self.transitions:
                    break
                    
                # Get possible next characters and their probabilities
                next_chars = list(self.transitions[current_state].keys())
                if not next_chars:
                    break
                    
                # Convert transition counts to probabilities
                probs = np.array(list(self.transitions[current_state].values()))
                probs = probs / probs.sum()
                
                # Select next character based on transition probabilities
                next_char = np.random.choice(next_chars, p=probs)
                word += next_char
                current_state = word[-self.order:]
                
            if len(word) >= min_length:
                return word
            return None
            
        except Exception as e:
            logger.error(f"Error generating word: {str(e)}")
            return None
            
    def get_state_probabilities(self, state: str) -> Optional[dict]:
        """
        Get transition probabilities for a given state.
        
        Args:
            state (str): The state to get probabilities for
            
        Returns:
            Optional[dict]: Dictionary of next characters and their probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting probabilities")
            
        state = state.lower()
        if state not in self.transitions:
            return None
            
        # Convert transition counts to probabilities
        transitions = self.transitions[state]
        total = sum(transitions.values())
        if total == 0:
            return None
            
        return {char: count/total for char, count in transitions.items()}
        
    def save(self, filepath: str) -> None:
        """
        Save the Markov Chain model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        import pickle
        # Convert defaultdict to regular dict for serialization
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'transitions': dict(self.transitions),
                'start_states': dict(self.start_states),
                'is_trained': self.is_trained
            }, f)
            
    def load(self, filepath: str) -> None:
        """
        Load the Markov Chain model from disk.
        
        Args:
            filepath (str): Path to load the model from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid data
        """
        import pickle
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.order = data['order']
            # Restore defaultdict structure
            self.transitions = defaultdict(lambda: defaultdict(int))
            for state, transitions in data['transitions'].items():
                self.transitions[state].update(transitions)
            self.start_states = defaultdict(int, data['start_states'])
            self.is_trained = data['is_trained']
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid model file: {str(e)}") 