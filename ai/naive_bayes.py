import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List, Dict, Any, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict

logger = logging.getLogger(__name__)

class WordNaiveBayes:
    """
    Naive Bayes classifier for word prediction.
    Uses word frequencies and patterns to estimate probabilities.
    """
    def __init__(self):
        """Initialize the Naive Bayes classifier."""
        self.word_counts = {}
        self.total_words = 0
        self.letter_counts = defaultdict(int)
        self.position_counts = defaultdict(lambda: defaultdict(int))
        self.pattern_counts = defaultdict(int)
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        self.training_stats = {}
        
    def train(self, words: List[str], labels: List[str], validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the Naive Bayes classifier on word data.
        
        Args:
            words: List of training words
            labels: List of corresponding labels (e.g., valid/invalid)
            validation_split: Proportion of data to use for validation
            
        Returns:
            Dict[str, float]: Dictionary containing training metrics
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        if not words or not labels:
            raise ValueError("Training data cannot be empty")
            
        if len(words) != len(labels):
            raise ValueError("Number of words and labels must match")
            
        # Validate input data and normalize case
        valid_words = [word.upper() for word in words if isinstance(word, str) and word.isalpha()]
        if not valid_words:
            raise ValueError("No valid words found in training data")
            
        logger.info(f"Training Naive Bayes classifier on {len(valid_words)} words")
        
        try:
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                valid_words, labels, test_size=validation_split, random_state=42
            )
            
            # Convert words to character n-gram features
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_val_vec = self.vectorizer.transform(X_val)
            
            # Store feature names and class names
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.class_names = np.unique(labels)
            
            # Train the classifier
            self.classifier.fit(X_train_vec, y_train)
            
            # Evaluate on validation set
            y_pred = self.classifier.predict(X_val_vec)
            accuracy = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred, output_dict=True)
            
            # Store training statistics
            self.training_stats = {
                'accuracy': accuracy,
                'classification_report': report,
                'n_features': len(self.feature_names),
                'n_classes': len(self.class_names),
                'n_samples': len(valid_words)
            }
            
            self.is_trained = True
            logger.info(f"Training completed. Validation accuracy: {accuracy:.3f}")
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
        
    def predict(self, word: str) -> str:
        """
        Predict the label for a given word.
        
        Args:
            word: The word to classify
            
        Returns:
            str: Predicted label
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input word is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before making predictions")
            
        if not isinstance(word, str) or not word.isalpha():
            raise ValueError("Input must be a valid alphabetic word")
            
        try:
            # Transform the input word (normalize case)
            X = self.vectorizer.transform([word.upper()])
            
            # Make prediction
            return self.classifier.predict(X)[0]
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
        
    def predict_proba(self, word: str) -> Dict[str, float]:
        """
        Get probability estimates for each class.
        
        Args:
            word: The word to classify
            
        Returns:
            Dict[str, float]: Dictionary mapping labels to probabilities
            
        Raises:
            RuntimeError: If model is not trained
            ValueError: If input word is invalid
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before making predictions")
            
        if not isinstance(word, str) or not word.isalpha():
            raise ValueError("Input must be a valid alphabetic word")
            
        try:
            # Transform the input word (normalize case)
            X = self.vectorizer.transform([word.upper()])
            
            # Get probability estimates
            probs = self.classifier.predict_proba(X)[0]
            
            # Create dictionary mapping labels to probabilities
            return dict(zip(self.classifier.classes_, probs))
            
        except Exception as e:
            logger.error(f"Error getting probability estimates: {str(e)}")
            raise RuntimeError(f"Probability estimation failed: {str(e)}")
        
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each feature (character n-gram).
        
        Returns:
            Dict[str, float]: Dictionary mapping features to their importance scores
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Classifier must be trained before getting feature importance")
            
        try:
            # Get feature importance scores
            importance_scores = np.abs(self.classifier.coef_).mean(axis=0)
            
            # Create dictionary mapping features to importance scores
            return dict(zip(self.feature_names, importance_scores))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise RuntimeError(f"Feature importance calculation failed: {str(e)}")
            
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the training process.
        
        Returns:
            Dict[str, Any]: Dictionary containing training statistics
        """
        return self.training_stats.copy()
        
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'classifier': self.classifier,
                    'is_trained': self.is_trained,
                    'feature_names': self.feature_names,
                    'class_names': self.class_names,
                    'training_stats': self.training_stats
                }, f)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            
    def load(self, filepath: str) -> None:
        """
        Load the trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains invalid data
        """
        import pickle
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = data['is_trained']
            self.feature_names = data['feature_names']
            self.class_names = data['class_names']
            self.training_stats = data['training_stats']
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Invalid model file: {str(e)}") 