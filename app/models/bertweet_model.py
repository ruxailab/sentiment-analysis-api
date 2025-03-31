"""
This module defines the BertweetSentiment class, which is a PyTorch model for sentiment analysis 
using the BERTweet model architecture.
"""
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BertweetSentiment(nn.Module):
    """
    A sentiment analysis model based on BERTweet architecture.
    
    This class provides methods for sentiment analysis on text data using the BERTweet model,
    which is specifically designed for processing social media text. It handles tokenization,
    inference, batch processing, and model persistence.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the BERTweet model for sentiment analysis.
        
        Args:
            config: Configuration dictionary containing model settings and device info.
                Expected structure:
                - debug: Boolean for debug mode
                - sentiment_analysis.bertweet.model_name: Model identifier
                - sentiment_analysis.bertweet.device: Device to run the model on ('cpu', 'cuda', etc.)
                - sentiment_analysis.bertweet.cache_size: (Optional) Size for LRU cache
        
        Raises:
            ValueError: If required configuration parameters are missing
            RuntimeError: If model initialization fails
        """
        super(BertweetSentiment, self).__init__()
        
        # Initialize logging based on debug setting
        self.debug = config.get('debug', False)
        self._configure_logging()
        
        # Extract configuration
        if 'sentiment_analysis' not in config or 'bertweet' not in config.get('sentiment_analysis', {}):
            raise ValueError("Configuration must contain 'sentiment_analysis.bertweet' section")
            
        self.config = config.get('sentiment_analysis').get('bertweet')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device', 'cpu')
        self.cache_size = self.config.get('cache_size', 128)
        
        if not self.model_name:
            raise ValueError("Model name must be specified in configuration")
        
        logger.info(f"Initializing BertweetSentiment with model: {self.model_name} on device: {self.device}")
        
        try:
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Initialize the model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Extract class labels from model configuration
            self.model_config = self.model.config
            self._extract_class_labels()
            
            logger.info(f"Model initialized successfully with {len(self.class_labels)} classes: {self.class_labels}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def _configure_logging(self) -> None:
        """Configure logging based on debug setting."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _extract_class_labels(self) -> None:
        """Extract class labels from model configuration."""
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            # Default to positive/negative if labels aren't found
            logger.warning("No labels found in model config. Using default labels.")
            self.class_labels = ["negative", "neutral", "positive"]
    
    @lru_cache(maxsize=128)
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text with caching for repeated inputs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary of tokenized inputs
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    def forward(self, text: Union[str, List[str]]) -> Union[
        Tuple[Any, torch.Tensor, str, float], 
        List[Tuple[Any, torch.Tensor, str, float]]
    ]:
        """
        Perform sentiment analysis on the given text(s).

        Args:
            text: Input text or list of texts for sentiment analysis.

        Returns:
            For single text: Tuple of (model_outputs, probabilities, predicted_label, confidence)
            For multiple texts: List of such tuples
        """
        if isinstance(text, list):
            return self.batch_process(text)
        
        try:
            # Tokenize the input
            inputs = self._tokenize(text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run model in evaluation mode
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(**inputs)
            
            # Convert logits to probabilities
            probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Get the predicted sentiment class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_label = self.class_labels[predicted_class]
            confidence = probabilities[0][predicted_class].item()
            
            if self.debug:
                logger.debug(f"Text: '{text[:50]}...' → {predicted_label} ({confidence:.4f})")
                
            return outputs, probabilities, predicted_label, confidence
        except Exception as e:
            logger.error(f"Inference error for text '{text[:50]}...': {str(e)}")
            # Return a default value in case of error
            return None, None, "error", 0.0
    
    def batch_process(
        self, 
        texts: List[str], 
        batch_size: int = 16, 
        show_progress: bool = False
    ) -> List[Tuple[Any, torch.Tensor, str, float]]:
        """
        Process a batch of texts for sentiment analysis.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            
        Returns:
            List of tuples containing model outputs, probabilities, predicted labels, and confidence scores
        """
        results = []
        
        # Use tqdm for progress tracking if requested
        iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            try:
                # Tokenize batch
                batch_inputs = self.tokenizer(batch_texts, return_tensors="pt", 
                                             truncation=True, padding=True)
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                
                # Process batch
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(**batch_inputs)
                
                probabilities = F.softmax(outputs.logits, dim=-1)
                
                # Process each item in batch
                for j, probs in enumerate(probabilities):
                    predicted_class = torch.argmax(probs).item()
                    predicted_label = self.class_labels[predicted_class]
                    confidence = probs[predicted_class].item()
                    
                    results.append((outputs, probs, predicted_label, confidence))
            except Exception as e:
                logger.error(f"Batch processing error at index {i}: {str(e)}")
                # Add None results for the failed batch
                results.extend([(None, None, "error", 0.0)] * len(batch_texts))
                
        return results
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the model on a dataset with ground truth labels.
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels corresponding to texts
            
        Returns:
            Dictionary with evaluation metrics (accuracy, per-class F1, etc.)
        """
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        
        results = self.batch_process(texts, show_progress=True)
        predicted_labels = [result[2] for result in results]
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predicted_labels, labels) if pred == true)
        accuracy = correct / len(texts) if texts else 0.0
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in set(self.class_labels):
            tp = sum(1 for pred, true in zip(predicted_labels, labels) if pred == cls and true == cls)
            fp = sum(1 for pred, true in zip(predicted_labels, labels) if pred == cls and true != cls)
            fn = sum(1 for pred, true in zip(predicted_labels, labels) if pred != cls and true == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
        
        return {
            "accuracy": accuracy,
            "class_metrics": class_metrics
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Directory path to save the model to
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        try:
            # Save the model
            self.model.save_pretrained(path)
            # Save the tokenizer
            self.tokenizer.save_pretrained(path)
            # Save class labels 
            with open(os.path.join(path, "class_labels.txt"), "w") as f:
                f.write("\n".join(self.class_labels))
                
            logger.info(f"Model successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> "BertweetSentiment":
        """
        Load a saved model from the specified path.
        
        Args:
            path: Directory path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded BertweetSentiment model
        """
        config = {
            "debug": False,
            "sentiment_analysis": {
                "bertweet": {
                    "model_name": path,
                    "device": device
                }
            }
        }
        
        model = cls(config)
        
        # Load custom class labels if available
        class_labels_path = os.path.join(path, "class_labels.txt")
        if os.path.exists(class_labels_path):
            with open(class_labels_path, "r") as f:
                model.class_labels = [line.strip() for line in f.readlines()]
                
        return model


if __name__ == "__main__":
    # Example configuration
    config = {
        'debug': True,
        'sentiment_analysis': {
            'default_model': "bertweet",
            'bertweet': {
                'model_name': "finiteautomata/bertweet-base-sentiment-analysis",
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'cache_size': 256
            }
        }
    }
    
    print(f"Using device: {config['sentiment_analysis']['bertweet']['device']}")
    
    # Initialize model
    model = BertweetSentiment(config)
    print(f"Model initialized with classes: {model.class_labels}")
    
    # Single text example
    text = "I love the new features of the app! 😍"
    _, _, sentiment, confidence = model(text)
    print(f"Text: '{text}'\nSentiment: {sentiment}\nConfidence: {confidence:.4f}\n")
    
    # Batch processing example
    texts = [
        "I hate the new features of the app! 😡",
        "This product is just okay, nothing special",
        "Hi how are u?",
        "The service was excellent and the staff was very friendly!"
    ]
    
    print("Batch processing example:")
    results = model.batch_process(texts, show_progress=True)
    for i, (_, _, sentiment, confidence) in enumerate(results):
        print(f"Text: '{texts[i]}'\nSentiment: {sentiment}\nConfidence: {confidence:.4f}\n")
    
    # Evaluation example (with mock data)
    print("Evaluation example:")
    eval_texts = texts
    eval_labels = ["negative", "neutral", "neutral", "positive"]
    eval_results = model.evaluate(eval_texts, eval_labels)
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    for cls, metrics in eval_results['class_metrics'].items():
        print(f"{cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Model saving example (commented out to avoid unintended file creation)
    # model.save_model("./saved_sentiment_model")
    
    # Model loading example
    # loaded_model = BertweetSentiment.load_model("./saved_sentiment_model")
