"""
This module is responsible for handling the sentiment analysis data layer.
It provides a unified interface for different sentiment analysis models
and handles data processing, caching, and result formatting.
"""
import logging
import time
from typing import Dict, List, Union, Any, Optional, Tuple
from functools import lru_cache

# Model Layer
from app.models.bertweet_model import BertweetSentiment

# Set up logger
logger = logging.getLogger(__name__)

class SentimentDataLayer:
    """
    Data layer for sentiment analysis operations.
    
    This class provides an abstraction over different sentiment analysis models
    and offers unified interfaces for text analysis, batch processing, and result caching.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Sentiment Data Layer.
        
        Args:
            config: Configuration dictionary containing model settings and parameters.
                Expected structure:
                - debug: Boolean for debug mode
                - sentiment_analysis.default_model: Model type to use
                - sentiment_analysis.[model_name]: Model-specific configuration
                
        Raises:
            ValueError: If the specified model is not supported
            RuntimeError: If model initialization fails
        """
        self.debug = config.get('debug', False)
        self._configure_logging()
        
        # Validate configuration
        if 'sentiment_analysis' not in config:
            raise ValueError("Configuration must contain 'sentiment_analysis' section")
            
        self.config = config.get('sentiment_analysis')
        self.default_model = self.config.get('default_model')
        
        if not self.default_model:
            raise ValueError("Default model must be specified in configuration")
        
        # Cache for sentiment results
        self.cache_size = self.config.get('cache_size', 1000)
        self.cache_enabled = self.config.get('enable_cache', True)
        
        logger.info(f"Initializing SentimentDataLayer with model: {self.default_model}")
        
        try:
            # Initialize the appropriate model based on the configuration
            if self.default_model == "bertweet":
                self.model = BertweetSentiment(config)
            # To add more models, uncomment and modify the following code:
            # elif self.default_model == "another_model":
            #     self.model = AnotherModel(config)
            else:
                raise ValueError(f"Unsupported sentiment analysis model: {self.default_model}")
                
            # Store model metadata
            self.model_info = {
                'name': self.default_model,
                'class_labels': getattr(self.model, 'class_labels', None)
            }
            
            logger.info(f"Model initialized successfully with {len(self.model.class_labels)} classes: {self.model.class_labels}")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment model: {str(e)}")
            raise RuntimeError(f"Sentiment model initialization failed: {str(e)}")
    
    def _configure_logging(self) -> None:
        """Configure logging based on debug setting."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @lru_cache(maxsize=1000)
    def _cached_analyze(self, text: str) -> Dict[str, Any]:
        """
        Internal method for cached sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        start_time = time.time()
        outputs, probabilities, predicted_label, confidence = self.model(text)
        processing_time = time.time() - start_time
        
        # Format the result
        return {
            'label': predicted_label,
            'confidence': confidence,
            'processing_time_ms': round(processing_time * 1000, 2),
            # Optional additional data
            'model': self.default_model,
            'timestamp': time.time(),
            # We only include probabilities if they're available
            'probabilities': {
                label: prob.item() for label, prob in 
                zip(self.model.class_labels, probabilities[0])
            } if probabilities is not None else None
        }
    
    def analyze(self, text: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """
        Perform sentiment analysis on the given text.
        
        Args:
            text: Input text for sentiment analysis
            include_probabilities: Whether to include class probabilities in the result
            
        Returns:
            Dictionary containing sentiment analysis results including predicted label and confidence
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'error': 'Empty text provided'
            }
            
        try:
            # Use cached analysis if enabled
            if self.cache_enabled:
                result = self._cached_analyze(text)
                logger.debug(f"Analyzed text with {len(text)} chars: {result['label']} ({result['confidence']:.4f})")
            else:
                # Direct analysis without caching
                outputs, probabilities, predicted_label, confidence = self.model(text)
                result = {
                    'label': predicted_label,
                    'confidence': confidence
                }
            
            # Filter out probabilities if not requested
            if not include_probabilities and 'probabilities' in result:
                result.pop('probabilities')
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                'error': 'An unexpected error occurred while processing the request.',
                'label': 'error',
                'confidence': 0.0
            }
    
    def batch_analyze(
        self, 
        texts: List[str], 
        include_probabilities: bool = False,
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            include_probabilities: Whether to include class probabilities in the results
            show_progress: Whether to display a progress bar for processing
            
        Returns:
            List of dictionaries containing sentiment analysis results
        """
        if not texts:
            return []
            
        try:
            # Use the batch processing capability of the model if available
            if hasattr(self.model, 'batch_process'):
                batch_results = self.model.batch_process(texts, show_progress=show_progress)
                
                # Format results
                formatted_results = []
                for i, (_, probs, label, confidence) in enumerate(batch_results):
                    result = {
                        'label': label,
                        'confidence': confidence,
                        'model': self.default_model
                    }
                    
                    # Add probabilities if requested
                    if include_probabilities and probs is not None:
                        result['probabilities'] = {
                            cls: probs[i].item() for i, cls in enumerate(self.model.class_labels)
                        }
                        
                    formatted_results.append(result)
                    
                return formatted_results
            else:
                # Fall back to processing texts one by one
                logger.warning("Batch processing not available in the model, processing texts individually")
                return [self.analyze(text, include_probabilities) for text in texts]
                
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            # Return error for each text
            return [
                {
                    'error': 'An unexpected error occurred during batch processing.',
                    'label': 'error',
                    'confidence': 0.0
                }
                for _ in texts
            ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded sentiment model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.default_model,
            'class_labels': getattr(self.model, 'class_labels', []),
            'cache_enabled': self.cache_enabled,
            'cache_size': self.cache_size
        }
        
    def clear_cache(self) -> None:
        """Clear the sentiment analysis cache."""
        if hasattr(self._cached_analyze, 'cache_clear'):
            self._cached_analyze.cache_clear()
            logger.info("Sentiment analysis cache cleared")
    
    def sample_analysis(self, sample_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a sample analysis to verify the model is working.
        
        Args:
            sample_text: Optional text to use for the sample analysis
            
        Returns:
            Sentiment analysis results for the sample text
        """
        text = sample_text or "I am feeling quite happy about this new feature!"
        result = self.analyze(text, include_probabilities=True)
        
        # Add the sample text to the result
        result['sample_text'] = text
        return result


if __name__ == "__main__":
    import json
    from pprint import pprint
    
    # Example configuration
    config = {
        'debug': True,
        'sentiment_analysis': {
            'default_model': "bertweet",
            'enable_cache': True,
            'cache_size': 500,
            'bertweet': {
                'model_name': 'finiteautomata/bertweet-base-sentiment-analysis',
                'device': 'cpu'
            }
        }
    }
    
    print("\n=== Sentiment Analysis Data Layer Demo ===\n")
    
    # Initialize the data layer
    sentiment_data = SentimentDataLayer(config)
    
    # Display model information
    print("Model Information:")
    pprint(sentiment_data.get_model_info())
    print("\n" + "-"*50 + "\n")
    
    # Individual text analysis examples
    print("Individual Text Analysis Examples:")
    positive_text = "I love this product! It's amazing and exceeded my expectations. 😍"
    negative_text = "I'm really disappointed with this service. It's terrible and frustrating. 😠"
    neutral_text = "The product arrived today. It seems to be working as described."
    
    # Analyze with probabilities
    print("\nPositive text (with probabilities):")
    pprint(sentiment_data.analyze(positive_text, include_probabilities=True))
    
    # Analyze without probabilities
    print("\nNegative text:")
    pprint(sentiment_data.analyze(negative_text))
    
    print("\nNeutral text:")
    pprint(sentiment_data.analyze(neutral_text))
    
    # Empty text example
    print("\nEmpty text handling:")
    pprint(sentiment_data.analyze(""))
    
    print("\n" + "-"*50 + "\n")
    
    # Batch processing example
    print("Batch Processing Example:")
    batch_texts = [
        "I absolutely love this app! The interface is so intuitive.",
        "This is the worst experience I've ever had with customer service.",
        "The package arrived on schedule. Contents were as described online.",
        "I'm not sure if I should upgrade to the premium version or not.",
        "The weather today is quite nice, might go for a walk later."
    ]
    
    print(f"\nAnalyzing batch of {len(batch_texts)} texts:")
    batch_results = sentiment_data.batch_analyze(batch_texts, show_progress=True)
    
    for i, result in enumerate(batch_results):
        print(f"\nText {i+1}: \"{batch_texts[i][:50]}...\"")
        print(f"Label: {result['label']}, Confidence: {result['confidence']:.4f}")
        
    print("\n" + "-"*50 + "\n")
    
    # Cache demonstration
    print("Cache Demonstration:")
    print("First analysis (uncached):")
    start = time.time()
    result1 = sentiment_data.analyze(positive_text)
    time1 = time.time() - start
    print(f"Time: {time1*1000:.2f}ms")
    
    print("\nSecond analysis (should use cache):")
    start = time.time()
    result2 = sentiment_data.analyze(positive_text)
    time2 = time.time() - start
    print(f"Time: {time2*1000:.2f}ms")
    print(f"Cache speedup: {time1/time2 if time2 > 0 else 'infinite'}x")
    
    # Sample analysis
    print("\nSample Analysis:")
    sample_result = sentiment_data.sample_analysis()
    pprint(sample_result)
    
    # Clean up
    print("\nClearing cache...")
    sentiment_data.clear_cache()
