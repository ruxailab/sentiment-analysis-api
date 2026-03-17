"""
This module defines the BertweetSentiment class, which is a PyTorch model for sentiment analysis using the Bertweet model.
"""
import torch
import torch.nn as nn
import logging  

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class BertweetSentiment(nn.Module):
    def __init__(self,config: dict)->None:
        """
        Initialize the Bertweet model for sentiment analysis.
        :param config: The configuration object containing model and device info.
        """
        

        super(BertweetSentiment, self).__init__()

        self.config = config.get('sentiment_analysis').get('bertweet')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        try:
            # Initialize the Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True, clean_up_tokenization_spaces=True)

            # Initialize the Model
            self.model= AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)

            logger.info(f"Successfully loaded model: {self.model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load BERTweet model: {str(e)}")
            raise e

        # Load the model configuration to get class labels
        self.model_config = self.model.config

        # Get Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

    def forward(self,texts):
        """
        Perform sentiment analysis on a single text or a list of texts (Batch).

        Args:
            texts (str or list): Input text or list of texts for sentiment analysis.

        Returns:
            list: A list of dictionaries containing text, label, and confidence score.
        """
        # Handle backward compatibility: wrap single string in a list
        if isinstance(texts, str):
            texts = [texts]

        
        # Tokenize the input texts with padding and truncation
        # Padding ensures all sequences in the batch have the same length
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)

        # Disable gradient calculation for efficiency
        with torch.no_grad():
            # Forward pass
            outputs = self.model(**inputs)

            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Process results while maintaining the original order
        results = []
        confidences, predicted_classes = torch.max(probabilities, dim=1)
        
        for i in range(len(texts)):
            label = self.class_labels[predicted_classes[i].item()]
            results.append({
                "text": texts[i],
                "label": label,
                "confidence": confidences[i].item()
            })

        return results


# if __name__ == "__main__":
#     config = {
#         'debug': True,
#         'sentiment_analysis': {
#             'default_model': "bertweet",  # Specify the default sentiment analysis model (e.g., bertweet, another_model)
#             'bertweet': {
#                 'model_name': "finiteautomata/bertweet-base-sentiment-analysis",
#                 'device': 'cpu'
#             }
#         }
#     }
#     logger.info(f"Config loaded: {config}")
    
#     model = BertweetSentiment(config)
    
#     logger.info(f"Model Labels: {model.class_labels}")

#     test_texts = [
#             "I love the new features of the app!",
#             "I hate the new features of the app!",
#             "Hi how are u?"
#         ]

#     logger.info(f"Running batch inference on {len(test_texts)} samples...")
#     try:
#         results = model(test_texts)
        
#         # Display Results in a clean format
#         logger.info("--- Batch Results ---")
#         for res in results:
#             logger.info(f"Text: {res['text']}")
#             logger.info(f"Sentiment: {res['label']} | Confidence: {res['confidence']:.4f}")
#             logger.info("-" * 20)
            
#     except Exception as e:
#         logger.error(f"An error occurred during testing: {e}")

# # Run:
# # python -m app.models.bertweet_model