"""
This module defines the BertweetSentiment class, optimized with ONNX Runtime 
for low-latency CPU sentiment analysis using the Bertweet model.
"""
import torch
import torch.nn as nn
import logging

from transformers import AutoTokenizer
# Injecting Hugging Face Optimum for ONNX Runtime acceleration
from optimum.onnxruntime import ORTModelForSequenceClassification

logger = logging.getLogger(__name__)

class BertweetSentiment(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Initialize the ONNX-optimized Bertweet model for sentiment analysis.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('sentiment_analysis').get('bertweet')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        super(BertweetSentiment, self).__init__()
        
        logger.info(f"Initializing ONNX-optimized sentiment model: {self.model_name} on {self.device}")

        # Initialize the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the Model dynamically into an ONNX graph using export=True
        # This bypasses the heavy native PyTorch execution
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_name,
            export=True
        )
        
        # Load the model configuration to get class labels
        self.model_config = self.model.config

        # Get Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

    def forward(self, text) -> tuple:
        """
        Perform sentiment analysis on the given text using ONNX runtime optimizations.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            tuple: Model outputs, probabilities, predicted label, and confidence score.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Forward pass through the ONNX graph
        outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the predicted sentiment
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get the corresponding class label
        predicted_label = self.class_labels[predicted_class]

        return outputs, probabilities, predicted_label, probabilities[0][predicted_class].item()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        'debug': True,
        'sentiment_analysis': {
            'default_model': "bertweet",
            'bertweet': {
                'model_name': "finiteautomata/bertweet-base-sentiment-analysis",
                'device': 'cpu'
            }
        }
    }
    print("Testing ONNX Inference Implementation...")
    model = BertweetSentiment(config)
    print("Model initialized successfully.")
    
    texts_to_test = [
        "I love the new features of the app!",
        "I hate the new features of the app!",
        "Hi how are u?"
    ]
    
    for t in texts_to_test:
        _, _, label, conf = model(t)
        print(f"Text: '{t}' | Label: {label} | Confidence: {conf:.4f}")