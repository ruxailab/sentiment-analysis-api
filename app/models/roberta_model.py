"""
This module defines the RoBERTaSentiment class, which is a PyTorch model for sentiment analysis using the RoBERTa model.
"""
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Mapping from RoBERTa labels to standard labels
LABEL_MAPPING = {
    "positive": "POS",
    "neutral": "NEU",
    "negative": "NEG"
}

class RoBERTaSentiment(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Initialize the RoBERTa model for sentiment analysis.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('sentiment_analysis').get('roberta')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        super(RoBERTaSentiment, self).__init__()

        # Initialize the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the Model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

        # Load the model configuration to get class labels
        self.model_config = self.model.config

        # Get Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

    def forward(self, text) -> tuple:
        """
        Perform sentiment analysis on the given text.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            tuple: Model outputs, probabilities, predicted label, and confidence score.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the predicted sentiment
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get the corresponding class label
        raw_label = self.class_labels[predicted_class]

        # Map the label to standard format (POS, NEU, NEG)
        predicted_label = LABEL_MAPPING.get(raw_label.lower(), raw_label)

        return outputs, probabilities, predicted_label, probabilities[0][predicted_class].item()