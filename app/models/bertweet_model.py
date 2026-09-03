"""
This module defines the BertweetSentiment class, which is a PyTorch model for sentiment analysis using the Bertweet model.
"""
import threading
from typing import List, Union

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertweetSentiment(nn.Module):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: dict = None):
        """Return the singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(BertweetSentiment, cls).__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: dict) -> None:
        """
        Initialize the Bertweet model for sentiment analysis.
        Heavy model weights are loaded only once for the process lifetime.
        :param config: The configuration object containing model and device info.
        """
        if self._initialized:
            return

        self.debug = config.get('debug')

        self.config = config.get('sentiment_analysis').get('bertweet')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        super(BertweetSentiment, self).__init__()
        # Initialize the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the Model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Load the model configuration to get class labels
        self.model_config = self.model.config

        # Get Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

        self._initialized = True

    def forward(self, text: Union[str, List[str]]):
        """
        Perform sentiment analysis on one text or a batch of texts.

        Args:
            text: Input text, or list of texts for batch inference.

        Returns:
            Single text: (outputs, probabilities, predicted_label, confidence).
            Batch: list of (predicted_label, confidence) tuples.
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else list(text)

        if not texts:
            return [] if not single_input else (None, None, None, None)

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=1)

            labels = []
            confidences = []
            for i in range(len(texts)):
                predicted_class = predicted_classes[i].item()
                labels.append(self.class_labels[predicted_class])
                confidences.append(probabilities[i][predicted_class].item())

        if single_input:
            return outputs, probabilities, labels[0], confidences[0]

        return list(zip(labels, confidences))


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
#     print("config",config)
#     model = BertweetSentiment(config)
#     print("model",model)
#     print("model.class_labels",model.class_labels)

#     text = "I love the new features of the app!"
#     print(model(text))

#     text = "I hate the new features of the app!"
#     print(model(text))

#     text = "Hi how are u?"
#     print(model(text))

# # Run:
# # python -m app.models.bertweet_model
