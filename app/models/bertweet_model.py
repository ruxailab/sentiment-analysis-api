"""
This module defines the BertweetSentiment class, which is a PyTorch model for sentiment analysis using the Bertweet model.
"""
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertweetSentiment(nn.Module):
    def __init__(self,config: dict)->None:
        """
        Initialize the Bertweet model for sentiment analysis.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

        self.config = config.get('sentiment_analysis').get('bertweet')
        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        super(BertweetSentiment, self).__init__()
        # Initialize the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initialize the Model
        self.model= AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

        # Load the model configuration to get class labels
        self.model_config = self.model.config

        # Get Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

    def forward(self,texts)->list:
        """
        Perform sentiment analysis on the given text.

        Args:
            texts (str): Input text for sentiment analysis.

        Returns:
            list: A list of dictionaries containing text, label, and confidence score.
        """

        # Handle single string input for consistency
        if isinstance(texts, str):
            texts = [texts]

        # Process as batch with padding and truncation (The Core Fix)
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Get the highest probability and its index
        confidences, predicted_classes = torch.max(probabilities, dim=1)

        results = []
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

#     test_texts = [
#             "I love the new features of the app!",
#             "I hate the new features of the app!",
#             "Hi how are u?"
#         ]

#     print(model(test_texts))
# # Run:
# # python -m app.models.bertweet_model