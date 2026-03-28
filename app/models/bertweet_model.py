"""
This module defines the BertweetSentiment class, which is a PyTorch model for sentiment analysis using the Bertweet model.
"""
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any

class BertweetSentiment(nn.Module):
    def __init__(self,config: dict)->None:
        """
        Initialize the Bertweet model for sentiment analysis.
        :param config: The configuration object containing model and device info.
        """
        self.debug = config.get('debug')

         # ✅ Add null check
        sentiment_config = config.get('sentiment_analysis')
        if not sentiment_config:
            raise ValueError("'sentiment_analysis' not found in config")

        self.config = sentiment_config.get('bertweet')
        if not self.config:
            raise ValueError("'bertweet' not found in sentiment_analysis config")


        self.model_name = self.config.get('model_name')
        self.device = self.config.get('device')

        super(BertweetSentiment, self).__init__()
        # Initialize the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Initializing the Model
        self.model= AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)

        # Loading the model configuration to get class labels
        self.model_config = self.model.config

        # Geting the Labels
        if hasattr(self.model_config, 'id2label'):
            self.class_labels = [self.model_config.id2label[i] for i in range(len(self.model_config.id2label))]
        else:
            self.class_labels = None

    def forward(self,text)-> Dict[str, Any]:
        """
        Perform sentiment analysis on the given text.

        Args:
            text (str): Input text for sentiment analysis.

        Returns:
            Dict: Model outputs, probabilities, predicted label, and confidence score.
        """
        # Tokenizing  the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Converting logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # to get the predicted sentiment
        predicted_class = torch.argmax(probabilities, dim=1).item()

        
        # Converting it to the integer explicitly
        predicted_class = int(torch.argmax(probabilities, dim=1).item())

        # Adding a null check
        if self.class_labels is None:
           raise ValueError("Class labels not available")


        # Geting the corresponding class label
        predicted_label = self.class_labels[predicted_class]

        return {
    "logits": outputs.logits.tolist(),
    "probabilities": probabilities.tolist(),
    "label": predicted_label,
    "score": probabilities[0][predicted_class].item()
}


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