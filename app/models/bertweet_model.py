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

    def forward(self,text)->tuple:
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
        predicted_label = self.class_labels[predicted_class]

        return outputs, probabilities, predicted_label, probabilities[0][predicted_class].item()
    
    def batch_forward(self, texts: list) -> list:
        """
        Perform sentiment analysis on a list of texts in batch.

        Args:
            texts (list): List of input texts for sentiment analysis.
            batch_size (int, optional): Maximum number of texts to process in one forward pass.
                If None, it uses a default value from the configuration (or processes all if not set).

        Returns:
            list: A list of dictionaries with 'label' and 'confidence' for each text.
        """
        # batch_size get it from the configuration
        batch_size = self.config.get("batch_size", len(texts))
        
        results = []
        # If the number of texts exceeds the batch_size, split them
        if len(texts) > batch_size:
            for i in range(0, len(texts), batch_size):
                sub_texts = texts[i : i + batch_size]
                results.extend(self.batch_forward(sub_texts))
            return results

        # Otherwise, process the batch at once.
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for i in range(probabilities.size(0)):
            predicted_class = torch.argmax(probabilities[i]).item()
            predicted_label = self.class_labels[predicted_class]
            confidence = probabilities[i][predicted_class].item()
            results.append({
                "label": predicted_label,
                "confidence": confidence
            })
        return results


if __name__ == "__main__":
    config = {
        'debug': True,
        'sentiment_analysis': {
            'default_model': "bertweet",  # Specify the default sentiment analysis model (e.g., bertweet, another_model)
            'bertweet': {
                'model_name': "finiteautomata/bertweet-base-sentiment-analysis",
                'device': 'cpu'
            }
        }
    }
    print("config",config)
    model = BertweetSentiment(config)
#     print("model",model)
#     print("model.class_labels",model.class_labels)

    text = "I love the new features of the app!"
    print(model(text))

    # text = "I hate the new features of the app!"
    # print(model(text))

    # text = "Hi how are u?"
    # print(model(text))

# # Run:
# # python -m app.models.bertweet_model