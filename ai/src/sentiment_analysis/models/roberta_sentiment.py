import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from src.utils import config, id2label


class RobertaSentiment(nn.Module):
    def __init__(self):
        super(RobertaSentiment, self).__init__()

        model_name="finiteautomata/bertweet-base-sentiment-analysis"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model
        self.model= AutoModelForSequenceClassification.from_pretrained(model_name)

        # Load the model configuration to get class labels
        self.config = self.model.config

        # Get Labels
        if hasattr(self.config, 'id2label'):
            self.class_labels = [self.config.id2label[i] for i in range(len(self.config.id2label))]
        else:
            self.class_labels = None

        
    def forward(self,text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Forward pass
        outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the predicted sentiment
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get the corresponding class label
        predicted_label = self.class_labels[predicted_class]

        return outputs,probabilities,predicted_label, probabilities[0][predicted_class].item()