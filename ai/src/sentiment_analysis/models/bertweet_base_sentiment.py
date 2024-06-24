import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import config, id2label,num_classes


class BertTweetBaseSentiment(nn.Module):
    def __init__(self):
        super(BertTweetBaseSentiment, self).__init__()

        # Model
        # self.model= AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
        self.model= AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=num_classes)

        
    def forward(self,input_ids, attention_mask):
        # Forward pass
        output=self.model(input_ids=input_ids, attention_mask=attention_mask)    

        # Get logits
        y_logits=output.logits

        # Apply Sigmoid
        y_scores=torch.sigmoid(y_logits)  # shape: (batch_size, num_classes)

        return y_logits,y_scores


if __name__ == "__main__":
    model=BertTweetBaseSentiment()
    print(model.model)
    # summary(model, (16, 3, 224, 224))  #/
