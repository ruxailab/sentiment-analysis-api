import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RoberteaSentiment(nn.Module):
    def __init__(self):
        super(RoberteaSentiment, self).__init__()

        # Enc

        self.model= AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base")
        # self.model = RobertaModel.from_pretrained(model_name)
        # self.drop = nn.Dropout(p=0.3)
        # self.out = nn.Linear(self.model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        pass
        # _, pooled_output = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # )
        # output = self.drop(pooled_output)
        # return self.out(output)


# from torchsummary import summary
# model=RoberteaSentiment()
# print(model.model)
# summary(model, (16, 3, 224, 224))  /