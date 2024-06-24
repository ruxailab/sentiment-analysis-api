import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import config, id2label


class RoberteaSentiment(nn.Module):
    def __init__(self):
        super(RoberteaSentiment, self).__init__()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("tae898/emoberta-base")

        # Model
        self.model= AutoModelForSequenceClassification.from_pretrained("tae898/emoberta-base")

        
    def forward(self,text):

        # print(text)

        # # Tokenize the input text
        # # inputs = self.tokenizer(text, truncation=True,return_tensors="pt")
        # tokens = self.tokenizer(text, truncation=True,padding='max_length', max_length=128)
        
        # tokens = self.tokenizer("I am Happy.I am Sad") #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
        # tokens = self.tokenizer("I am Happy.I am Sad",padding=True, truncation=True, max_length=256) #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
        tokens = self.tokenizer(("I am Happy.","I am Sad.")) #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
        print(tokens)
        

        tokens["input_ids"] = torch.tensor(tokens["input_ids"]).view(1, -1).to(config['training']['device'])
        tokens["attention_mask"] = (torch.tensor(tokens["attention_mask"]).view(1, -1).to(config['training']['device']))


        print(tokens["input_ids"].shape)
        print(tokens["attention_mask"].shape)
        # Forward pass
        outputs = self.model(**tokens)


        outputs = torch.softmax(outputs["logits"].detach().cpu(), dim=1).squeeze().numpy()
        print(outputs)




        # outputs = {id2emotion[idx]: prob.item() for idx, prob in enumerate(outputs)}
        # outputs = {print(idx,prob)for idx, prob in enumerate(outputs)}

        # get arg max
        # arg_max = torch.argmax(outputs).item()

        # get the label
        # label = id2label[arg_max]

        # print(label)
        # print(outputs)






# from torchsummary import summary
# model=RoberteaSentiment()
# print(model.model)
# summary(model, (16, 3, 224, 224))  /