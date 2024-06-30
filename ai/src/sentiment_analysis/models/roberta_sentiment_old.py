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

#         # print(text)

#         # # Tokenize the input text
#         # # inputs = self.tokenizer(text, truncation=True,return_tensors="pt")
#         # tokens = self.tokenizer(text, truncation=True,padding='max_length', max_length=128)
        
#         # tokens = self.tokenizer("I am Happy.I am Sad") #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
#         # tokens = self.tokenizer("I am Happy.I am Sad",padding=True, truncation=True, max_length=256) #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
#         tokens = self.tokenizer(("I am Happy.","I am Sad.")) #{'input_ids': [0, 100, 524, 9899, 4, 100, 524, 10738, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
#         print(tokens)
        

#         tokens["input_ids"] = torch.tensor(tokens["input_ids"]).view(1, -1).to(config['training']['device'])
#         tokens["attention_mask"] = (torch.tensor(tokens["attention_mask"]).view(1, -1).to(config['training']['device']))


#         print(tokens["input_ids"].shape)
#         print(tokens["attention_mask"].shape)
#         # Forward pass
#         outputs = self.model(**tokens)


#         outputs = torch.softmax(outputs["logits"].detach().cpu(), dim=1).squeeze().numpy()
#         print(outputs)




#         # outputs = {id2emotion[idx]: prob.item() for idx, prob in enumerate(outputs)}
#         # outputs = {print(idx,prob)for idx, prob in enumerate(outputs)}

#         # get arg max
#         # arg_max = torch.argmax(outputs).item()

#         # get the label
#         # label = id2label[arg_max]

#         # print(label)
#         # print(outputs)






# # from torchsummary import summary
# # model=RoberteaSentiment()
# # print(model.model)
# # summary(model, (16, 3, 224, 224))  /