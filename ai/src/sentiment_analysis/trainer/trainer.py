import torch
from tqdm import tqdm  # Import the tqdm function
from torch.utils.data import  DataLoader

from src.utils import config
from src.data_loaders.dailydialgoue_dataset import DailyDialogueDataSet
from src.models.bertweet_base_sentiment import BertTweetBaseSentiment
import gc

def collate_fn(batch):

    # utterances_count = torch.tensor([item['utterances_count'] for item in batch])
    # utterances_count = torch.tensor([item['utterances_count'] for item in batch])
    # input_ids = torch.stack([item['input_ids'] for item in batch])
    # attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # labels = torch.stack([item['labels'] for item in batch])
    # actions = torch.stack([item['actions'] for item in batch])
    # print(utterances_count)

    # for item in batch:
    #     print(item['utterances_count'])

    utterances_count=[]
    input_ids=[]
    attention_mask=[]
    labels=[]
    actions=[]

    for item in batch:
        for i in range(item['utterances_count']):
            utterances_count.append(1)
            input_ids.append(item['input_ids'][i])
            attention_mask.append(item['attention_mask'][i])
            labels.append(item['labels'][i])
            actions.append(item['actions'][i])

    utterances_count = torch.tensor(utterances_count)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    actions = torch.stack(actions)

    # print(utterances_count.shape) # torch.Size([17])
    # print(input_ids.shape) # torch.Size([17, 128])
    # print(attention_mask.shape) # torch.Size([17, 128])
    # print(labels.shape) # torch.Size([17])
    # print(actions.shape) # torch.Size([17])

    return utterances_count,input_ids, attention_mask, labels, actions

class Trainer():
    def __init__(self,model:BertTweetBaseSentiment,csv_path:str):
        # Model to be trained
        self.model=model
        self.model.to(config['training']['device'])

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['training']['learning_rate'])

        # Learning rate scheduler

        # Criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # Load the dataset
        self.dataset = DailyDialogueDataSet(csv_path)

        # Create a DataLoader
        self.train_data_loader = DataLoader(dataset=self.dataset, batch_size=config['training']['batch_size'], shuffle=True,collate_fn=collate_fn)


    def train(self):
        print("Training ...")
        for epoch in range(config['training']['num_epochs']):
            # Set the model to training mode
            self.model.train()

            # for utterances_count,input_ids, attention_mask, labels, actions in tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
            for utterances_count,input_ids, attention_mask, labels, actions in self.train_data_loader:
                # Move the input tensors to the device
                input_ids = input_ids.to(config['training']['device'])
                attention_mask = attention_mask.to(config['training']['device'])
                labels = labels.to(config['training']['device'])

                # Forward pass
                # y_pred,_=self.model(input_ids, attention_mask)
                y_logits,_=self.model(input_ids, attention_mask)
                print(labels.shape)
                print(y_logits.shape)

                # # Ensure labels are within the valid range
                # num_classes = y_logits.size(1)
                # assert labels.max().item() < num_classes, f"Label index out of range: {labels.max().item()} >= {num_classes}"
                # assert labels.min().item() >= 0, f"Negative label index: {labels.min().item()}"
                

                # # Compute the loss    
                # loss = self.criterion(y_logits,labels)
                # print(loss)

                # # Backward pass
                # loss.backward()

                # # Update the weights
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                # del loss
                # del input_ids,attention_mask,labels

                # torch.cuda.empty_cache()
                # gc.collect()  

    



                        
            # # Convert logits to probabilities
            # probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # del input_ids,attention_mask

                
    

if __name__ == "__main__":
    # Create an instance of the model
    model = BertTweetBaseSentiment()
    # print(model)

    Trainer(model,config['data']['train_file']).train()
# PS D:\sentiment-analysis-api> python -m src.trainer.trainer