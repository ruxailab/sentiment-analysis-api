import torch
from tqdm import tqdm  # Import the tqdm function
from torch.utils.data import  DataLoader

from src.utils import config
from src.data_loaders.dataset import MELDDataset
from src.models.roberta_sentiment import RoberteaSentiment

class Trainer():
    def __init__(self,model:RoberteaSentiment,csv_path:str):
        # Model to be trained
        self.model=model
        self.model.to(config['training']['device'])

        # Load the dataset
        self.dataset = MELDDataset(csv_path)
        # Create a DataLoader
        self.train_data_loader = DataLoader(dataset=self.dataset, batch_size=config['training']['batch_size'], shuffle=True)


    def train(self):
        for epoch in range(config['training']['num_epochs']):
            for batch in tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}"):
                utterance, label = batch
                # print(utterance,label)

                # utterance=utterance.to(config['device'])

                # Forward pass
                self.model(utterance)


                break

                
    

if __name__ == "__main__":
    # Create an instance of the model
    model = RoberteaSentiment()

    Trainer(model,config['data']['train_file']).train()
# PS D:\sentiment-analysis-api> python -m src.trainer.trainer