import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import pandas as pd

class DailyDialogueDataSet(Dataset):
    def __init__(self, csv_path):
        # Read CSV
        self.data = pd.read_csv(csv_path)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')

        self.max_model_input_size = self.tokenizer.model_max_length


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get utterance
        utterances = self.data.loc[idx, "text"]
        # Split the utterances by __eou__ and remove the last empty string
        utterances = utterances.split("__eou__")[:-1]
        # print(utterances)
 
        # Get Label
        labels = self.data.loc[idx, "emotion"]
        # Split the labels by space and convert them to integers
        labels = list(map(int, labels.split()))
        # print(labels)

        # Get actions
        actions = self.data.loc[idx, "act"]
        # Split the actions by space and convert them to integers
        actions = list(map(int, actions.split()))


        input_ids=[]
        attention_mask=[]

        for utterance in utterances:
            input_ids_attention_mask = self.tokenizer(utterance, padding='max_length', max_length=self.max_model_input_size, truncation=True)
            input_ids.append(input_ids_attention_mask["input_ids"])
            attention_mask.append(input_ids_attention_mask["attention_mask"])
       
    
        return {
                'utterances_count': len(utterances),
                'input_ids': torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                'labels': torch.tensor(labels, dtype=torch.long),
                'actions': torch.tensor(actions,dtype=torch.long)
            }
    

if __name__ == "__main__":
    dataset=DailyDialogueDataSet("./data/dailydialog/train/train.csv")

    for example in dataset:
        assert example['utterances_count']==example['attention_mask'].shape[0]==len(example['labels'])==len(example['actions']), "Lengths of utterances, labels and actions should be the same"
        break
        # print(example)

# PS D:\sentiment-analysis-api> python -m src.data_loaders.dailydialgoue_dataset 