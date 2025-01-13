import pandas as pd
import torch
from torch.utils.data import Dataset

# from src.data_loaders.custom_augmentation import CustomAugmentation


class MELDDataset(Dataset):
    def __init__(self, csv_path):
        # Read CSV
        self.data = pd.read_csv(csv_path,nrows=100)

        # self.transform = CustomAugmentation()


        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get utterance
        utterance = self.data.loc[idx, "Utterance"]

        # Get Label
        # label = self.data.loc[idx, "Sentiment"]
        label = self.data.loc[idx, "Emotion"]  

        return utterance, label
        
        
        # # Get the input text and label
        # text = self.data.loc[idx, "text"]
        # label = self.data.loc[idx, "emotion"]

        # # Encode the text
        # encoding = self.tokenizer.encode_plus(
        #     text,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     return_token_type_ids=False,
        #     padding="max_length",
        #     return_attention_mask=True,
        #     return_tensors="pt",
        #     truncation=True,
        # )

        # return {
        #     "text": text,
        #     "input_ids": encoding["input_ids"].flatten(),
        #     "attention_mask": encoding["attention_mask"].flatten(),
        #     "label": torch.tensor(label, dtype=torch.long),
        # }