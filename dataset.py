import pandas as pd
import os
from config import Config 
from torch.utils.data import TensorDataset,ConcatDataset
import torch
from transformers import BertTokenizer

def load_data(kfold,folds,k_folds_pt_dir) :
    train_dataset = []
    for fold in range(folds) :
        file_path = f"{k_folds_pt_dir}/fold_{fold}.pt"
        if fold == kfold :
            val_dataset=torch.load(file_path)
        else : 
            data = torch.load(file_path)
            train_dataset.append(data)
    train_dataset = ConcatDataset(train_dataset)
    print(len(train_dataset))
    print(len(val_dataset))
    print(train_dataset[0])
    print(val_dataset[0])
    return train_dataset,val_dataset
