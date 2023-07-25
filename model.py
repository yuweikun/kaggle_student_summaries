import torch
import torch.nn as nn
from transformers import BertModel,AutoModel
import torch.nn.functional as F

class ModelForRegression(nn.Module):
    def __init__(self, bert_model_name,hidden_dim=200):
        super(ModelForRegression, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name,mirror='tuna')
        self.fc1=nn.Linear(768,hidden_dim)
        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.fc2=nn.Linear(hidden_dim,2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x=self.fc1(outputs.last_hidden_state[:,0,:])
        x=self.bn1(F.relu(x))
        x=self.fc2(x)
        return x
