import torch
import torch.nn as nn
from transformers import BertModel

class BertForRegression(nn.Module):
    def __init__(self, bert_model_name, hidden_size=768, num_labels=2):
        super(BertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained('./bert',mirror='tuna')
        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.regression_head(pooled_output)
        return logits
    