import pandas as pd
import torch
import os
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer

def preprocess_data(data_dir1, data_dir2, train_save_file=None, val_save_file=None):
    if train_save_file is not None and os.path.exists(train_save_file):
        train_dataset = torch.load(train_save_file)
    else:
        # 从CSV文件中读取数据
        train_pro = pd.read_csv(data_dir1)
        train_sum = pd.read_csv(data_dir2)
        train_data = train_pro.merge(train_sum, on="prompt_id")
        train_data.drop(["prompt_id", "student_id"], axis=1, inplace=True)

        # 初始化分词器
        tokenizer = BertTokenizer.from_pretrained('./bert', do_lower_case=True)
        sentences = []
        labels_content = []
        labels_wording = []
        # 分词和处理数据
        for index in range(train_data.shape[0]):
            sent = ""
            for column in ["text", "prompt_text", "prompt_title", "prompt_question"]:
                sent += str(train_data[column][index])
            sentences.append(sent)

            for column in ["content", "wording"]:
                if column == "content":
                    labels_content.append(train_data[column][index])
                if column == "wording":
                    labels_wording.append(train_data[column][index])

        # 对句子进行分词，并创建输入张量
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels_content = torch.tensor(labels_content, dtype=torch.float32)
        labels_wording = torch.tensor(labels_wording, dtype=torch.float32)

        train_dataset = TensorDataset(input_ids, attention_masks, labels_content, labels_wording)

        if train_save_file is not None:
            torch.save(train_dataset, train_save_file)

    if val_save_file is not None and os.path.exists(val_save_file):
        val_dataset = torch.load(val_save_file)
    else:
        # 将训练数据集拆分为训练集和验证集
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        _, val_dataset = random_split(train_dataset, [train_size, val_size])

        if val_save_file is not None:
            torch.save(val_dataset, val_save_file)

    return train_dataset, val_dataset
