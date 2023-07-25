import pandas as pd
import os
from config import Config 
from torch.utils.data import TensorDataset
import torch
from transformers import BertTokenizer

"""
数据预处理,分成K折,保存在k_folds_dir
"""
def preprocess_data(data_dir1, data_dir2,k_folds_dir,folds):
    # 从CSV文件中读取数据
    train_pro = pd.read_csv(data_dir1)
    train_sum = pd.read_csv(data_dir2)
    train_data = train_pro.merge(train_sum, on="prompt_id")

    # 根据prompt_id分成4折
    id2fold = {
        "814d6b": 0,
        "39c16e": 1,
        "3b9047": 2,
        "ebad26": 3,
    }
    # 将"prompt_id" 列通过字典 "id2fold" 进行映射
    train_data["fold"] = train_data["prompt_id"].map(id2fold)
    # 丢弃两个没用的数据
    train_data.drop(["prompt_id", "student_id"], axis=1, inplace=True)

    os.makedirs(k_folds_dir, exist_ok=True)
    # 将数据根据 "fold" 列拆分为四个 DataFrame，并保存为四个 CSV 文件
    for fold in range(folds):
        fold_data = train_data[train_data["fold"] == fold]
        output_file = f"{k_folds_dir}/fold_{fold}.csv"
        fold_data.to_csv(output_file, index=False)


"""
用分词器对csv文件进行tokenize并保存Pt文件
"""
def tokenize_and_save_csv (tokenizer,k_folds_dir,k_folds_pt_dir,folds) :
    for fold in range(folds):
        file_name = f"{k_folds_dir}/fold_{fold}.csv"
        file = pd.read_csv(file_name)

        sentences = []
        labels_content = []
        labels_wording = []

        # 分词和处理数据
        for index in range(1,file.shape[0]):
            sent = ""
            for column in ["text","prompt_question"]:
                sent += str(file[column][index])
            sentences.append(sent)

            for column in ["content", "wording"]:
                if column == "content":
                    labels_content.append(file[column][index])
                if column == "wording":
                    labels_wording.append(file[column][index])
            # 对句子进行分词，并创建输入张量
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
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

        dataset = TensorDataset(input_ids, attention_masks, labels_content, labels_wording)     
        
        os.makedirs(k_folds_pt_dir, exist_ok=True)      
        torch.save(dataset, f"{k_folds_pt_dir}/fold_{fold}.pt")

# k_folds_dir = "data/k_folds"
# k_folds_pt_dir= "data/k_folds_pt"
# tokenizer = BertTokenizer.from_pretrained('./bert', do_lower_case=True)
#tokenize_and_save_csv(tokenizer,k_folds_dir,k_folds_pt_dir) 
#preprocess_data("data\prompts_train.csv","data\summaries_train.csv","data/k_folds")