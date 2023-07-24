import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import SequentialSampler

def predict(model, data_dir1, data_dir2, output_file):
    # 加载测试数据集
    test_pro = pd.read_csv(data_dir1)
    test_sum = pd.read_csv(data_dir2)
    test_data = test_pro.merge(test_sum, on="prompt_id")
    test_data.drop(["prompt_id"], axis=1, inplace=True)

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bert-pretrain/bert', do_lower_case=True)
    # 准备数据进行预测
    sentences = []
    for index in range(test_data.shape[0]):
        sent = ""
        for column in ["text", "prompt_text", "prompt_title", "prompt_question"]:
            sent += str(test_data[column][index])
        sentences.append(sent)
       
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
    
    
    #小批量
    batch_size =16
    predict_dataset = TensorDataset(input_ids,attention_masks)
    dataloader = DataLoader(predict_dataset,
                           sampler =SequentialSampler(predict_dataset),
                           batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 定义空列表存储预测结果
    pred_content = []
    pred_wording = []
    # 进行预测
    model.eval()
    for idx,batch in enumerate(dataloader):
        input_ids, attention_masks = [item.to(device) for item in batch]   
        with torch.no_grad(): 
            logits = model(input_ids=input_ids, attention_mask=attention_masks)
            # 获取 content 和 wording 的预测结果
            pred_content.append(logits[:, 0].cpu().numpy())
            pred_wording.append(logits[:, 1].cpu().numpy())
            
    # 合并预测结果为一个完整的numpy数组
    pred_content = np.concatenate(pred_content)
    pred_wording = np.concatenate(pred_wording)
    
    # 创建 DataFrame 保存预测结果
    submission_df = pd.DataFrame({
        'student_id': test_data["student_id"],
        'content': pred_content,
        'wording': pred_wording
    })

    # 保存预测结果到输出文件
    submission_df.to_csv(output_file, index=False)
    print("ok")
if __name__ == "__main__":
    data_dir1 = "/kaggle/input/commonlit-evaluate-student-summaries/prompts_test.csv"
    data_dir2 = "/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv"
    output_file = "./submission.csv"

    bert_model_name = 'bert-base-uncased'
    model = BertForRegression(bert_model_name)
    model.load_state_dict(torch.load("/kaggle/input/student/best_model.pt"))
    model.cuda()

    predict(model, data_dir1, data_dir2, output_file)