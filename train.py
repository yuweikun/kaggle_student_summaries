import torch 
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from preprocess import preprocess_data
from model import BertForRegression
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import numpy as np

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

data_dir1 = "./data/prompts_train.csv"
data_dir2 = "./data/summaries_train.csv"
train_save_file = "./data/train_dataset.pt"
val_save_file = "./data/val_dataset.pt"

batch_size = 16
train_dataset,val_dataset = preprocess_data(data_dir1,data_dir2,train_save_file,val_save_file)
#train_dataset,val_dataset = preprocess_data(data_dir1,data_dir2)
# 为训练和验证集创建 Dataloader，对训练样本随机洗牌
train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = batch_size, # 以小批量进行训练
        )

# 验证集不需要随机化，这里顺序读取就好
validation_dataloader = DataLoader(
            val_dataset, # 验证样本
            sampler = SequentialSampler(val_dataset), # 顺序选取小批量
            batch_size = batch_size 
        )

bert_model_name='bert-base-uncased'
model=BertForRegression(bert_model_name)
# 在 gpu 中运行该模型
model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
num_epochs = 5

# 数据处理和加载
train_texts = []
train_content_scores = []
train_wording_scores = []

def train_model(model, train_dataloader, validation_dataloader,optimizer, criterion, num_epochs):
    best_mcrmse = float('inf')
    model.train()
    for epoch in range(num_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
        print('Training...')
        # 统计单次 epoch 的训练时间
        t0 = time.time()
        # 重置每次 epoch 的训练总 loss
        total_loss = 0.0
        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels_content, labels_wording = [item.to(device) for item in batch]
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # 计算单个目标值的损失
            pred_content, pred_wording = logits[:, 0], logits[:, 1]
            loss_content = criterion(pred_content, labels_content)
            loss_wording = criterion(pred_wording, labels_wording)
            
            # 合并两个目标值的损失
            loss = loss_content + loss_wording
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)        
        avg_loss = total_loss / len(train_dataloader)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_loss))
        print("  Training epcoh took: {:}".format(training_time))

        total_eval_loss = 0
        content_rmse_sum = 0
        wording_rmse_sum = 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            input_ids, attention_mask, labels_content, labels_wording = [item.to(device) for item in batch] 
            with torch.no_grad():  
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_content, pred_wording = logits[:, 0], logits[:, 1]
            loss_content = criterion(pred_content, labels_content)
            loss_wording = criterion(pred_wording, labels_wording)
            # 合并两个目标值的损失
            loss = loss_content + loss_wording
            total_eval_loss += loss.item()

            # Convert predictions and labels to NumPy arrays on CPU
            pred_content = pred_content.cpu().numpy()
            pred_wording = pred_wording.cpu().numpy()
            gt_content = labels_content.cpu().numpy()
            gt_wording = labels_wording.cpu().numpy()

            # Calculate RMSE for content and wording
            content_rmse = np.sqrt(((pred_content - gt_content)**2).mean())
            wording_rmse = np.sqrt(((pred_wording - gt_wording)**2).mean())

            content_rmse_sum += content_rmse
            wording_rmse_sum += wording_rmse

        avg_eval_loss = total_eval_loss / len(validation_dataloader)
        avg_content_rmse = content_rmse_sum / len(validation_dataloader)
        avg_wording_rmse = wording_rmse_sum / len(validation_dataloader)
        mcrmse = (avg_content_rmse + avg_wording_rmse) / 2
        print("  Average evaluation loss: {:.4f}".format(avg_eval_loss))
        print("  Average content RMSE: {:.4f}".format(avg_content_rmse))
        print("  Average wording RMSE: {:.4f}".format(avg_wording_rmse))
        print("  MCRMSE: {:.4f}".format(mcrmse))
        # 保存效果最好的模型
        if mcrmse < best_mcrmse:
            best_mcrmse = mcrmse
            torch.save(model.state_dict(), "best_model.pt")

# 开始训练
train_model(model, train_dataloader,validation_dataloader,optimizer, criterion, num_epochs)