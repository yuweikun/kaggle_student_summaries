import torch 
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from preprocessV2 import preprocess_data,tokenize_and_save_csv
import argparse
import os
from model import ModelForRegression
from dataset import load_data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding,
)
from config import Config
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

def train_model(model, optimizer, criterion, num_epochs, folds,input_dir,checkpoints_dir,batch_size):
    
    # 初始化 total_mcrmse 用于累加每个折叠的 mcrmse
    total_mcrmse = 0
    os.makedirs(checkpoints_dir, exist_ok=True)  
    # 对每个折叠进行循环
    for fold in range(folds):
        train_dataset,val_dataset = load_data(fold,folds,input_dir)

        # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
        train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = batch_size, # 以小批量进行训练
            drop_last=True
        )

        # 验证集不需要随机化，这里顺序读取就好
        validation_dataloader = DataLoader(
            val_dataset, # 验证样本
            sampler = SequentialSampler(val_dataset), # 顺序选取小批量
            batch_size = batch_size ,
            drop_last=True
        )
        print("")
        print('======== Folds {:} / {:} ========'.format(fold + 1, folds))
        # 在当前折叠上进行训练和评估，并得到 mcrmse
        mcrmse = train_single_fold(model, train_dataloader, validation_dataloader, optimizer, criterion, num_epochs, checkpoints_dir,fold)

        # 累加当前折叠的 mcrmse 到 total_mcrmse
        total_mcrmse += mcrmse

    # 计算平均 CV 得分
    cv_score = total_mcrmse / folds
    print("CV Score:", cv_score)


def train_single_fold(model, train_dataloader, validation_dataloader,optimizer, criterion, num_epochs,checkpoints_dir,fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            torch.save(model.state_dict(), f"{checkpoints_dir}/best_model_{fold}")

    return best_mcrmse
def main():
    # 创建 ArgumentParser 并添加参数
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    # parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    # parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length")
    # parser.add_argument("--fold", type=int, default=None, help="Fold")
    # 解析命令行参数并更新配置参数
    args = parser.parse_args()

    
    CFG = Config(**vars(args))
    preprocess_data(CFG.data_dir1,CFG.data_dir2,CFG.k_folds_dir,CFG.folds)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name_or_path)

    tokenize_and_save_csv(tokenizer,CFG.k_folds_dir,CFG.k_folds_pt_dir,CFG.folds)

    model = ModelForRegression(CFG.model_name_or_path,CFG.hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate)
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 开始训练
    train_model(model, optimizer, criterion, CFG.num_train_epochs,CFG.folds,CFG.k_folds_pt_dir,CFG.checkpoints_dir,CFG.batch_size)

if __name__ == "__main__":
    main()