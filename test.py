import pandas as pd
from transformers import BertTokenizer
import tqdm
from torch.utils.data import TensorDataset, random_split
import multiprocessing
import os
import torch
from torch import nn
from d2l import torch as d2l


def Dataset(data_dir1,data_dir2,kfold_id=None,kfold_num=None,is_train=True):

    train_pro = pd.read_csv(data_dir1)
    # print(len(train_pro))
    train_sum = pd.read_csv(data_dir2)
    # print(len(train_sum))
    train_data = train_pro.merge(train_sum, on="prompt_id")
    train_data.drop(["prompt_id", "student_id"], axis=1, inplace=True)

    # print(train_data.shape[0])


    prompt_all=[]
    sentences = []
    labels_content = []
    labels_wording = []
    # max_len = 0

    for index in tqdm.tqdm(range(train_data.shape[0]), total=train_data.shape[0]):

        for column in ["text"]:
            sent = str(train_data[column][index])
            sentences.append(sent)

        prompt=''
        for column in ["prompt_question", "prompt_title", "prompt_text"]:
            prompt+=str(train_data[column][index])
        prompt_all.append(prompt)

        for column in ["content", "wording"]:
            if column == "content":
                labels_content.append(train_data[column][index])
            if column == "wording":
                labels_wording.append(train_data[column][index])
    if kfold_id is not None and kfold_num is not None:
        avg_len=len(sentences)//kfold_num

        val_sentences=sentences[kfold_id*avg_len:(kfold_id+1)*avg_len]
        val_prompt_all=prompt_all[kfold_id*avg_len:(kfold_id+1)*avg_len]
        val_label_content=labels_content[kfold_id*avg_len:(kfold_id+1)*avg_len]
        val_label_wording=labels_wording[kfold_id*avg_len:(kfold_id+1)*avg_len]
        # val_dataset=[val_sentences,val_prompt_all,val_label_content,val_label_wording]

        train_sentences=sentences[:kfold_id*avg_len]+sentences[(kfold_id+1)*avg_len:]
        train_prompt_all=prompt_all[:kfold_id*avg_len]+prompt_all[(kfold_id+1)*avg_len:]
        train_label_content=labels_wording[:kfold_id*avg_len]+labels_content[(kfold_id+1)*avg_len:]
        train_label_wording=labels_wording[:kfold_id*avg_len]+labels_wording[(kfold_id+1)*avg_len:]
        # train_dataset=[train_sentences,train_prompt_all,train_label_content,train_label_wording]
    else:
        train_len=int(0.8*len(sentences))
        train_sentences = sentences[:train_len]
        train_prompt_all = prompt_all[:train_len]
        train_label_content = labels_wording[:train_len]
        train_label_wording = labels_wording[:train_len]
        # train_dataset = [train_sentences, train_prompt_all, train_label_content, train_label_wording]

        val_sentences = sentences[train_len:]
        val_prompt_all = prompt_all[train_len:]
        val_label_content = labels_content[train_len:]
        val_label_wording = labels_wording[train_len:]

    if is_train:
        return train_sentences,train_prompt_all,train_label_content,train_label_wording
    else:
        return val_sentences,val_prompt_all,val_label_content,val_label_wording
# data_dir1='prompts_train.csv'
# data_dir2='summaries_train.csv'
# data=Dataset(data_dir1,data_dir2,is_train=True)
# print(len(data[0]))


class All_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, tokenizer=None):
        self.tokenizer = tokenizer
        sentences=[self.tokenizer.tokenize(sentence.lower()) for sentence in dataset[0]]
        prompts=[self.tokenizer.tokenize(prompt.lower()) for prompt in dataset[1]]

        # all_text_prompt_tokens=[[p_tokens, h_tokens] for p_tokens, h_tokens in zip(*[self.tokenizer.tokenize([s.lower() for s in sentences])for sentences in dataset[:2]])]
        # self.labels = torch.tensor(dataset[2])
        all_text_prompt_tokens=[[sentence,prompt] for sentence,prompt in zip(*(sentences,prompts))]
        self.content=torch.tensor(dataset[2])
        self.wording=torch.tensor(dataset[3])
        # self.tokenizer = tokenizer
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_text_prompt_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        '''使句子对与max_len对齐'''
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = self._get_tokens_and_segments(p_tokens, h_tokens)
        ''''''
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.convert_token_to_ids('<pad>')] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def _get_tokens_and_segments(self,tokens_a,tokens_b=None):
        tokens = ['<cls>'] + tokens_a + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b is not None:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.content[idx],self.wording[idx]

    def __len__(self):
        return len(self.all_token_ids)


def load_data(data_dir1,data_dir2,batch_size, num_steps=512):

    num_workers = 4
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, mirror='tuna')
    train_data = Dataset(data_dir1,data_dir2, is_train=True)
    test_data = Dataset(data_dir1,data_dir2,is_train=False)
    print("111")
    train_set = All_Dataset(train_data, num_steps,tokenizer)
    print("222")
    test_set = All_Dataset(test_data, num_steps,tokenizer)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    print("333")
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter


data_dir1='data/csv/prompts_train.csv'
data_dir2='data/csv/summaries_train.csv'
train_iter,test_iter=load_data(data_dir1,data_dir2,64)

print(len(train_iter))
