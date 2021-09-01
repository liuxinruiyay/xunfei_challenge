import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from mymodel import BERT, CustomBERTModel
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from load_dataset import PaperTestDataset, create_test_data_loader
import warnings
warnings.filterwarnings('ignore')
import pynvml

def test_file_preprocess(file_path):
    file_handle = open(file_path,'r')
    lines = file_handle.readlines()

    tmpstr = ""
    tmpstr_list = []
    for line in lines:
        if line.find("test_") != -1:
            tmpstr_list.append(tmpstr)
            tmpstr = ""
            line = line.replace('\n',' ')
            tmpstr = tmpstr + line
        else:
            line = line.replace('\n',' ')
            tmpstr = tmpstr + line
    tmpstr_list.append(tmpstr)

    paperid_list = []
    title_list = []
    abstract_list = []
    tmpstr_list = tmpstr_list[1:]
    # print(tmpstr_list)
    for str in tmpstr_list:
        tmp = str.split('\t')
        paper_id = tmp[0].strip(' ')
        title = tmp[1]
        abstract = tmp[2]
        paperid_list.append(paper_id)
        title_list.append(title)
        abstract_list.append(abstract)
    return paperid_list, title_list, abstract_list


file_path = "./paper_data/test.csv"
paperid_list = test_file_preprocess(file_path)[0]
title_list = test_file_preprocess(file_path)[1]
abstract_list = test_file_preprocess(file_path)[2]
# print(len(paperid_list))
# 将title和abstract进行拼接，中间用句号隔开
text_list = []
for index in range(0,len(title_list)):
    tmp = title_list[index].replace('"','') + '. '+ abstract_list[index].replace('"','')
    text_list.append(tmp)
data_frame = pd.DataFrame(paperid_list)
data_frame = data_frame.rename(columns={0:'paperid'})
data_frame['text'] = text_list
# print(data_frame['text'][0])


# Model parameter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_data_loader = create_test_data_loader(data_frame, tokenizer, MAX_SEQ_LEN, batch_size=32)


def evaluate(model, test_data_loader):
    total_out = None
    model.eval()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    for batch in test_data_loader:
        titletext = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = torch.randint(0,1,(len(batch['input_ids']),),dtype=torch.long).to(device)
        # print("sssss")
        
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.used/1024**2)
        # # print("before:"+str(before))
        output = model(texts=titletext, attn_mask=attention_mask)
        # # print(type(model))
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print(meminfo.used/1024**2)
        # # print("after:"+str(after))
        
        logits = output.cpu().detach() # logits:(batch_size*39)
        del output
        #y_pred.extend(torch.argmax(logits, 1).tolist())
        if total_out is None:
            total_out = logits
        else:
            total_out = torch.cat((total_out, logits),axis=0)
    return total_out

def evaluate_k_fold(model_list, test_data_loader):
    total_out = None
    for model in model_list:
        load_path = "./pretrain_oversample_lstm/"+model
        state_dict = torch.load(load_path, map_location=device)
        print(f'Model loaded from <== {load_path}')
        model = CustomBERTModel().to(device)
        model.load_state_dict(state_dict['model_state_dict'])
        if total_out is None:
            k_fold_out = evaluate(model, test_data_loader)/10
            total_out = k_fold_out
        else:
            k_fold_out = evaluate(model, test_data_loader)/10
            total_out = total_out + k_fold_out
    return total_out

model_list = ["0-model-0.8202038924930491.pt", "1-model-0.8192771084337349.pt", "2-model-0.8099740452354468.pt", "3-model-0.8103448275862069.pt", "4-model-0.8146088246199481.pt", "5-model-0.8190582128290693.pt", "6-model-0.8147942157953282.pt", "7-model-0.8086763070077865.pt", "8-model-0.818131256952169.pt", "9-model-0.8151649981460882.pt"]
total_out = evaluate_k_fold(model_list, test_data_loader)
print(total_out.shape)
total_out = total_out.numpy()
#total_out = old_total_out + total_out
y_pred = np.argmax(total_out,axis=-1)

data_frame['label_num'] = y_pred
print(data_frame['text'][:5])
print(y_pred[:5])
print(data_frame['label_num'][:5])
data_frame.to_csv('./processed_data/submission_unprocessed.csv', mode='a', header=True, index=None)
print(data_frame.shape)


