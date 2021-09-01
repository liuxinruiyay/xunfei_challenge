import torch
import pandas as pd
import torch.nn as nn
from mymodel import BERT, CustomBERTModel
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from load_dataset import PaperTestDataset, create_test_data_loader
import warnings
warnings.filterwarnings('ignore')

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

def evaluate(model, test_data_loader):
    y_pred = []
    model.eval()
    for batch in test_data_loader:
        titletext = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # labels = torch.randint(0,1,(len(batch['input_ids']),),dtype=torch.long).to(device)
        output = model(texts=titletext, attn_mask=attention_mask)
        logits = output
        y_pred.extend(torch.argmax(logits, 1).tolist())
    return y_pred

file_path = "./paper_data/test.csv"
paperid_list = test_file_preprocess(file_path)[0]
title_list = test_file_preprocess(file_path)[1]
abstract_list = test_file_preprocess(file_path)[2]

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
#print(test_data_loader)

# 读取模型参数
# load_path="./pretrain/model-0.8137164040778498.pt"
load_path="./pretrain_lstm/model-0.796.pt"
state_dict = torch.load(load_path, map_location=device)
print(f'Model loaded from <== {load_path}')
# best_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
#                                                           num_labels=39,
#                                                           output_hidden_states=False,
#                                                           output_attentions=False).to(device)
# # best_model = BertForSequenceClassification().to(device)
best_model = CustomBERTModel().to(device)
best_model.load_state_dict(state_dict['model_state_dict'])
y_pred = evaluate(best_model, test_data_loader)

data_frame['label_num'] = y_pred
print(data_frame['text'][:5])
print(y_pred[:5])
print(data_frame['label_num'][:5])
# data_frame.to_csv('./processed_data/submission.csv', mode='a', header=True, index=None)
# print(data_frame.shape)

label_dict = {'cs.CL': 0, 'cs.NE': 1, 'cs.DL': 2, 'cs.CV': 3, 'cs.LG': 4, 'cs.DS': 5, 'cs.IR': 6, 'cs.RO': 7, 'cs.DM': 8, 'cs.CR': 9, 'cs.AR': 10, 'cs.NI': 11, 'cs.AI': 12, 'cs.SE': 13, 'cs.CG': 14, 'cs.LO': 15, 'cs.SY': 16, 'cs.GR': 17, 'cs.PL': 18, 'cs.SI': 19, 'cs.OH': 20, 'cs.HC': 21, 'cs.MA': 22, 'cs.GT': 23, 'cs.ET': 24, 'cs.FL': 25, 'cs.CC': 26, 'cs.DB': 27, 'cs.DC': 28, 'cs.CY': 29, 'cs.CE': 30, 'cs.MM': 31, 'cs.NA': 32, 'cs.PF': 33, 'cs.OS': 34, 'cs.SD': 35, 'cs.SC': 36, 'cs.MS': 37, 'cs.GL': 38}
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]
key = get_key(label_dict, 2)[0]
print(type(key))

sub = data_frame
category_list = []
for label_num in sub["label_num"]:
    key = get_key(label_dict, int(label_num))[0]
    category_list.append(key)
sub["categories"] = category_list
sub = sub.drop(['text','label_num'], axis=1)
sub.to_csv("./processed_data/submission_processed_lstm.csv", index=0,encoding='utf-8')

