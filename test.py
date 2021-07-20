import torch
import pandas as pd
import torch.nn as nn
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
        output = model(titletext, attention_mask = attention_mask)
        logits = output.logits
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
load_path="./pretrain/model-0.7984.pt"
state_dict = torch.load("./pretrain/model-0.7984.pt", map_location=device)
print(f'Model loaded from <== {load_path}')
best_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=39,
                                                          output_hidden_states=False,
                                                          output_attentions=False).to(device)
# # best_model = BertForSequenceClassification().to(device)
best_model.load_state_dict(state_dict['model_state_dict'])
y_pred = evaluate(best_model, test_data_loader)

data_frame['label_num'] = y_pred
print(data_frame['text'][:5])
print(y_pred[:5])
print(data_frame['label_num'][:5])
data_frame.to_csv('./processed_data/submission.csv', mode='a', header=True, index=None)
print(data_frame.shape)


