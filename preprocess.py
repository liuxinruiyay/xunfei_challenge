import pandas as pd
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split


def train_file_preprocess(file_path):
    file_handle = open(file_path,'r')
    lines = file_handle.readlines()

    tmpstr = ""
    tmpstr_list = []
    for line in lines:
        if line.find("train_") != -1:
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
    label_list = []
    tmpstr_list = tmpstr_list[1:]
    # print(tmpstr_list)
    for str in tmpstr_list:
        tmp = str.split('\t')
        paper_id = tmp[0].strip(' ')
        title = tmp[1]
        abstract = tmp[2]
        label = tmp[3].strip(' ')
        paperid_list.append(paper_id)
        title_list.append(title)
        abstract_list.append(abstract)
        label_list.append(label)
    return paperid_list, title_list, abstract_list, label_list

file_path = "./paper_data/train.csv"
paperid_list = train_file_preprocess(file_path)[0]
title_list = train_file_preprocess(file_path)[1]
abstract_list = train_file_preprocess(file_path)[2]
label_list = train_file_preprocess(file_path)[3]


# 获取标签映射表
label_count = 0
label_dict = {}
label_num_list = []
for label in label_list:
    if label not in label_dict:
        label_dict[label] = label_count
        label_count += 1
for label in label_list:
    tmp = label_dict[label]
    label_num_list.append(tmp)
print(len(label_dict))
print(label_dict)


# 将title和abstract进行拼接，中间用句号隔开
text_list = []
for index in range(0,len(title_list)):
    tmp = title_list[index].replace('"','') + '. '+ abstract_list[index].replace('"','')
    text_list.append(tmp)

data_frame = pd.DataFrame(text_list)
data_frame = data_frame.rename(columns={0:'text'})
data_frame['label'] = label_num_list
print(data_frame.head())

# # 对数据集进行切分
# train_df, valid_df = train_test_split(data_frame, train_size=0.9)
# train_df.to_csv('./processed_data/train_df.csv', mode='a', header=True, index=None)
# valid_df.to_csv('./processed_data/valid_df.csv', mode='a', header=True, index=None)
# print(train_df.shape)
# print(valid_df.shape)

#所有数据集作为训练数据
data_frame.to_csv('./processed_data/train_df.csv', mode='a', header=True, index=None)
print(data_frame.shape)