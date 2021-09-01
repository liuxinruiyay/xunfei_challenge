
import pandas as pd
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
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

print("Original data distribution: "+str(Counter(data_frame['label'])))
X = data_frame['text']
Y = data_frame['label']
over_strat = {3: 11038, 0: 4260, 11: 3218, 9: 2798, 12: 2706, 5: 2509, 28: 1994, 13: 1940, 7: 1884, 15: 1741, 4: 1352, 16: 1292, 29: 1228, 27: 998, 23: 984, 21: 943, 18: 841, 6: 770, 26: 719, 1: 704, 14: 683, 20: 677, 19: 603, 2: 537, 8: 523, 25: 500, 10: 500, 30: 500, 17: 500, 31: 500, 24: 500, 22: 500, 32: 500, 36: 500, 35: 500, 33: 500, 37: 500, 34: 500, 38: 500}
over = RandomOverSampler(sampling_strategy=over_strat)
# under_strat = {3: 5000, 0: 4260, 11: 3218, 9: 2798, 12: 2706, 5: 2509, 28: 1994, 13: 1940, 7: 1884, 15: 1741, 4: 1352, 16: 1292, 29: 1228, 27: 998, 23: 984, 21: 943, 18: 841, 6: 770, 26: 719, 1: 704, 14: 683, 20: 677, 19: 603, 2: 537, 8: 523, 25: 500, 10: 500, 30: 500, 17: 500, 31: 500, 24: 500, 22: 500, 32: 500, 36: 500, 35: 500, 33: 500, 37: 500, 34: 500, 38: 500}
# under = RandomUnderSampler(sampling_strategy=under_strat)
# steps = [('o',over),('u',under)]
steps = [('o', over)]
pipeline = Pipeline(steps=steps)
X = np.array(X).reshape(-1,1)
X_res, Y_res = pipeline.fit_resample(X, Y)
X_res = X_res.reshape(-1)
print("Resampled data distribution: "+str(Counter(Y_res)))
new_df = pd.DataFrame(X_res)
new_df = new_df.rename(columns={0:'text'})
new_df['label'] = Y_res
print(new_df[:5])
new_df.to_csv('./processed_data/train_df_over_sampled.csv', mode='a', header=True, index=None)

# # 对数据集进行切分
# train_df, valid_df = train_test_split(new_df, train_size=0.9, shuffle=True)
# train_df.to_csv('./processed_data/train_df.csv', mode='a', header=True, index=None)
# valid_df.to_csv('./processed_data/valid_df.csv', mode='a', header=True, index=None)
# print(train_df.shape)
# print(valid_df.shape)




