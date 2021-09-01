
from os import truncate
from random import shuffle
import pandas as pd
from pandas.core.frame import DataFrame
import torch.nn as nn
import numpy as np
import math
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split
from translation_ct import *

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

# over_strat = {3: 11038, 0: 4260, 11: 3218, 9: 2798, 12: 2706, 5: 2509, 28: 1994, 13: 1940, 7: 1884, 15: 1741, 4: 1352, 16: 1292, 29: 1228, 27: 998, 23: 984, 21: 943, 18: 841, 6: 770, 26: 719, 1: 704, 14: 683, 20: 677, 19: 603, 2: 537, 8: 523, 25: 500, 10: 500, 30: 500, 17: 500, 31: 500, 24: 500, 22: 500, 32: 500, 36: 500, 35: 500, 33: 500, 37: 500, 34: 500, 38: 500}
# over = RandomOverSampler(sampling_strategy=over_strat)
# # under_strat = {3: 5000, 0: 4260, 11: 3218, 9: 2798, 12: 2706, 5: 2509, 28: 1994, 13: 1940, 7: 1884, 15: 1741, 4: 1352, 16: 1292, 29: 1228, 27: 998, 23: 984, 21: 943, 18: 841, 6: 770, 26: 719, 1: 704, 14: 683, 20: 677, 19: 603, 2: 537, 8: 523, 25: 500, 10: 500, 30: 500, 17: 500, 31: 500, 24: 500, 22: 500, 32: 500, 36: 500, 35: 500, 33: 500, 37: 500, 34: 500, 38: 500}
# # under = RandomUnderSampler(sampling_strategy=under_strat)
# # steps = [('o',over),('u',under)]
# steps = [('o', over)]
# pipeline = Pipeline(steps=steps)
# X = np.array(X).reshape(-1,1)
# X_res, Y_res = pipeline.fit_resample(X, Y)
# X_res = X_res.reshape(-1)
# print("Resampled data distribution: "+str(Counter(Y_res)))
# new_df = pd.DataFrame(X_res)
# new_df = new_df.rename(columns={0:'text'})
# new_df['label'] = Y_res
# print(new_df[:5])

# 扩充数据集

idx, keys, langs = generate_lang_dict()
label_num = Counter(data_frame['label']) #0-38
list_type = data_frame['label'].unique()
number = 500
# expected_num = {3: number, 0: number, 11: number, 9: number, 12: number, 5: number, 28: number, 13: number, 7: number, 15: number, 4: number, 16: number, 29: number, 27: number, 23: number, 21: number, 18: number, 6: number, 26: number, 1: number, 14: number, 20: number, 19: number, 2: number, 8: number, 25: number, 10: number, 30: number, 17: number, 31: number, 24: number, 22: number, 32: number, 36: number, 35: number, 33: number, 37: number, 34: number, 38: number}
expected_num = {3: 11038, 0: 4260, 11: 3218, 9: 2798, 12: 2706, 5: 2509, 28: 1994, 13: 1940, 7: 1884, 15: 1741, 4: 1352, 16: 1292, 29: 1228, 27: 998, 23: 984, 21: 943, 18: 841, 6: 770, 26: 719, 1: 704, 14: 683, 20: 677, 19: 603, 2: 537, 8: 523, 25: 500, 10: 500, 30: 500, 17: 500, 31: 500, 24: 500, 22: 500, 32: 500, 36: 500, 35: 500, 33: 500, 37: 500, 34: 500, 38: 500}
expand_res = []
interlingua_num = 2

# 2021.7.24版：调用bing api，每次传送一个样本
# for i in range(len(label_num)):
#     # 取子表
#     expand_res = []
#     sub_data_frame = data_frame[data_frame['label'].isin([list_type[i]])]
#     sub_len = len(sub_data_frame)
#     if sub_len < expected_num[i]:
#         # 先打乱再随机生成样本
#         # 规则：尽量每个样本都生成
#         delta = expected_num[i] - sub_len  # 差多少样本
#         if delta / sub_len <=1:
#             list_len = 1
#             # 任选delta个字符串生成
#             sub_data_frame = sub_data_frame.sample(frac = 1).reset_index(drop=True)
#             for j in range(delta):
#                 string = sub_data_frame['text'][j]
#                 # string, idx, 3, 'en', langs, 10
#                 res = trans_chain(string, idx, interlingua_num, 'en', langs, list_len)
#                 expand_res.extend(res)
#         else:
#             list_len = math.ceil(delta / sub_len)
#             # 需要用于生成的样本数
#             src_sample_num = math.ceil(delta / list_len)
#             sub_data_frame = sub_data_frame.sample(frac = 1).reset_index(drop=True)
#             j = delta - 1
#             for k in range(src_sample_num):
#                 string = sub_data_frame['text'][k]
#                 res = trans_chain(string, idx, interlingua_num, 'en', langs, list_len)
#                 expand_res.extend(res)
#                 j = j - list_len
#                 if j < 0:
#                     break
#             # expand_res = expand_res[:delta]
#         print(len(expand_res), sub_len + delta)
#         # 样本保存在expand_res中，加入数据集中
#         X = pd.concat([X, pd.DataFrame(expand_res)], axis=0)
#         expand_label = [i]*len(expand_res)
#         # expand_label = [i] * delta
#         Y = pd.concat([Y, pd.DataFrame(expand_label)], axis=0)
#         print(len(expand_res), len(expand_label))

# 2021.7.25 使用百度翻译api，每次传送最多三个样本
for i in range(len(label_num)):
    # 取子表
    expand_res = []
    CNT_NUM = 0
    sub_data_frame = data_frame[data_frame['label'].isin([list_type[i]])]
    sub_len = len(sub_data_frame)
    if sub_len < expected_num[i]:
        # 先打乱再随机生成样本
        # 规则：尽量每个样本都生成
        delta = expected_num[i] - sub_len  # 差多少样本
        if delta / sub_len <=1:
            list_len = 1
            # 任选delta个字符串生成
            sub_data_frame = sub_data_frame.sample(frac = 1).reset_index(drop=True)
            # for j in range(delta):
                # string = sub_data_frame['text'][j]
                # # string, idx, 3, 'en', langs, 10
                # res = trans_chain(string, idx, interlingua_num, 'en', langs, list_len)
                # expand_res.extend(res)
            j = 0
            while j < delta:
                loop_num = math.floor(delta / 3)
                final_num = delta - loop_num * 3
                for k in range(loop_num+1):
                    if k < loop_num:    # 传送3个
                        string = sub_data_frame['text'][3*k] + '\n' + sub_data_frame['text'][3*k+1] + '\n' + sub_data_frame['text'][3*k+2]
                        # res = trans_chain_baidu(string, idx, interlingua_num, 'en', langs, 3, list_len)
                        ######
                        CNT_NUM += 3*list_len
                        ######
                        # expand_res.extend(res)
                        j = j + 3
                    else:
                        string = sub_data_frame['text'][3*k]
                        for kk in range(final_num-1):
                            string = string + '\n' + sub_data_frame['text'][3*k+kk+1]
                        # res = trans_chain_baidu(string, idx, interlingua_num, 'en', langs, final_num, list_len)
                        ######
                        CNT_NUM += final_num*list_len
                        ######
                        # expand_res.extend(res)
                        j = j + final_num
        else:
            list_len = math.ceil(delta / sub_len)
            # 需要用于生成的样本数
            src_sample_num = math.ceil(delta / list_len)
            sub_data_frame = sub_data_frame.sample(frac = 1).reset_index(drop=True)
            j = delta - 1
            loop_num = math.floor(src_sample_num / 3)
            final_num = src_sample_num - loop_num * 3
            for k in range(loop_num + 1):
                if k < loop_num:
                    string = sub_data_frame['text'][3*k] + '\n' + sub_data_frame['text'][3*k+1] + '\n' + sub_data_frame['text'][3*k+2]
                    # res = trans_chain_baidu(string, idx, interlingua_num, 'en', langs, 3, list_len)
                    ######
                    CNT_NUM += 3*list_len
                    ######
                    # expand_res.extend(res)
                    j = j - list_len * 3
                else:
                    string = sub_data_frame['text'][3*k]
                    for kk in range(final_num-1):
                        string = string + '\n' + sub_data_frame['text'][3*k+kk+1]
                    # res = trans_chain_baidu(string, idx, interlingua_num, 'en', langs, final_num, list_len)
                    ######
                    CNT_NUM += final_num*list_len
                    ######
                    # expand_res.extend(res)
                    j = j - list_len * final_num
                if j < 0:
                    break
            # expand_res = expand_res[:delta]

            # for k in range(src_sample_num):
            #     string = sub_data_frame['text'][k]
            #     res = trans_chain(string, idx, interlingua_num, 'en', langs, list_len)
            #     expand_res.extend(res)
            #     j = j - list_len
            #     if j < 0:
            #         break
            # expand_res = expand_res[:delta]
        print(len(expand_res), sub_len + delta)
        # 样本保存在expand_res中，加入数据集中
        X = pd.concat([X, pd.DataFrame(expand_res)], axis=0)
        expand_label = [i]*len(expand_res)
        # expand_label = [i] * delta
        Y = pd.concat([Y, pd.DataFrame(expand_label)], axis=0)
        print(len(expand_res), len(expand_label))


new_df = pd.DataFrame(X)
new_df = new_df.rename(columns={0:'text'})
new_df['label'] = Y

new_df.to_csv('./processed_data/df_ct.csv', mode='a', header=True, index=None)
# 对数据集进行切分
train_df, valid_df = train_test_split(new_df, train_size=0.9, shuffle=True)
train_df.to_csv('./processed_data/train_df_ct.csv', mode='a', header=True, index=None)
valid_df.to_csv('./processed_data/valid_df_ct.csv', mode='a', header=True, index=None)
print(train_df.shape)
print(valid_df.shape)




