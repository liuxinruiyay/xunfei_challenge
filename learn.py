import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
#from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.metrics import accuracy_score
# y_pred = [0, 2, 1, 3]
# y_true = [0, 1, 2, 3]
# acc = accuracy_score(y_true, y_pred)
# print(acc)
# encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased")
# data_frame = pd.read_csv('./processed_data/train_df.csv')
# str = data_frame['text'][5]
# input_id = tokenizer(str, padding=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
# print(PAD_INDEX)
# print(UNK_INDEX)
# print(input_id['input_ids'].shape)
import torch
a = torch.randn(3,16)
b = torch.randn(3,16)
c = torch.cat((a, b),axis=0)

print(a)
print(b)
print(c)

