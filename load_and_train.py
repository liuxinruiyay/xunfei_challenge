import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from train_utils import train
from mymodel import BERT
import torch.optim as optim

# Model parameter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fields
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('label', label_field)]
print("sssss")
# TabularDataset
train_data, valid_data = TabularDataset.splits(path='./processed_data', train='train_df.csv', validation='valid_df.csv',
                                           format='CSV', fields=fields, skip_header=True)


# Iterators
train_iter = BucketIterator(train_data, batch_size=64, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid_data, batch_size=64, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)

# model = BERT().to(device)
# optimizer = optim.Adam(model.parameters(), lr=2e-5)
# train(model, optimizer, train_iter, valid_iter, num_epochs=5, file_path='./pretrain', best_valid_acc = 0.0)