import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from train_utils import train, train_with_focal_loss, train_use_lstm
from mymodel import BERT, CustomBERTModel
from load_dataset import PaperDataset, create_data_loader
import warnings
warnings.filterwarnings('ignore')

# Model parameter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取csv文件
df_train = pd.read_csv("./processed_data/train_df_0.9.csv")
df_val = pd.read_csv("./processed_data/valid_df_0.1.csv")

# 获取dataloader
train_data_loader = create_data_loader(df_train, tokenizer, MAX_SEQ_LEN, batch_size=32)
valid_data_loader = create_data_loader(df_val, tokenizer, MAX_SEQ_LEN, batch_size=32)

# 训练模型
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
#                                                           num_labels=39,
#                                                           output_hidden_states=False,
#                                                           output_attentions=False)
# model = BERT().to(device)
model = CustomBERTModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
# train(model, optimizer, train_data_loader, valid_data_loader, num_epochs=10, file_path='./pretrain_clean_text', best_valid_acc = 0.0)
#train_with_focal_loss(model, optimizer, train_data_loader, valid_data_loader, num_epochs=10, file_path='./pretrain_focal_loss', best_valid_acc = 0.0)
train_use_lstm(model, optimizer, train_data_loader, valid_data_loader, num_epochs=10, file_path='./pretrain_lstm', best_valid_acc = 0.0)
