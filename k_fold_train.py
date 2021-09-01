import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
from train_utils import train, train_with_focal_loss, train_with_kfold
from mymodel import BERT
from load_dataset import PaperDataset, create_data_loader
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Model parameter
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 读取整个训练数据集csv文件
df_train = pd.read_csv("./processed_data/train_df_all.csv")
# X = df_train['text']
# Y = df_train['label']
kf = KFold(n_splits=10, shuffle=True, random_state=14)
for k, (train_index, valid_index) in enumerate(kf.split(df_train)):
    print("K_Fold_index:"+str(k))
    train_kf = df_train.iloc[train_index]
    valid_kf = df_train.iloc[valid_index]
    train_data_loader = create_data_loader(train_kf, tokenizer, MAX_SEQ_LEN, batch_size=32)
    valid_data_loader = create_data_loader(valid_kf, tokenizer, MAX_SEQ_LEN, batch_size=32)
    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    train_with_kfold(k, model, optimizer, train_data_loader, valid_data_loader, num_epochs=5, file_path='./pretrain_kfold', best_valid_acc = 0.0)



