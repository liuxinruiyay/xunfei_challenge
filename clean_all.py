
##去除文本中停用词，对文本进行清洗

import re
import torch.nn
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# stop_words = nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemer = PorterStemmer()
# print(stop_words)
special_chars = re.compile('[^-9a-z#+_]')
add_space = re.compile('[/(){}\[\]\\@;]')

def remove_SW_Stem(text):
    text = [stemer.stem(words) for words in text.split(" ") if words not in stop_words]
    return " ".join(text)

def clean_text(text):
    # text = text.lower()
    # text = add_space.sub(" ",text)
    # text = special_chars.sub(" ",text)
    text = remove_SW_Stem(text)
    return text

data_frame = pd.read_csv("./processed_data/final_dataset1.0.csv")
new_text_list = []
for text in data_frame['text']:
    text = clean_text(text)
    new_text_list.append(text)

data_frame['text'] = new_text_list
data_frame.to_csv('./processed_data/final_train.csv', mode='a', header=True, index=None)
