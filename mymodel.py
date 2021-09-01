import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel


class BERT(nn.Module):
    
    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=39)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, texts, labels, attn_mask):
        loss, logits = self.encoder(input_ids=texts, labels=labels, attention_mask=attn_mask)[:39]
        # logits = self.sigmoid(logits)
        return loss, logits

class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        ### New layers:
        self.lstm = nn.LSTM(768, 256, batch_first=True,bidirectional=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256*512, 39)
        

    def forward(self, texts, attn_mask):
        outputs = self.bert(input_ids=texts, attention_mask=attn_mask)
        sequence_output = outputs.last_hidden_state
        #print(sequence_output.shape)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
        #print(lstm_output.shape) # batch*256*512
        #hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
        # print(hidden.shape)
        flattened = self.flatten(lstm_output)
        #print(flattened.shape)
        linear_output = self.linear(flattened) ### assuming that you are only using the output of the last LSTM cell to perform classification
        
        return linear_output
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # model = BERT().to(device)
# model = CustomBERTModel().to(device)
# inputs = torch.randint(120,200,(3,256)).to(device)
# # labels = torch.tensor([0,1,2], dtype=torch.long).to(device)
# attention_mask = torch.ones((3,256), dtype=torch.long).to(device)
# # print(attention_mask)
# outputs = model(texts=inputs, attn_mask=attention_mask)

# print(outputs.shape)
