import torch.nn as nn
from transformers import BertForSequenceClassification


class BERT(nn.Module):
    
    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=39)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:39]

        return loss, text_fea