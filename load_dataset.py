import pandas 
import torch
from torch.utils.data import Dataset, DataLoader

class PaperDataset(Dataset):
    """Custom Dataset class."""
    def __init__(self, texts, targets, tokenizer, max_len):
        super().__init__()
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        """ Method to support indexing and return dataset[idx] """
        text = str(self.texts[index])
        target = self.targets[index]
        encoding = self.tokenizer.encode_plus(text, 
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            return_token_type_ids=False,
                                            pad_to_max_length=True,
                                            truncation=True)
        return {
        'text' : text,
        'input_ids' : encoding['input_ids'].flatten(),
        'attention_mask' : encoding['attention_mask'].flatten(),
        'targets' : torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = PaperDataset(
        texts=df['text'].to_numpy(),
        targets=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    # shuffle true is recommended so that batched between epochs donot look alike
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
        # pin_memory=True  # For faster data transfer from host to GPU in CUDA-enabled GPUs   
    )


class PaperTestDataset(Dataset):
    """Custom Test Dataset class."""
    def __init__(self, texts, tokenizer, max_len):
        super().__init__()
        self.texts = texts
        #self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        """ Method to support indexing and return dataset[idx] """
        text = str(self.texts[index])
        # target = self.targets[index]
        encoding = self.tokenizer.encode_plus(text, 
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            return_token_type_ids=False,
                                            pad_to_max_length=True,
                                            truncation=True)
        return {
        'text' : text,
        'input_ids' : encoding['input_ids'].flatten(),
        'attention_mask' : encoding['attention_mask'].flatten(),
        # 'targets' : torch.tensor(target, dtype=torch.long)
        }
    
def create_test_data_loader(df, tokenizer, max_len, batch_size):
    ds = PaperTestDataset(
        texts=df['text'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    # shuffle true is recommended so that batched between epochs donot look alike
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
        # pin_memory=True  # For faster data transfer from host to GPU in CUDA-enabled GPUs   
    )