import numpy as np
from torch.utils.data import Dataset, DataLoader, dataset

from data_processing import build_vocab, build_label_vocab, load_patterns

class ChatDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.load_vocab()
        self.load_data()

    def load_vocab(self):
        self.token_to_idx, _ = build_vocab(self.data_file)
        self.label_to_idx, _ = build_label_vocab(self.data_file)
        self.vocab_size = len(self.token_to_idx)
        self.class_size = len(self.label_to_idx)

    def load_data(self):
        self.X_train, self.y_train = load_patterns(self.data_file, self.token_to_idx, self.label_to_idx)
        #print(self.X_train)

    def __getitem__(self, index):
        x = self.X_train[index][:min(10,len(self.X_train[index]))]
        x = x + [0]*(10-len(x))
        x = np.array(x)
        
        return x, self.y_train[index]

    def __len__(self):
        return len(self.X_train)


#dataset = ChatDataset("chat_data.json")
#for i in range(10):
#    print(dataset[i])
        


