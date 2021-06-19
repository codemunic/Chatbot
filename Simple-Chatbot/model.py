import torch.nn as nn
import torch 


class BiLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout=0.2):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dp = dropout
        self.num_layers = 1

        self.embed = self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers,
                            bidirectional = True, batch_first = True)

        self.linear = nn.Linear(self.hidden_size*2,self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=self.dp)


    def forward(self,text):
        #text = [bs,sent_len]
        #print(text.size())

        text_embed = self.embed(text)
        #text_embed = [bs,sent_len,embed_size]
        #print("text embed: ",text_embed.size())
        _,(text_lstm,_) = self.lstm(text_embed)
        #print(text_lstm.size())
        #text_lstm = [2, bs, hidden_size]
        text_concat = torch.cat((text_lstm[0], text_lstm[1]), 1)
        text_lin = self.linear(text_concat)
        text_lin = self.dropout(text_lin)
        text_out = self.out(text_lin)

        return text_out

