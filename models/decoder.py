import torch
import torch.nn as nn
# from torchvision import models
# import config


class DecoderRNN(nn.Module):
    '''
    The DecoderRNN class takes in the embedding size, hidden size, vocabulary size, and number of layers
    as input. It then creates an LSTM layer, a dropout layer, an embedding layer, and a linear layer.
    The forward function takes in the features and captions as input and returns the outputs.
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions):
        """
        The function takes in the features and captions and returns the output of the linear layer.
        
        :param features: (batch_size, embed_size) Features extracted from an image.
        :param captions: (batch_size, max_length) Text Captions associated with the image.
        :return: The output of the linear layer.
        """
        # Create embedded word vectors for each word in the captions shape: (batch_size, captions_length-1, embed_size
        embeddings = self.dropout(self.embedding(captions))

        #add new dimension second (batch_size, embed_size) -> (batch_size, 1, embed_size)
        features = features.unsqueeze(1)    #shape: (batch_size, 1, embed_size)

        #stack the features and captions
        # print(features.shape)
        embeddings = embeddings[:, 0, :, :].squeeze(dim=1)
        # print(embeddings.shape)

        embeddings = torch.cat((features, embeddings), dim=1) #shape: (batch_size, caption length+1, embed_size)

        # print(f"embeddings {embeddings.shape}")    -> [8, 41, 256])
        hidden_out, _ = self.lstm(embeddings)   #shape:(batch_size, caption length, hidden_size)

        
        #fully connect layer 
        outputs = self.linear(hidden_out)   #shape:(batch_size, caption_length, vocab_size)
        outputs = outputs[:, :-1, :]
        # print(outputs.shape)
        return outputs
        # ----------------------------------------------------
        # embeddings = self.dropout(self.embedding(captions))
        # embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        # hiddens, _ = self.lstm(embeddings)
        # outputs = self.linear(hiddens)
        # return outputs
