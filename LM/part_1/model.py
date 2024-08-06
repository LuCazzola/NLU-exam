"""
model's architecture in pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_RNN(nn.Module):
    """
    Language Model Recurrent Neural Network (LM_RNN) class that defines the architecture of the model.
    
    Args:
        emb_size (int): Size of the embedding vectors.
        hidden_size (int): Number of features in the hidden state.
        output_size (int): Size of the output vocabulary.
        pad_index (int, optional): Index of the padding token. Default is 0.
        out_dropout (float, optional): Dropout probability for the output layer. Default is 0.1.
        hid_dropout (float, optional): Dropout probability for the rnn hidden layer. Default is 0.0.
        emb_dropout (float, optional): Dropout probability for the embedding layer. Default is 0.1.
        n_layers (int, optional): Number of recurrent layers. Default is 1.
        recLayer_type (str, optional): Type of recurrent layer ('vanilla' or 'LSTM'). Default is 'vanilla'.
        dropout_enabled (bool, optional): Flag to enable dropout layers. Default is False.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.1, hid_dropout=0.0, out_dropout=0.1, n_layers=1, recLayer_type='vanilla', dropout_enabled=False):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if recLayer_type == 'vanilla' :
            # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, dropout=hid_dropout, bidirectional=False, batch_first=True)
        elif recLayer_type == 'LSTM' :
             # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        else :
            print("Unsupported Recurrent Cell type.\n   - available choices : {vanilla, LSTM}")
            exit()

        # Dropout layers are added only if it's set the flag '--dropout_enabled'
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.dropoutEmb = None
        self.dropoutOut = None
        if dropout_enabled :
            self.dropoutEmb = nn.Dropout(p=emb_dropout, inplace=False)
            self.dropoutOut = nn.Dropout(p=out_dropout, inplace=False)

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        # generate word embedding for input sequence
        emb = self.embedding(input_sequence)

        # First dropout layer
        if  self.dropoutEmb is not None :
            emb = self.dropoutEmb(emb)

        # Recurrent layer
        rnn_out, _  = self.rnn(emb)

        # Second dropout layer
        if  self.dropoutOut is not None :
            rnn_out = self.dropoutOut(rnn_out)

        # Last linear layer
        output = self.output(rnn_out).permute(0,2,1)
        return output 