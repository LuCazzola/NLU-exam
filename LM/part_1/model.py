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
    A RNN based model for language modeling.

    Args:
        emb_size (int): Size of the word embeddings.
        hidden_size (int): Number of features in the hidden state of the RNN.
        output_size (int): Size of the output vocabulary.
        pad_index (int, optional): Padding index for the embedding layer. Default is 0.
        emb_dropout (float, optional): Dropout probability for the embedding layer. Default is 0.1.
        hid_dropout (float, optional): Dropout probability for the recurrent layer. Default is 0.1.
        out_dropout (float, optional): Dropout probability for the output layer. Default is 0.1.
        n_layers (int, optional): Number of recurrent layers. Default is 1.
        recLayer_type (str, optional): Type of recurrent cell used. Options are 'vanilla', 'LSTM', or 'GRU'. Default is 'vanilla'.
        dropout_enabled (bool, optional): Flag to enable or disable dropout layers. Default is False.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.1, hid_dropout=0.1, out_dropout=0.1, n_layers=1, recLayer_type='vanilla', dropout_enabled=False):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Dropout layers are zeroed if flag '--dropout_enabled' is unset (to suppress warnings)
        if not dropout_enabled or n_layers == 1 :
            hid_dropout = 0.0
            
        if recLayer_type == 'vanilla' :
            # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, dropout=hid_dropout, bidirectional=False, batch_first=True)
        elif recLayer_type == 'LSTM' :
             # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, dropout=hid_dropout, bidirectional=False, batch_first=True)
        elif recLayer_type == 'GRU' :
            # Pytorch's GRU layer: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            self.rnn = nn.GRU(emb_size, hidden_size, n_layers, dropout=hid_dropout, bidirectional=False, batch_first=True)
        else :
            print("Unsupported Recurrent Cell type.\n   - available choices : {vanilla, LSTM, GRU}")
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
        """
        Forward pass through the model.

        Args:
            input_sequence (torch.Tensor): Input tensor containing token IDs.

        Returns:
            torch.Tensor: Output tensor with predictions for each token in the sequence.
        """
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