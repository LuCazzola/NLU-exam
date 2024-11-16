"""
model's architecture in pytorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

from utils import DEVICE

class LM_RNN(nn.Module):
    """
    A RNN based model for language modeling.

    Args:
        emb_size (int): Size of the word embeddings.
        hidden_size (int): Size of the hidden layer.
        output_size (int): Size of the output layer (vocabulary size).
        pad_index (int, optional): Padding index for embeddings. Default is 0.
        emb_dropout (float, optional): Dropout probability for the embedding layer. Default is 0.1.
        hid_dropout (float, optional): Dropout probability for the hidden layers. Default is 0.1.
        out_dropout (float, optional): Dropout probability for the output layer. Default is 0.1.
        n_layers (int, optional): Number of recurrent layers. Default is 1.
        recLayer_type (str, optional): Type of recurrent layer ('LSTM' or 'GRU'). Default is 'LSTM'.
        var_dropout (bool, optional): Flag to enable variational dropout. Default is False.
        weight_tying (bool, optional): Flag to enable weight tying between embedding and output layers. Default is False.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.1, hid_dropout=0.1, out_dropout=0.1, n_layers=1, recLayer_type='LSTM', var_dropout=False, weight_tying=False):
        super(LM_RNN, self).__init__()
        
        # token embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Recurrent layer
        #   It is defined stack of LSTMs so we can modify dropout 
        #   behavior between components
        self.n_layers = n_layers
        if recLayer_type == 'LSTM' :
            stacked_rnns = [nn.LSTM(emb_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True) for _ in range(n_layers)]
        elif recLayer_type == 'GRU' :
            stacked_rnns = [nn.GRU(emb_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True) for _ in range(n_layers)]
        else :
            raise ValueError("Unsupported Recurrent Cell type.\n   - available choices : LSTM")
        # stack rnns as single module to handle them as a single layer
        self.rnns = nn.ModuleList(stacked_rnns)


        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        # apply weight tying between embedding and output layer if 'weight_tying' is enabled
        if weight_tying: 
            if emb_size != hidden_size:
                raise ValueError("flag 'weight_tying' is enabled, embedding_size must match hidden_size")
            # set bias to zero
            with torch.no_grad():
                self.output.bias.zero_()
            # Tie output weights to embedding
            self.output.weight = self.embedding.weight
        
        
        # Choose among variational dropout and standard with 'var_dropout' flag
        self.var_dropout = var_dropout
        if var_dropout :
            self.dropoutEmb = varDropout(p=emb_dropout)
            self.dropoutHid = varDropout(p=hid_dropout)
            self.dropoutOut = varDropout(p=out_dropout)
        else :   
            self.dropoutEmb = nn.Dropout(p=emb_dropout, inplace=False)
            self.dropoutHid = nn.Dropout(p=hid_dropout, inplace=False)
            self.dropoutOut = nn.Dropout(p=out_dropout, inplace=False)
    

    def forward(self, input_sequence):
        """
        Forward pass through the model.

        Args:
            input_sequence (Tensor): Input sequence tensor.

        Returns:
            Tensor: Output tensor after passing through the model.
        """
        # generate word embedding for input sequence
        emb = self.embedding(input_sequence)
        # Apply dropout to embedding layer
        emb = self.dropoutEmb(emb)

        # Recurrent layers
        rnn_out = emb
        for layer, rnn in enumerate(self.rnns) :
            rnn.flatten_parameters()
            rnn_out, _  = rnn(rnn_out)
            # Apply dropout to rec. hidden layer if it's not the last one
            rnn_out = self.dropoutHid(rnn_out) if layer != self.n_layers - 1 else rnn_out

        # Apply dropout to rnn output layer
        rnn_out = self.dropoutOut(rnn_out)
        # Last linear layer
        output = self.output(rnn_out).permute(0,2,1)

        return output


class varDropout(nn.Module):
    """
    Variational dropout class module.

    Args:
        p (float, optional): Dropout probability. Default is 0.1.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p=p

    # assumes (batch, seq_len, hidden) shape for batch
    def forward(self, batch, p=0.1):
        """
        Apply variational dropout to the input batch.

        Args:
            batch (Tensor): Input tensor of shape (batch, seq_len, hidden).
            p (float, optional): Dropout probability. Default is 0.1.

        Returns:
            Tensor: Tensor with dropout applied.
        """
        if not self.training :
            # identity bypass if the model is in evaluation mode
            return batch
        
        # each element has an idependent probability 'p' of being zeroed
        seq_elem_mask = torch.ones((batch.size(0), 1, batch.size(2)), device=batch.device)
        partial_mask = torch.bernoulli(seq_elem_mask - self.p)
        # expand over the seq_len dimension (this makes sequence elements share the mask)
        mask = partial_mask.expand_as(batch)
        
        # apply mask and normalize
        return (batch * mask)/(1 - self.p)
