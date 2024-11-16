import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, emb_size, hid_size, out_slot, out_int, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout_enable=False, emb_dropout=0.1, hid_dropout=0.1, out_dropout=0.1):
        """
        A neural network model for intent and slot classification using LSTM.

        Args:
            emb_size (int): Size of word embeddings.
            hid_size (int): Size of hidden layers.
            out_slot (int): Number of output slots for slot filling.
            out_int (int): Number of output intents.
            vocab_len (int): Size of the vocabulary.
            n_layer (int, optional): Number of LSTM layers. Defaults to 1.
            pad_index (int, optional): Padding index for the embedding layer. Defaults to 0.
            bidirectional (bool, optional): If True, uses a bidirectional LSTM. Defaults to False.
            dropout_enable (bool, optional): If True, applies dropout. Defaults to False.
            emb_dropout (float, optional): Dropout rate for embeddings. Defaults to 0.1.
            hid_dropout (float, optional): Dropout rate for hidden layers. Defaults to 0.1.
            out_dropout (float, optional): Dropout rate for output layers. Defaults to 0.1.
        """
        
        super(ModelIAS, self).__init__()
        # Dropout layers are zeroed if flag '--dropout_enabled' is unset (to suppress warnings)
        if not dropout_enable or n_layer == 1 :
            hid_dropout=0.0

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, dropout=hid_dropout, bidirectional=bidirectional, batch_first=True)    

        self.slot_out = nn.Linear(2*hid_size, out_slot) if bidirectional else nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        # Dropout layer How/Where do we apply it?
        self.dropout_enable = dropout_enable
        if dropout_enable :
            self.dropoutEmb = nn.Dropout(emb_dropout)
            self.dropoutOut = nn.Dropout(out_dropout)
        
        
    def forward(self, utterance, seq_lengths):
        """
        Forward pass of the model.

        Args:
            utterance (Tensor): Tensor containing the input utterances with shape (batch_size, seq_len).
            seq_lengths (Tensor): Tensor containing the lengths of the sequences in the batch.

        Returns:
            tuple: A tuple containing:
                - Tensor: Slot logits with shape (batch_size, classes, seq_len).
                - Tensor: Intent logits with shape (batch_size, out_int).
        """
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # Dropout on embeddings
        if self.dropout_enable :
            utt_emb = self.dropoutEmb(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Dropout on slots and intents hiddens
        if self.dropout_enable :
            utt_encoded = self.dropoutOut(utt_encoded)
            last_hidden = self.dropoutOut(last_hidden)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
