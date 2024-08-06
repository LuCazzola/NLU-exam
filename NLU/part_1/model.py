import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, emb_size, hid_size, out_slot, out_int, vocab_len, n_layer=1, pad_index=0, bidirectional=False, dropout_enable=False, emb_dropout=0.1, hid_dropout=0.1, out_dropout=0.1):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        if not dropout_enable and n_layer == 1 :
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
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # Dropout on embeddings
        if self.dropout_enable :
            utt_emb = self.dropoutEmb(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Dropout on rnn outputs
        if self.dropout_enable :
            packed_output = self.dropoutOut(packed_output)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
