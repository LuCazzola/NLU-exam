import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel


class ModelIAS(nn.Module):

    def __init__(self, out_slot, out_int, vocab_len, dropout_enable=False, emb_dropout=0.1, out_dropout=0.1, bert_version="bert-base-uncased", num_heads=4):
        super(ModelIAS, self).__init__()
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)
        self.bertModel = BertModel.from_pretrained(bert_version)
        #for p in self.bertModel.parameters():
        #  p.requires_grad = False
          
        hid_size = self.bertModel.config.hidden_size

        # computes self-attention for subtokens and returns unified representation
        self.merger = SubtokenMerger(hid_size, num_heads, self.tokenizer)

        # classifier for slots and intents
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
        # Dropout layers
        self.dropout_enable = dropout_enable
        if dropout_enable :
            self.dropoutEmb = nn.Dropout(emb_dropout)
            self.dropoutOut = nn.Dropout(out_dropout)
        
        
    def forward(self, bert_inputs):
        
        bert_out = self.bertModel(**bert_inputs)
        
        # Process the batch
        pooler_output = bert_out.pooler_output
        last_hidden_states = bert_out.last_hidden_state

        # Dropout on embeddings
        if self.dropout_enable :
            pooler_output = self.dropoutEmb(pooler_output)
            last_hidden_states = self.dropoutEmb(last_hidden_states)

        # Forward the grouped embeddings through self.merger
        #
        #token_embeddings = self.merger(bert_inputs, last_hidden_states)
        token_embeddings = last_hidden_states

        # Dropout on rnn outputs
        if self.dropout_enable :
            token_embeddings = self.dropoutOut(token_embeddings)
        
        # Compute slot logits
        slots = self.slot_out(token_embeddings)
        # Compute intent logits
        intent = self.intent_out(pooler_output)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent



class SubtokenMerger(nn.Module):
    def __init__(self, hidden_size, num_heads, tokenizer):
        super(SubtokenMerger, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.tokenizer = tokenizer
        self.num_heads = num_heads

    def forward(self, bert_inputs, token_embeddings):
        batch_size = token_embeddings.shape[0]
        hidden_size = token_embeddings.shape[-1]

        # compute mapping between words and corresponding subtokens
        word_tok_map = self.get_subtok_map(bert_inputs, batch_size)
        # use the mapping to get the relative tensors
        subtokens = self.get_subtokens(token_embeddings, word_tok_map, batch_size, hidden_size) # shape : (batch_size, subtokenized_words_count, subtok_count, hidden_size)
        subtokenized_words_count = subtokens.shape[1]
        subtok_count = subtokens.shape[2]

        # Reshape to (batch_size * subtokenized_words_count, subtok_count, hidden_size) to prepare for multihead self-attention 
        attn_input = subtokens.view(-1, subtok_count, hidden_size)
        # compute attention mask for paddings
        attn_mask = (subtokens.sum(dim=-1) != 0).view(-1, subtok_count).float().unsqueeze(-1).to(token_embeddings.device)
        attn_mask = attn_mask.expand(-1, -1, subtok_count) * attn_mask.transpose(1, 2).expand(-1, subtok_count, -1)
        attn_mask_preExpand = attn_mask.clone()
        # expand to all attention heads
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, subtok_count, subtok_count)  

        # Apply multihead attention
        # 
        attn_output , attn_weights = self.self_attention(attn_input, attn_input, attn_input, attn_mask=attn_mask, need_weights=True)

        # mask attention weights and sum over subtok dimension
        subtok_contributions = (attn_mask_preExpand * attn_weights).sum(dim=1) 
        # normmalize
        subtok_contributions = subtok_contributions / (subtok_contributions.sum(dim=-1, keepdim=True) + 1e-8)
        subtok_contributions = subtok_contributions.view(batch_size, subtokenized_words_count, subtok_count)

        # Substitute subtokens with unified representation
        for batch_idx, mappings in enumerate(word_tok_map):
            for tok_idx, mapping in enumerate(mappings):
                unified_word = (token_embeddings[batch_idx, mapping] * subtok_contributions[batch_idx, tok_idx, :len(mapping)].unsqueeze(-1)).sum(dim=0)
                token_embeddings[batch_idx, mapping[0]] = unified_word  # Substitute the first subtoken with the unified representation
                token_embeddings[batch_idx, mapping[1:]] *= 0.0  # Zero all the other subtokens
        
        return token_embeddings
    

    def get_subtok_map (self, bert_inputs, batch_size):
        # Create a dictionary to map subtokens to their original word
        idx_dict = [[] for _ in range(batch_size)]

        # Group tensors belonging to the same original word
        for batch, ids in enumerate(bert_inputs["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(ids)

            key = 0
            for idx, token in enumerate(tokens):
                if token == "[PAD]":
                    continue

                if not token.startswith("##"):
                    idx_dict[batch].append([])
                    idx_dict[batch][key].append(idx)
                    key += 1
                else:
                    idx_dict[batch][key-1].append(idx)
        
        # Drop lists of length 1
        for batch in range(batch_size):
            idx_dict[batch] = [tok_pointer for tok_pointer in idx_dict[batch] if len(tok_pointer) > 1]

        return idx_dict
    

    def get_subtokens (self, subtoken_embeddings, word_tok_map, batch_size, hidden_size):
        
        subtok_count = max([len(mapping) for batch in word_tok_map for mapping in batch])

        ensambler_input = []
        for batch_idx in range(batch_size):
            # each element of the list is a tensor of shape (n_subtokens, hidden_size), where
            # n_subtokens is the number of subtokens in which in the word has been decomposed
            sequence_word_subtok = [subtoken_embeddings[batch_idx, mapping] for mapping in word_tok_map[batch_idx]]

            if len(sequence_word_subtok) > 0:
                sequence_word_subtok_padded = torch.stack([torch.cat([word, torch.zeros((subtok_count - word.shape[0], hidden_size), device=subtoken_embeddings.device)], dim=0) for word in sequence_word_subtok], dim=0)
                # pad the sequence of subtokens to have a tensor of shape (n_words, max_subtokens, hidden_size)
                ensambler_input.append(sequence_word_subtok_padded)
            else :
                # if the sequence has no sub-tokens, append a zero mask
                ensambler_input.append(torch.zeros(1, subtok_count, hidden_size))
        
        ensambler_input = pad_sequence(ensambler_input, batch_first=True, padding_value=0).to(subtoken_embeddings.device)

        return ensambler_input
