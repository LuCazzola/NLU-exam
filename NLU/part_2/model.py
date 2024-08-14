import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

class ModelIAS(nn.Module):

    def __init__(self, out_slot, out_int, vocab_len, finetune_bert=True, bert_version="bert-base-uncased", dropout_enable=False, int_dropout=0.1, slot_dropout=0.1, merger_enable=False,  num_heads=4) :
        """
        Initialize the ModelIAS class.

        Args:
            out_slot (int): Number of slots (output size for slot filling).
            out_int (int): Number of intents (output size for intent classification).
            vocab_len (int): Vocabulary length.
            finetune_bert (bool, optional): Whether to finetune BERT. Defaults to True.
            bert_version (str, optional): BERT model version. Defaults to "bert-base-uncased".
            dropout_enable (bool, optional): Whether to use dropout. Defaults to False.
            int_dropout (float, optional): Dropout rate for intents. Defaults to 0.1.
            slot_dropout (float, optional): Dropout rate for slots. Defaults to 0.1.
            merger_enable (bool, optional): Whether to use subtoken merger. Defaults to False.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super(ModelIAS, self).__init__()
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)
        self.bertModel = BertModel.from_pretrained(bert_version)
        if not finetune_bert :
            for p in self.bertModel.parameters():
                p.requires_grad = False
          
        hid_size = self.bertModel.config.hidden_size

        # computes self-attention for subtokens and returns unified representation
        # in the first subtoken of each word (which has been decomposed in subtokens)
        self.merger_enable = merger_enable
        if merger_enable :
            self.merger = SubtokenMerger(hid_size, num_heads, self.tokenizer)

        # classifier for slots and intents
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
        # Dropout layers
        self.dropout_enable = dropout_enable
        if dropout_enable :
            self.dropoutIntents = nn.Dropout(int_dropout)
            self.dropoutSlots = nn.Dropout(slot_dropout)

         
    def forward(self, bert_inputs, subtoken_positions=None):
        """
        Forward pass of the model.

        Args:
            bert_inputs (dict): Input for BERT.
            subtoken_positions (list, optional): Subtoken positions. Defaults to None.

        Returns:
            tuple: Logits for slots, logits for intents, and subtoken weights.
        """
        # embed inputs with BERT
        bert_out = self.bertModel(**bert_inputs)
        # extract values
        pooler_output = bert_out.pooler_output          # for Intents
        last_hidden_states = bert_out.last_hidden_state # for Slots

        if self.merger_enable :
            # computes self-attention for subtokens and returns unified representation
            # in the first subtoken of each word (which has been decomposed in subtokens)
            last_hidden_states, subtok_weights = self.merger(bert_inputs, last_hidden_states, subtoken_positions)
        else :
            # weights used for ensambling the result in the merger (batch, number of sub-tokenized words, weights)
            subtok_weights = None

        # Dropout layers before classifiers
        if self.dropout_enable :
            pooler_output = self.dropoutIntents(pooler_output)
            last_hidden_states = self.dropoutSlots(last_hidden_states)
        
        # Compute logits for intents & logits
        intent = self.intent_out(pooler_output)
        slots = self.slot_out(last_hidden_states)
        
        # We need this for computing the loss (batch_size, classes, seq_len)
        slots = slots.permute(0,2,1) 

        return slots, intent, subtok_weights


class SubtokenMerger(nn.Module):
    """
    Module to merge into a single representation subtokens belonging to the same word.

    Args:
        hidden_size (int): Size of hidden layer.
        num_heads (int): Number of attention heads.
        tokenizer (BertTokenizer): BERT tokenizer.
    """
    def __init__(self, hidden_size, num_heads, tokenizer):
        super(SubtokenMerger, self).__init__()

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.tokenizer = tokenizer
        self.num_heads = num_heads

    def forward(self, bert_inputs, token_embeddings, subtoken_positions):
        """
        Forward pass for subtoken merging.

        Args:
            bert_inputs (dict): Input for BERT.
            token_embeddings (torch.Tensor): Token embeddings.
            subtoken_positions (list): Subtoken positions.

        Returns:
            tuple: Unified token embeddings and subtoken weights.
        """
        batch_size = token_embeddings.shape[0]
        hidden_size = token_embeddings.shape[-1]

        # compute mapping between words and corresponding subtokens
        word_tok_map = self.get_subtok_map(bert_inputs, subtoken_positions, batch_size)

        # use the mapping to get the relative tensors
        subtokens = self.get_subtokens(token_embeddings, word_tok_map, batch_size, hidden_size) # shape : (batch_size, subtokenized_words_count, subtok_count, hidden_size)
        subtokens = subtokens.to(token_embeddings.device)
        
        subtokenized_words_count = subtokens.shape[1]
        subtok_count = subtokens.shape[2]

        # Reshape to (batch_size * subtokenized_words_count, subtok_count, hidden_size) to prepare for multihead self-attention 
        attn_input = subtokens.view(-1, subtok_count, hidden_size)
        # compute attention mask and padding for the attention layer
        attn_padd = (subtokens.sum(dim=-1) != 0).float().view(-1, subtok_count)
        attn_mask = attn_padd.clone().unsqueeze(-1)
        attn_mask = attn_mask.expand(-1, -1, subtok_count) * attn_mask.transpose(1, 2).expand(-1, subtok_count, -1)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, subtok_count, subtok_count) # expand to all attention heads

        # Apply multihead attention
        # 
        _ , attn_weights = self.self_attention(attn_input, attn_input, attn_input, key_padding_mask=attn_padd, attn_mask=attn_mask, need_weights=True)
    
        # take mean over heads to restore (batch_size, subtok_count, subtok_count)
        attn_mask = attn_mask.view(-1, self.num_heads, subtok_count, subtok_count).mean(dim=1)
        # mask attention weights and sum over subtok dimension
        masked_weights = (attn_mask * attn_weights).sum(dim=1) 
        # normmalize
        subtok_contributions = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
        subtok_contributions = subtok_contributions.view(batch_size, subtokenized_words_count, subtok_count)

        # Substitute subtokens with unified representation
        new_token_embeddings = token_embeddings.clone()
        for batch_idx, mappings in enumerate(word_tok_map):
            for word_idx, mapping in enumerate(mappings):
                # weighted sum of subtokens by the attention weights
                coefficients = subtok_contributions[batch_idx, word_idx, :len(mapping)].unsqueeze(-1)
                unified_word = (token_embeddings[batch_idx, mapping] * coefficients).sum(dim=0)
                
                # Substitute the first subtoken with the unified representation
                new_token_embeddings[batch_idx, mapping[0]] = unified_word  
                # Zero all the other subtokens (not really necessary as they're masked in the loss function)    
                new_token_embeddings[batch_idx, mapping[1:]] = 0.0  

        # Format better attention weights for validate later results
        subtok_weights = word_tok_map.copy()
        for batch_idx, mapping in enumerate(word_tok_map):
            if len(mapping) > 0:
                for word_idx, word in enumerate(mapping) :
                    # subtok_weights = (batch_size, subtokenized_words_count, subtok_count)
                    subtok_weights[batch_idx][word_idx] = subtok_contributions[batch_idx, word_idx, :len(word)].tolist()
            else :
                subtok_weights[batch_idx] = []

        # return the new embeddings and the attention weights
        return new_token_embeddings, subtok_weights
    

    def get_subtok_map (self, bert_inputs, subtoken_positions, batch_size):
        """
        Compute mapping between words and corresponding subtokens.

        Args:
            bert_inputs (dict): Input for BERT.
            subtoken_positions (list): Subtoken positions.
            batch_size (int): Size of the batch.

        Returns:
            list: Mapping of words to subtokens.
        """
        word_tok_map = [[] for _ in range(batch_size)]

        for batch_id in range(batch_size):
            # if the sequence contains subtokens
            if len(subtoken_positions[batch_id]) > 0:
                subtoks = sorted(subtoken_positions[batch_id])

                # divide the sequence such that each element in the list is another list
                # storing positions of subtokens related to the word
                # ex :
                #   before : [1, 2, 5, 6, 7]           (subtokens positions in the sentence)
                #   after  : [[0, 1, 2], [4, 5, 6, 7]] (divided in words adding subtoken heads)
                word_tok_map[batch_id].append([subtoks[0]-1, subtoks[0]])
                for i in range(1, len(subtoks)):
                    if subtoks[i] != subtoks[i-1] + 1:
                        word_tok_map[batch_id].append([subtoks[i]-1])
                    word_tok_map[batch_id][-1].append(subtoks[i])

        return word_tok_map
    

    def get_subtokens (self, subtoken_embeddings, word_tok_map, batch_size, hidden_size):
        """
        Extract subtoken's BERT output from the batch.

        Args:
            subtoken_embeddings (torch.Tensor): Subtoken embeddings.
            word_tok_map (list): Mapping of words to subtokens.
            batch_size (int): Size of the batch.
            hidden_size (int): Size of hidden layer.

        Returns:
            torch.Tensor: Padded subtoken tensor.
        """
        # get the maximum number of subtokens in a word
        subtok_count = max([len(mapping) for batch in word_tok_map for mapping in batch])

        ensambler_input = []
        for batch_idx in range(batch_size):
            # each element of the list is a tensor of shape (n_subtokens, hidden_size), where
            # n_subtokens is the number of subtokens in which in the word has been decomposed
            sequence_word_subtok = [subtoken_embeddings[batch_idx, mapping] for mapping in word_tok_map[batch_idx]]

            if len(sequence_word_subtok) > 0:
                sequence_word_subtok_padded = torch.stack([torch.cat([word, torch.zeros((subtok_count - word.shape[0], hidden_size), device=subtoken_embeddings.device )], dim=0) for word in sequence_word_subtok], dim=0)
                # pad the sequence of subtokens to have a tensor of shape (n_words, max_subtokens, hidden_size)
                ensambler_input.append(sequence_word_subtok_padded)
            else :
                # if the sequence has no sub-tokens, append a zero mask
                ensambler_input.append(torch.zeros(1, subtok_count, hidden_size))
        
        ensambler_input = pad_sequence(ensambler_input, batch_first=True, padding_value=0)

        return ensambler_input