
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer, BertModel

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
        attn_mask = (subtokens.sum(dim=-1) != 0).view(-1, subtok_count).float().unsqueeze(-1)
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
                sequence_word_subtok_padded = torch.stack([torch.cat([word, torch.zeros(subtok_count - word.shape[0], hidden_size)], dim=0) for word in sequence_word_subtok], dim=0)
                # pad the sequence of subtokens to have a tensor of shape (n_words, max_subtokens, hidden_size)
                ensambler_input.append(sequence_word_subtok_padded)
            else :
                # if the sequence has no sub-tokens, append a zero mask
                ensambler_input.append(torch.zeros(1, subtok_count, hidden_size))
        
        ensambler_input = pad_sequence(ensambler_input, batch_first=True, padding_value=0)

        return ensambler_input



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(["StarLord was StarLordStar", "I saw a man with a telescope",  "I didn't StarLordStarLord"], return_tensors="pt", padding=True)
    
    hidden_size =768
    num_heads = 12

    output = model(**inputs)
    subtoken_merger = SubtokenMerger(hidden_size, num_heads, tokenizer)
    subtoken_merger(inputs, output.last_hidden_state)