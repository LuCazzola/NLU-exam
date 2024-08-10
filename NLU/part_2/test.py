import torch.nn as nn
from transformers import BertTokenizer, BertModel

import torch
import torch.nn as nn

class SimpleMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SimpleMultiheadAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, attn_weights = self.self_attention(x, x, x)
        return attn_output

def get_num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

hidden_size = 768
input_tensor = torch.rand(2, 10, hidden_size)  # Batch size of 2, sequence length of 10

# Initialize models with different number of heads
model_4_heads = SimpleMultiheadAttention(hidden_size, 4)
model_8_heads = SimpleMultiheadAttention(hidden_size, 8)

total_params_4, trainable_params_4 = get_num_parameters(model_4_heads)
total_params_8, trainable_params_8 = get_num_parameters(model_8_heads)

print(f'Model with 4 heads: Total parameters: {total_params_4}, Trainable parameters: {trainable_params_4}')
print(f'Model with 8 heads: Total parameters: {total_params_8}, Trainable parameters: {trainable_params_8}')

# Verify the detailed parameter counts
for name, param in model_4_heads.named_parameters():
    if param.requires_grad:
        print(f"4 heads - {name}: {param.numel()}")

for name, param in model_8_heads.named_parameters():
    if param.requires_grad:
        print(f"8 heads - {name}: {param.numel()}")