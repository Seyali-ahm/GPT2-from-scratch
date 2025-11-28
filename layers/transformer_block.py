import torch
import torch.nn as nn
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.att = MultiHeadAttention(d_in=config["emb_dim"],
                                      d_out=config["emb_dim"],
                                      context_length=config["context_length"],
                                      dropout=config["drop_rate"],
                                      num_heads=config["n_heads"],
                                      qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut
        
        
        shortcut = x
        
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        
        return x
    
    
