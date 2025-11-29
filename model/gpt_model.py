import torch.nn as nn
import torch
from layers.transformer_block import TransformerBlock
from layers.layernorm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        cfg = self.cfg
        context_size = cfg["context_length"]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, float("-inf"), logits)

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and idx_next.item() == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx