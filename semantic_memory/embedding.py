"""Qwen3-Embedding encoder per the model card: left padding, last-token pooling, L2 normalisation, query instruction prefix."""
import time

import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), lengths]


class QwenEncoder:
    def __init__(self, model_dir, device='cuda', max_length=512, instruction=DEFAULT_INSTRUCTION, dtype=torch.float16):
        from transformers import AutoModel, AutoTokenizer
        self.name = str(model_dir); self.device = device; self.max_length = max_length; self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
        self.model = AutoModel.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        self.dim = self.model.config.hidden_size
        self.encode_seconds = 0.; self.encoded = 0

    def format_query(self, query):
        return f'Instruct: {self.instruction}\nQuery:{query}' if self.instruction else query

    @torch.no_grad()
    def encode(self, texts, batch_size=32, is_query=False, max_length=None):
        max_length = max_length or self.max_length
        texts = [self.format_query(t) for t in texts] if is_query else list(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        start = time.perf_counter()
        for at in range(0, len(order), batch_size):
            idx = order[at:at + batch_size]
            batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(self.device)
            hidden = self.model(**batch).last_hidden_state
            emb = F.normalize(last_token_pool(hidden, batch['attention_mask']).float(), p=2, dim=1)
            out[idx] = emb.cpu().numpy()
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.encode_seconds += time.perf_counter() - start; self.encoded += len(texts)
        return out

    def vram(self):
        if self.device != 'cuda':
            return None
        return dict(allocated=torch.cuda.memory_allocated(), peak_allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.memory_reserved())
