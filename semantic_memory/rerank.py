"""Qwen3-Reranker scorer per the model card: chat-template prefix/suffix, P(yes) from the last-token yes/no logits."""
import time

import torch

DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class QwenReranker:
    def __init__(self, model_dir, device='cuda', max_length=1024, instruction=DEFAULT_INSTRUCTION, dtype=torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = str(model_dir); self.device = device; self.max_length = max_length; self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        self.yes = self.tokenizer.convert_tokens_to_ids('yes'); self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False); self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.score_seconds = 0.; self.scored = 0

    def format(self, query, doc):
        return f'<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}'

    @torch.no_grad()
    def score(self, query, docs, batch_size=16):
        pairs = [self.format(query, d) for d in docs]
        scores = []
        start = time.perf_counter()
        for at in range(0, len(pairs), batch_size):
            chunk = pairs[at:at + batch_size]
            enc = self.tokenizer(chunk, padding=False, truncation='longest_first', return_attention_mask=False,
                                 max_length=self.max_length - len(self.prefix) - len(self.suffix))
            enc['input_ids'] = [self.prefix + ids + self.suffix for ids in enc['input_ids']]
            batch = self.tokenizer.pad(enc, padding=True, return_tensors='pt').to(self.device)
            # Only the last position is needed: run the base model and project a single hidden vector through lm_head
            # instead of materialising [batch, seq, vocab] logits for every position (which exhausted 8 GiB).
            hidden = self.model.model(**batch).last_hidden_state[:, -1, :]
            logits = self.model.lm_head(hidden)
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            scores.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().tolist())
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.score_seconds += time.perf_counter() - start; self.scored += len(pairs)
        return scores
