"""Batched greedy generation with the calibrated local answer model (Qwen3-4B, bitsandbytes nf4, thinking disabled)."""
import time

import torch


class QwenGenerator:
    def __init__(self, model_dir, quant='nf4', device='cuda', max_prompt_tokens=3072):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.name = f'{model_dir}@{quant}'; self.device = device; self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
        if quant == 'nf4':
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
        elif quant == '8bit':
            q = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError('quant must be nf4 or 8bit')
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=q, device_map=device).eval()
        self.generated_tokens = 0; self.prompt_tokens = 0; self.seconds = 0.; self.calls = 0

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=384):
        texts = [self.tokenizer.apply_chat_template([dict(role='user', content=p)], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_prompt_tokens).to(self.device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        torch.cuda.synchronize(); self.seconds += time.perf_counter() - t0; self.calls += 1
        new = out[:, batch['input_ids'].shape[1]:]
        self.prompt_tokens += int(batch['attention_mask'].sum()); self.generated_tokens += int((new != (self.tokenizer.pad_token_id or -1)).sum())
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in new]

    def stats(self):
        return dict(name=self.name, calls=self.calls, prompt_tokens=self.prompt_tokens, generated_tokens=self.generated_tokens, seconds=self.seconds,
                    tokens_per_second=self.generated_tokens / self.seconds if self.seconds else None,
                    vram_peak_allocated=torch.cuda.max_memory_allocated() if self.device == 'cuda' else None)
