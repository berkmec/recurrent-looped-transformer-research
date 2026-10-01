"""G0 answer-model calibration on the local GPU: load Qwen3-4B in 8-bit and 4-bit (bitsandbytes), measure load time,
VRAM, tokens/s and format validity on fixed short prompts (TR+EN). Development-only; no benchmark claim."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = '__RLT_ROOT__/models/Qwen3-4B'
PROMPTS = [
    'What is the capital of France? Answer with one word.',
    "Ali is older than Ayşe. Is Ayşe younger than Ali? Answer yes or no.",
    'A train leaves at 9:00 and arrives at 11:30. How long is the trip? Answer briefly.',
    'List two prime numbers greater than 10.',
    "Türkiye'nin başkenti neresidir? Tek kelimeyle cevap ver.",
    "Ali, Ayşe'den büyüktür. Ayşe, Ali'den küçük müdür? Evet ya da hayır de.",
    'Bir tren 9:00\'da kalkıp 11:30\'da varıyor. Yolculuk ne kadar sürer? Kısa cevap ver.',
    '10\'dan büyük iki asal sayı yaz.',
]


def run(config_name, quant):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if quant == '8bit':
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map='cuda').eval()
    load_s = time.perf_counter() - t0
    vram_after_load = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
    outputs = []
    gen_tokens = 0; gen_seconds = 0.
    for prompt in PROMPTS:
        messages = [dict(role='user', content=prompt)]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok([text], return_tensors='pt').to('cuda')
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        torch.cuda.synchronize(); dt = time.perf_counter() - t1
        new = out[0][inputs['input_ids'].shape[1]:]
        answer = tok.decode(new, skip_special_tokens=True).strip()
        gen_tokens += int(new.numel()); gen_seconds += dt
        outputs.append(dict(prompt=prompt, answer=answer, new_tokens=int(new.numel()), seconds=dt))
    result = dict(config=config_name, quant=quant, load_seconds=load_s, vram_after_load=vram_after_load,
                  vram_peak_allocated=torch.cuda.max_memory_allocated(), vram_peak_reserved=torch.cuda.max_memory_reserved(),
                  tokens_per_second=gen_tokens / gen_seconds if gen_seconds else None, generated_tokens=gen_tokens, outputs=outputs,
                  torch=torch.__version__, thinking='disabled via enable_thinking=False', decoding='greedy, max_new_tokens 48')
    del model; torch.cuda.empty_cache()
    return result


def main():
    out = ROOT / 'results/G0'; out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, quant in [('qwen3-4b-8bit', '8bit'), ('qwen3-4b-nf4', '4bit')]:
        try:
            results.append(run(name, quant))
            print(json.dumps({k: v for k, v in results[-1].items() if k != 'outputs'}, ensure_ascii=False), flush=True)
        except Exception as e:  # keep the failure as data
            results.append(dict(config=name, quant=quant, status='FAILED', error=repr(e)[:500]))
            print('FAILED', name, repr(e)[:300], flush=True)
            torch.cuda.empty_cache()
    (out / 'lm-calibration.json').write_text(json.dumps(dict(model=MODEL, gpu=torch.cuda.get_device_name(0), results=results,
                                                              role='G0 development calibration; not a benchmark'), indent=1, ensure_ascii=False) + '\n', encoding='utf8')
    print('CALIBRATION DONE')


if __name__ == '__main__':
    main()
