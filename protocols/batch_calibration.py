"""D2/D5 (package P5): generator throughput calibration — tokens/s and peak VRAM versus batch size for the pinned nf4 model,
with greedy decoding on fixed prompts. Same-semantics check: batched greedy outputs must equal the batch-1 outputs per prompt
(left padding); any mismatch is reported, never hidden. GPU only; run after the GPU chain. Writes results/G0/batch-calibration.json.
Usage: python -B protocols/batch_calibration.py [--batches 1,2,4,8] [--new-tokens 128] [--prompts 8]"""
import argparse
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ['Explain in two sentences why the sky is blue.', 'List three prime numbers greater than 100 and explain briefly.',
           'Summarise the plot of a heist film in three sentences.', 'What is the capital of Australia? Answer with one word, then justify.',
           'Give a short definition of entropy for a high-school student.', 'Translate "good morning, how are you?" into Turkish and French.',
           'Describe the rules of tic-tac-toe.', 'Name two differences between lists and tuples in Python.']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', default='1,2,4,8'); parser.add_argument('--new-tokens', type=int, default=128); parser.add_argument('--prompts', type=int, default=8)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'G0' / 'batch-calibration.json'))
    args = parser.parse_args()
    import torch
    from language.generator import QwenGenerator
    pins = json.loads((ROOT / 'protocols' / 'G0-pins.json').read_text(encoding='utf8'))
    model_dir = next(v['local_dir'] for k, v in pins['models'].items() if k.endswith('Qwen3-4B'))
    gen = QwenGenerator(model_dir)
    prompts = PROMPTS[:args.prompts]
    reference = None; rows = []
    for b in [int(x) for x in args.batches.split(',')]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter(); outputs = []; before = gen.generated_tokens
        for at in range(0, len(prompts), b):
            outputs.extend(gen.generate(prompts[at:at + b], max_new_tokens=args.new_tokens))
        torch.cuda.synchronize(); seconds = time.perf_counter() - t0; tokens = gen.generated_tokens - before
        if reference is None:
            reference = outputs
        mismatches = sum(1 for a, c in zip(reference, outputs) if a != c)
        rows.append(dict(batch=b, seconds=seconds, generated_tokens=tokens, tokens_per_second=(tokens / seconds) if tokens else None,
                         peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30, mismatches_vs_batch1=mismatches, prompts=len(prompts)))
        print(json.dumps(rows[-1]), flush=True)
    report = dict(model=model_dir, new_tokens=args.new_tokens, prompts=prompts, rows=rows, platform=platform.platform(), torch=torch.__version__,
                  note='same-semantics gate: mismatches_vs_batch1 must be 0 for a batch size to be adopted; nf4 greedy; not a training throughput claim')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
