"""Generate one no-feedback response per MinAtar game and authoring condition."""
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / '.local_models/qwen38-27b'
OUT = ROOT / 'experiments/minatar_model_comparison'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--force-final', action='store_true', help='continue a saved thinking output after closing the thinking block')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--tp', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--no-eager', action='store_true', help='allow CUDA graphs (faster decode)')
    parser.add_argument('--tag', default=None, help='output name tag; defaults to thinking_/forced_final_/empty')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    llm = LLM(model=str(MODEL), trust_remote_code=True, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={'image': 0, 'video': 0}, enforce_eager=not args.no_eager)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    source = (ROOT / 'neural_compiler/ir.py').read_text() + '\n' + (ROOT / 'neural_compiler/mlp_backend.py').read_text()
    prompts = []
    names = []
    for game in ('breakout', 'seaquest', 'asterix'):
        taskfile = ROOT / 'blind_tasks/exp5_breakout_prompt.txt' if game == 'breakout' else OUT / (game + '_prompt.txt')
        task = taskfile.read_text()
        for mode in ('alone', 'compiler'):
            if mode == 'compiler':
                instruction = ('Use the following NeuralCompiler IR and backend to build a policy. '
                    'Return one complete Python code block defining build_policy() that returns MLPCompilerBackend.compile(program). '
                    'Do not train, test, run model forward passes or use feedback.\n\n' + source)
            elif game != 'breakout':
                instruction = ('Return one complete Python code block defining build_policy() returning an ExecutableMLP '
                    'constructed from directly written NumPy weight arrays. Do not use the compiler.\n')
            else:
                instruction = 'Follow the edits-block format in the task exactly.\n'
            messages = [{'role': 'user', 'content': task + '\n\n' + instruction}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking or args.force_final)
            if args.force_final:
                thinking_path = OUT / f'qwen38_thinking_{mode}_{game}.txt'
                prompt += thinking_path.read_text() + '\n</think>\n'
            prompts.append(prompt)
            tag = args.tag if args.tag is not None else 'forced_final_' if args.force_final else 'thinking_' if args.thinking else ''
            names.append(f'qwen38_{tag}{mode}_{game}.txt')
    outputs = llm.generate(prompts, params)
    log = []
    for name, output in zip(names, outputs):
        (OUT / name).write_text(output.outputs[0].text, encoding='utf-8')
        log.append({'file': name, 'tokens': len(output.outputs[0].token_ids), 'finish': output.outputs[0].finish_reason,
                    'max_tokens': args.max_tokens, 'thinking': args.thinking, 'force_final': args.force_final})
        print(name, 'tokens=', len(output.outputs[0].token_ids), 'finish=', output.outputs[0].finish_reason, flush=True)
    import json
    (OUT / f'qwen38_{args.tag or "run"}_log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
