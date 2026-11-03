"""Generate one no-feedback response per MinAtar game and authoring condition."""
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / '.local_models/qwen38-27b'
OUT = ROOT / 'experiments/minatar_model_comparison'


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    llm = LLM(model=str(MODEL), trust_remote_code=True, tensor_parallel_size=1,
              max_model_len=16384, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={'image': 0, 'video': 0}, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=4096)
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
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            names.append(f'qwen38_{mode}_{game}.txt')
    outputs = llm.generate(prompts, params)
    for name, output in zip(names, outputs):
        (OUT / name).write_text(output.outputs[0].text, encoding='utf-8')
        print(name, 'tokens=', len(output.outputs[0].token_ids), 'finish=', output.outputs[0].finish_reason, flush=True)


if __name__ == '__main__':
    main()
