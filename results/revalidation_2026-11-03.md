# Revalidation, 2026-11-03

Environment: Python 3, NumPy 2.4.6, `OPENBLAS_NUM_THREADS=1`. Evaluation used fixed seeds defined in the scripts. The following values are from this rerun, not independent replications of the original paper's evaluation harness or development process.

- `python -m unittest discover -s tests -v`: 2 regression tests passed (duplicate-value sorting and reverse-pivot shortest-path chain).
- `python stage2_transformer_benchmark.py --output-json results/transformer_revalidation.json`: 500/500 per task and configuration; shortest path uses 20 layers and a synchronous Bellman–Ford reference. The JSON includes exact parameter counts.
- `python llm_compiled_unseen_tasks.py`: both tasks 500/500; architecture `d=128, L=6, H=4`, 1,186,560 parameters.
- `python stage3_circuit_grafting.py`: host 500/500 before and after; max observed host-output drift 0; addition 500/500 random and 100/100 generated carry-chain cases. The host was created with reserved registers and inactive units.
- `python stage1_mlp_benchmark.py`: 200 episodes per game, seed 0, sticky probability 0.1, episode cap 10,000. Breakout 546.81 ± 15.29, 815 nonzeros, average length 7810.1; Seaquest 59.80 ± 3.49, 852 nonzeros, average length 2044.4; Asterix 64.85 ± 2.17, 637 nonzeros, average length 986.6. ± is episode standard error.
- `python stage1_5_go_compiler.py --checkpoints /tmp/go_checkpoints --games 200 --seed 31337`: `budget_0` 82.0% (black 75%, white 89%); `ppo_s0` 63.5% (56%, 71%); `ppo_s1` 66.5% (74%, 59%); `ppo_s2` 79.5% (84%, 75%). These checkpoint files were already present outside the repository and their original source was not independently checked.

Checkpoint fingerprints below are SHA-256 of the concatenation, in filename-sorted order, of each `.npy` filename encoded as UTF-8, a zero byte, and that file's bytes. They identify the tested local files but do not verify their provenance:

```text
budget_0 28e548ec24d88648b90f0a27642cd1778fe2796ae2c0745e86bacf0386b6d962
ppo_s0   31cd25ddc168135e3d2f5392edc671e9832bdacdd72314d272d45d1fedc1ece1
ppo_s1   2834abd687c19ae516f4dd4bed1ecd2736ff61ca46b8c681dab32ecb09c65209
ppo_s2   7536720e9f3b861a26f9c63c11883b40df209c5d27b8322f400992ce78ff031f
```
