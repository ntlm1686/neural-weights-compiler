# GPT-6.1 Terra blind weight reading, 2026-11-03

Three child agents were spawned with model `gpt-6.1-terra` and `fork_turns="none"`. Each received a fresh task instruction naming only its assigned files, forbidding other repository files and web searches, and asking it to report its observations, answer, confidence and tools. This is zero **conversation-history** context, not a guarantee against knowledge from pretraining or other inherited system instructions. The agents were not given the previous answers or the accompanying README. The results below summarize their final responses, not a full transcript of internal reasoning.

The execution harness differed from the original no-tool experiments. The text files were accessed via shell reading tools; output was truncated and the first agent subsequently read selected portions. This does not reproduce an experiment in which the entire text is delivered in one tool-free prompt. The first agent was assigned both game files in the same session, so those two judgments are not independent. The third agent used NumPy weight-only operations. Its answers have not been independently validated with the unavailable original training metadata.

## Experiment 1: grid-game policy weights

Files: `blind_tasks/exp1_network_A.txt` and `blind_tasks/exp1_network_B.txt`.

- A: **Pong or Breakout-like bottom-paddle game**, confidence approximately 45%. The agent saw strong weights at the lower grid for the controlled player and moving objects, and output action columns consistent with left/right control. It did not resolve Pong versus Breakout or establish the reward mechanism.
- B: **Snake**, confidence 25%, with other four-direction chase/avoidance games plausible. No verified ground-truth label for B is included in this repository.

The first combined read was truncated. The agent reported using `cat`, `sed` and `rg` to inspect selected file sections; it reported no computation, forward pass, external search or writes.

## Experiment 2: diffusion model

File: `blind_tasks/exp2_diffusion_mystery.txt`.

The agent guessed **centered handwritten digits**, confidence 35%, while explicitly acknowledging that other centered shapes could fit. It cited small border weights and dark border biases, with stronger central weights and some output biases near zero. The earlier repository report instead says filled upright triangles; the published article describes the triangle model as producing filled shapes. The local file's identity with a particular published checkpoint has not been independently verified.

The agent reported using `cat` only and receiving truncated output, with no forward computation, sampling or external search. On follow-up it confirmed that the quoted output biases were in the portion it received. File access itself does not meet the prompt's literal no-tools condition.

## Experiment 3: four middle Transformer blocks

Files: `blind_tasks/exp3_prompt.txt` and `blind_tasks/exp3_blocks_mystery.npz`.

The agent's best guess was **English, shuffled**: `python=0.34, romaji=0.30, english=0.36; as is=0.45, shuffled=0.55`. It said the evidence was weak and could not reliably classify the heads; L0H4 was a tentative broad-head candidate, with the others unclear. Its stated observations included small L0H4 Q/K norms and negative normalized OV traces, but it did not claim those statistics prove corpus or ordering.

It reported using NumPy on the specified weights only, including products, norms and traces, with no model/layer forward pass and no web search. The prior repository report labels this file Python, as-is, and predicts copy heads L2H1, L3H0, L3H1 and L3H5. The original article's Python case reports true copy heads L2H1, L3H1 and L3H5; a match of this local file to that exact published checkpoint is not independently established. Thus the earlier report's extra L3H0 should not be presented as a verified correct copy-head identification.

## Input fingerprints

SHA-256 of file bytes:

```text
ba1d97cfcb585f0ccab628ed1ef983b230d897b97ae6ff10f2de6eac6d390b31  exp1_network_A.txt
1c75e66aab1e611bdb07a87d6f70e3d2ba3d149e58f259b5ad7a485661244acf  exp1_network_B.txt
d8533653617af859a80f105b8a6360efbf7331be41f85f97212b75fcc63da897  exp2_diffusion_mystery.txt
1d6122b6eb6505820637edcc330e863baa459fa05cc7ca93319a615dcf8c39c3  exp3_prompt.txt
77ed8f7d418619c9390da0abea19de2a05d6e2200d8484473deafee15a6fd43f  exp3_blocks_mystery.npz
```
