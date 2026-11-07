# S AI 0.1 LLM

An experimental Russian and English decoder-only Transformer. The model and its byte-level BPE tokenizer start from scratch. Inference loads local PyTorch weights and runs autoregressive token sampling.

The first checkpoint learns to continue literary prose. It has a 96-token context, has no instruction tuning, and is suitable for inference and integration testing. The application still needs a chat-quality checkpoint and end-to-end staging tests before a provider cutover.

## Reproduce the CPU experiment

Use Python 3.12. For Linux x86-64 CPU, install the PyTorch CPU wheel first.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt -r requirements-data.txt
python prepare_corpus.py
python train.py --device cpu --width 96 --heads 4 --layers 3 --context 96 \
  --batch-size 16 --steps 1800 --eval-every 300 --threads 4 \
  --checkpoint runs/s_ai_0_1.pt
python generate.py --checkpoint runs/s_ai_0_1.pt \
  --prompt "Однажды утром я увидел" --seed 42 --max-new-tokens 40
```

`data/sources.json` records the works, split assignments, source URLs and checksums of the normalized text. The corpus consists of public-domain originals from [Project Gutenberg](https://www.gutenberg.org/policy/license) and [Russian Wikisource](https://ru.wikisource.org/). The Wikisource entries use fixed revisions. Preparation strips site and licensing boilerplate and checks each normalized text against its manifest. Cache downloads in `corpus/sources/` to protect reproducibility from changes at the source.

Training uses separate works for training, validation and test. Only the training split fits the tokenizer. Validation loss selects the checkpoint and test loss is calculated after selection. Results include per-language perplexity, counts, data checksums, hyperparameters and the random seed. Identical runs depend on matching software and hardware. For a different legally usable corpus, provide `train.jsonl`, `validation.jsonl` and `test.jsonl` with `id`, `language` and `text` fields, then pass `--data-dir`. Use `--help` for model dimensions and CUDA options.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the inference API and deployment procedure.

Run the causal attention, legacy checkpoint and readiness regression tests with:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

## Continue with instruction tuning

Run this experiment after completing the base training above. Keep the base weights and use a separate output directory for the instruction checkpoint.

```bash
python finetune.py --base runs/s_ai_0_1.pt --data-dir corpus --output runs/instruction-v1
python generate.py --checkpoint runs/instruction-v1/s_ai_0_1_instruct.pt \
  --instruction "Цвет: синий. Форма: круг. Назови форму." --top-k 1 --max-new-tokens 16
```

This stage starts from the base weights and vocabulary. It trains the model on generated Russian and English examples for small-integer addition, matching words, selecting a word by position, and extracting a field from a short description. Loss for instruction batches is applied to the answer and end token. The sampler gives equal weight to each task and balances yes and no within word comparison. By default, every fifth update uses original training prose to reduce forgetting.

The split is assigned to each input group within a task before rendering the two languages, template variants and input order. A separate validation split selects the checkpoint by answer loss. Two test splits measure greedy exact match, one with new inputs and familiar wording and one with new inputs and withheld wording. The script also measures language-model loss on the original held-out prose validation set. Open-ended diagnostics illustrate limitations, and their outputs are kept separate from exact-match scores.

`metrics.json` records counts and scores. `predictions.json` contains actual continuations before and after instruction tuning. In the API, send the output of `format_instruction()` in the existing `prompt` field and read `completion`. The model generates responses from the checkpoint. The synthetic task rules only create training and evaluation data.
