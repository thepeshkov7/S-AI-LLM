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
