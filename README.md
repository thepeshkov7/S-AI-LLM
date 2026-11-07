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

## S AI 0.2 bilingual CPU pilot

This second experiment initializes new weights and a new tokenizer. It samples half of each update's tokens in Russian and half in English. The article-level validation and test splits are disjoint from pretraining, and the generated chat benchmark stays outside all training splits.

```bash
python prepare_pilot.py
python prepare_translations.py
python make_chat_eval.py --output runs/wiki-pilot/chat-eval-v1.jsonl
python train.py --data-dir corpus/wiki-pilot --checkpoint runs/wiki-pilot/s_ai_0_2_pilot.pt \
  --device cpu --width 160 --heads 5 --layers 5 --context 256 --vocab-size 4096 \
  --batch-size 8 --steps 650 --eval-every 200 --threads 6 --learning-rate 0.0009 --balance-languages
python tune_pilot.py --base runs/wiki-pilot/s_ai_0_2_pilot.pt --wiki corpus/wiki-pilot \
  --translations corpus/translation-pilot --output runs/wiki-pilot/s_ai_0_2_pilot_instruct.pt
python evaluate_chat.py --checkpoint runs/wiki-pilot/s_ai_0_2_pilot_instruct.pt \
  --benchmark runs/wiki-pilot/chat-eval-v1.jsonl --output runs/wiki-pilot/chat-instruct-pilot.json
```

The [Wikipedia snapshot](https://huggingface.co/datasets/wikimedia/wikipedia) provides article text with original article links and attribution. The [Tatoeba project and contributors](https://tatoeba.org/en/terms_of_use) provide directly linked translations. The source manifests preserve article URLs and individual sentence authors, IDs, license URLs and checksums. Keep the API caches or the normalized snapshots to reproduce exact sample contents. Tatoeba's live API can change its results.

The chat benchmark is a frozen 400-item local suite with 200 cases per language. Half are automatically scored and half have reference criteria for separate manual assessment. Its v1 SHA-256 is `4755f3b03c068c856f89617cf8a17fdf4e2b2b2129496132a56b9bd929fd69e0`. The evaluation script produces predictions for an offline blind comparison. It makes no external model requests.

A future GPU candidate is width 384, 12 layers, six heads, a 24,000-token vocabulary and a 1,024-token context. The training command exposes optional CUDA BF16 and gradient accumulation. Before a full run, move the corpus to disk-backed token shards, add resumable training state, measure throughput and memory on the actual GPU, and agree on a cost cap. The CPU pilot uses a small in-memory corpus and saved inference checkpoints.

## Bilingual memory experiment

```bash
python tune_memory.py --base runs/wiki-pilot/s_ai_0_2_pilot_instruct.pt \
  --output runs/memory/s_ai_0_2_memory.pt
```

This warm-started CPU experiment teaches the Transformer to match questions against account-memory keys and measures both recall and abstention. A separate path generates short replies in the requested language. See [DEPLOYMENT.md](DEPLOYMENT.md#experimental-bilingual-memory-route) for the private API and the S AI server adapter.

## Bilingual chat tuning from the memory checkpoint

This experiment continues the S AI 0.2 Memory weights and reuses the original
tokenizer. It mixes authored everyday questions, generated short arithmetic
and context exercises, attributed translation pairs from the translation
pilot, and the existing fictional memory examples. A separate validation set
selects the checkpoint. Its everyday portion holds out wording for familiar
topics. The final benchmark also includes separate questions on new topics.

```bash
python tune_chat.py --base runs/memory/s_ai_0_2_memory.pt \
  --translations corpus/translation-pilot --output runs/chat/s_ai_0_2_chat.pt
python evaluate_chat.py --checkpoint runs/chat/s_ai_0_2_chat.pt \
  --benchmark data/chat-probe-v2.jsonl --language-prefix --output runs/chat/probe.json
```

The probe includes reference criteria for manual review. To compare with
the earlier chat benchmark, run `evaluate_chat.py` on the same frozen file
for both checkpoints with `--language-prefix`. The existing private
`/chat` endpoint uses that language prefix for one-turn generation.

## Packed bilingual pretraining and reviewed dialogue tuning

The packed trainer reads token streams from disk and gives each language half
of every batch. It checks each stream against its manifest. Documents are
separated by EOS tokens. Training windows can span those boundaries. Validation
and test streams come from held-out documents. The sampler, optimizer, latest
weights and best validation weights are saved together at evaluation points
and explicit pauses. A run resumes with the same data and configuration.

To build the corpus, sample pages from the dated Wikipedia snapshot and prepare
the public-domain book corpus with the original preparation script. The data
viewer can rate-limit requests. After completing a sufficient bilingual cache,
use `--cached-only` to build the available pages. The corpus manifest records
the exact offsets and article attribution used by the run.

```bash
python prepare_pilot.py --sample-pages 48 --output corpus/wiki-wide
python prepare_corpus.py --output corpus/books
python packed_corpus.py --source corpus/wiki-wide --source corpus/books \
  --output corpus/packed-wide --vocab-size 6144
python pretrain_packed.py --corpus corpus/packed-wide --output runs/wide/base \
  --device cpu --steps 1600 --stop-after 100 --eval-every 200 --eval-windows 48
python pretrain_packed.py --corpus corpus/packed-wide --output runs/wide/base \
  --device cpu --steps 1600 --resume --eval-every 200 --eval-windows 48
```

Download the [OASST2 ready messages export](https://huggingface.co/datasets/OpenAssistant/oasst2/blob/main/2023-11-05_oasst2_ready.messages.jsonl.gz).
The preparation script filters by review, rank and safety labels and excludes
identical normalized benchmark prompts. It keeps complete one- or two-turn
dialogues and separates train, validation and test by conversation tree. Tuning
uses loss on the final assistant turn only. The vocabulary is learned from the
pretraining corpus.

```bash
python make_chat_eval.py --output runs/wide/chat-eval-v1.jsonl
python - <<'PY'
import json
from pathlib import Path

run = Path("runs/wide")
rows = (run / "chat-eval-v1.jsonl").read_text(encoding="utf-8").splitlines()
automatic = [json.loads(row) for row in rows if json.loads(row)["score"] != "human"]
probe = [json.loads(row) for row in Path("data/chat-probe-v2.jsonl").read_text(encoding="utf-8").splitlines()]
(run / "combined.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in automatic + probe),
    encoding="utf-8",
)
PY
python prepare_oasst.py --source corpus/source/oasst2.messages.jsonl.gz \
  --output corpus/oasst2-clean --exclude runs/wide/combined.jsonl
python tune_oasst.py --base runs/wide/base/pretrained.pt \
  --dialogues corpus/oasst2-clean --output runs/wide/dialogue.pt
python evaluate_chat.py --checkpoint runs/wide/dialogue.pt \
  --benchmark runs/wide/combined.jsonl --language-prefix --output runs/wide/chat-dialogue.json
```

### Bilingual chat curriculum

Start from the 0.3 Dialogue checkpoint to retain its tokenizer and bilingual
pretraining. Mix reviewed dialogue and Tatoeba examples with authored everyday
questions and reproducible tasks covering arithmetic, fact extraction, order,
follow-ups and JSON. Each batch is balanced between Russian and English.
The trainer selects weights by validation loss averaged across task families
and languages. Use the separate generation benchmarks to assess chat quality.

```bash
python prepare_bilingual.py --dialogues corpus/oasst2-clean \
  --translations corpus/translation-pilot --output corpus/bilingual \
  --exclude runs/wide/combined.jsonl --exclude runs/wide/chat-eval-v1.jsonl
python tune_bilingual.py --base runs/wide/dialogue.pt \
  --curriculum corpus/bilingual --output runs/bilingual/s_ai_0_4_bilingual.pt
python evaluate_chat.py --checkpoint runs/bilingual/s_ai_0_4_bilingual.pt \
  --benchmark runs/wide/combined.jsonl --language-prefix --output runs/bilingual/chat.json
python evaluate_chat.py --checkpoint runs/bilingual/s_ai_0_4_bilingual.pt \
  --benchmark data/bilingual-probe-v3.jsonl --language-prefix --max-new-tokens 64 \
  --output runs/bilingual/probe.json
```

The new probe includes everyday paraphrases, novel tasks, language changes and
follow-ups. Score its answers against the attached rubrics. The research
checkpoint enables text generation through the private `/chat` route. Account
memory recall remains in the app's authenticated server adapter.

The 2,400-step CPU run improved the frozen automatic benchmark from 0/200
to 48/200, with 24/100 in each language. It answered all 40 time updates
correctly. Manual review found 3/30 correct on the original probe and 0/28 on
the new probe. The validation loss for everyday questions and reviewed dialogue
increased in both languages. Keep this checkpoint on a local test service.

### Bilingual copying and dialogue expansion

The next stage continues from the bilingual checkpoint. Its copy head learns
three modes for each output token. It can generate from the vocabulary, copy a
visible token, or copy the token with its leading space adjusted. This uses the
same vocabulary and causal attention as the previous checkpoint. The private
chat route loads both architectures.

The expanded corpus keeps the earlier examples for replay and adds reviewed
dialogues, everyday and language-switch requests, and variations of arithmetic,
fact extraction, ordering, JSON and conversational context. The old benchmarks
are excluded by normalized user turn. The training validation objective weights
everyday dialogue more heavily. The original evaluations and the new transfer
probe use greedy decoding on the selected checkpoint.

```bash
python prepare_grounded.py --previous corpus/bilingual \
  --oasst corpus/source/oasst2.messages.jsonl.gz --base runs/bilingual/s_ai_0_4_bilingual.pt \
  --exclude runs/wide/combined.jsonl --exclude runs/wide/chat-eval-v1.jsonl \
  --exclude data/bilingual-probe-v3.jsonl --output corpus/grounded
python tune_bilingual.py --with-copy-head --base runs/bilingual/s_ai_0_4_bilingual.pt \
  --curriculum corpus/grounded --output runs/grounded/s_ai_0_5_grounded.pt \
  --steps 3200 --batch-size 12 --eval-every 400 --learning-rate 0.00055
python evaluate_chat.py --checkpoint runs/grounded/s_ai_0_5_grounded.pt \
  --benchmark runs/wide/combined.jsonl --language-prefix --output runs/grounded/chat-v2.json
python evaluate_chat.py --checkpoint runs/grounded/s_ai_0_5_grounded.pt \
  --benchmark data/bilingual-probe-v3.jsonl --language-prefix --max-new-tokens 64 \
  --output runs/grounded/probe-v3.json
python evaluate_chat.py --checkpoint runs/grounded/s_ai_0_5_grounded.pt \
  --benchmark data/grounded-probe-v4.jsonl --language-prefix --max-new-tokens 64 \
  --output runs/grounded/probe-v4.json
```

An offline decoder experiment trains a small readout on the frozen model. It
selects entire spans of source text for a requested JSON object with `title`
and `time` keys. In arithmetic mode the readout classifies the operation and
code computes the result for two supported integers. Use `--typed-only` to
keep autoregressive generation for all other cases. Save and compare the
decoder results separately from the direct model outputs.

```bash
python span_chat.py train --with-arithmetic --base runs/grounded/s_ai_0_5_grounded.pt \
  --curriculum corpus/grounded --output runs/grounded/typed_head.pt --steps 1200 --threads 5
python span_chat.py score --typed-only --base runs/grounded/s_ai_0_5_grounded.pt \
  --head runs/grounded/typed_head.pt --benchmark runs/wide/combined.jsonl \
  --output runs/grounded/chat-v2-typed.json
```

Direct generation scored 26/100 in Russian and 27/100 in English on the frozen
automatic test. The offline decoder increased those scores to 51/100 and
56/100, including 40/40 JSON and 14/40 arithmetic cases across both languages.
The new transfer probe scored 1/18 with the decoder. Manual review found one
correct answer per language out of 15 on the original probe and 0/14 per
language on the new probe. The six release targets remain unmet.

### GPU throughput measurement

Prepare a larger corpus with a 24,000-token vocabulary before measuring the
40.1M-parameter candidate. With 65,536 tokens per optimizer step, 7,630 steps
would process about 500M tokens, including repeated samples. Measure the actual
device in a short run before committing to the full schedule. The example
pauses after 100 updates while retaining the full learning-rate schedule.

```bash
python pretrain_packed.py --corpus corpus/packed-gpu --output runs/gpu \
  --device cuda --bf16 --width 384 --heads 6 --layers 12 --context 1024 \
  --batch-size 8 --gradient-accumulation 8 --steps 7630 \
  --eval-every 100 --eval-windows 48 --stop-after 100
```
