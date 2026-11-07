# S AI 0.1 LLM

Experimental decoder-only Transformer language model trained **from random initialization**.

This is a small research prototype, not a ChatGPT replacement. No pretrained weights or external AI API are used.

## Quick start

Requires Python 3.10+ and PyTorch.

```bash
pip install -r requirements.txt
python train.py
python generate.py --prompt "Привет"
```

Training uses a tiny demonstration corpus and CPU by default, or CUDA if available. The generated output will initially be poor. Replace `data/train.txt` with appropriately licensed, curated training text for serious experiments.

Never commit API keys, personal information, or private datasets. The current repository visibility should be reviewed before uploading anything sensitive.

## Next steps

- Curate licensed multilingual data and train a subword tokenizer.
- Evaluate held-out loss, memorization and generation quality.
- Add a guarded inference service and connect it to S AI only after validation.
- Keep production OpenAI integration unchanged until a replacement is ready.
