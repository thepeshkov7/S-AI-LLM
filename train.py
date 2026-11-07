from pathlib import Path
import torch
from model import TinyLLM

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
data = Path("data/train.txt").read_bytes()
context = 64
if len(data) < context + 2:
    raise ValueError("Training text is too short")
tokens = torch.tensor(list(data), dtype=torch.long, device=device)
model = TinyLLM(context=context).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(301):
    starts = torch.randint(0, len(tokens) - context - 1, (16,), device=device)
    x = torch.stack([tokens[i:i + context] for i in starts])
    y = torch.stack([tokens[i + 1:i + context + 1] for i in starts])
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step} loss={loss.item():.4f} device={device}")

torch.save({"model": model.cpu().state_dict(), "context": context}, "s_ai_0_1.pt")
print("Saved s_ai_0_1.pt")
