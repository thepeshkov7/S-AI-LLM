import argparse
import torch
from model import TinyLLM

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="Привет")
parser.add_argument("--max-new-bytes", type=int, default=120)
args = parser.parse_args()
checkpoint = torch.load("s_ai_0_1.pt", map_location="cpu", weights_only=True)
model = TinyLLM(context=checkpoint["context"])
model.load_state_dict(checkpoint["model"])
model.eval()
ids = list(args.prompt.encode("utf-8"))
with torch.no_grad():
    for _ in range(args.max_new_bytes):
        x = torch.tensor([ids[-model.context:]], dtype=torch.long)
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1] / 0.9, dim=-1)
        ids.append(int(torch.multinomial(probs, 1).item()))
print(bytes(ids).decode("utf-8", errors="replace"))
