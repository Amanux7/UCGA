# demo_math.py   ← Save this file in C:\Users\91703\UCGA
import torch
import sys
sys.path.insert(0, ".")

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # python 2 or non-standard stream


from ucga.ucga_model import UCGAModel
from ucga.encoders.pretrained_encoder import PretrainedTextEncoder

print("🚀 UCGA Recursive Reasoning Demo (T=1 vs T=3)\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

problem = "A train leaves at 8:00 AM at 60 km/h. Another leaves at 9:00 AM at 80 km/h in the same direction. When does the second train catch the first?"

print(f"Problem: {problem}\n")

# ====================== T=1 ======================
print("="*65)
print("🟡 T = 1   (Single quick pass)")
model1 = UCGAModel(input_dim=128, state_dim=128, cognitive_steps=1, reasoning_steps=2).to(device)
encoder = PretrainedTextEncoder(output_dim=128).to(device)

x = encoder([problem]).to(device)

with torch.no_grad():
    out1, meta1 = model1(x, return_meta=True)

print("Answer (T=1)     :", out1)
print("Confidence       :", [round(c, 3) for c in meta1.get("confidences", [])])
print("Corrections      :", meta1.get("corrections", 0))
print("Steps            :", len(meta1.get("confidences", [])))

# ====================== T=3 ======================
print("\n" + "="*65)
print("🔴 T = 3   (Thinks longer + self-correction)")
model3 = UCGAModel(input_dim=128, state_dim=128, cognitive_steps=3, reasoning_steps=2).to(device)

with torch.no_grad():
    out3, meta3 = model3(x, return_meta=True)

print("Answer (T=3)     :", out3)
print("Confidence       :", [round(c, 3) for c in meta3.get("confidences", [])])
print("Corrections      :", meta3.get("corrections", 0))
print("Steps            :", len(meta3.get("confidences", [])))

print("\n🎥 Screen-record this output now for your X video!")