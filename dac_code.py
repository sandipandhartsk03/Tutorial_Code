import dac
from audiotools import AudioSignal
import torch
import numpy as np

# ------------------------
# Function to count parameters
# ------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

# ------------------------
# Load DAC 24 kHz model
# ------------------------
model_path = dac.utils.download(model_type="24khz")
model = dac.DAC.load(model_path)

# Count parameters before moving to GPU
total, trainable, non_trainable = count_parameters(model)
print("========== MODEL PARAMETERS ==========")
print(f"Total parameters         : {total:,}")
print(f"Trainable parameters     : {trainable:,}")
print(f"Non-trainable parameters : {non_trainable:,}")
print("======================================\n")

model.to("cuda")  # GPU for encoding
torch.cuda.empty_cache()

# ------------------------
# Load audio file
# ------------------------
signal = AudioSignal('/home/sandipandhar/Desktop/DAC-Code/Resampled/train_hindimale_00001_GT.wav')
signal.to(model.device)

# ------------------------
# Encode (light on memory)
# ------------------------
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# ------------------------
# ⚠ DO NOT USE model.decode(z)
# ------------------------

# Use safe streaming compression
signal_cpu = signal.cpu()
compressed = model.compress(signal_cpu)

# Save compressed file
compressed.save("compressed.dac")

# Load compressed file
compressed = dac.DACFile.load("compressed.dac")

# Safe streaming decode (very low GPU RAM)
y = model.decompress(compressed)

# Save output
y.write("output.wav")



