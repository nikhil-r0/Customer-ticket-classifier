from safetensors import safe_open
import torch

tensors = {}

with safe_open("/mnt/extra-maal/forge/customer-ticket-classifier/fine_tuned_tinyllama/adapter_model.safetensors", framework="pt", device="cpu") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

print(tensors.keys())
