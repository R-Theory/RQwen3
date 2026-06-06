import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import numpy as np
import torch
import torch.nn as nn
from src import *
from dataclasses import dataclass
from typing import Union
from abc import abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import Markdown
from datasets import load_dataset
from fastapi import FastAPI

app = FastAPI()
DEVICE = get_device()
MODEL_IDs = ["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"]

# Load once at startup — weights are cached on disk at ~/.cache/huggingface/
# so this won't re-download, but it still needs to load into RAM/VRAM once
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_IDs[0], device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


@app.get("/health")
async def health():
    return {"status": "ok"}

def generate(prompt: str, max_length: int = 50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_length=max_length)

    return {"message": prompt, "response": tokenizer.decode(outputs[0], skip_special_tokens=True)}

res = generate(prompt="Why is math so important?")

print(res)