!pip install -q kokoro-tts soundfile torch

import torch
import soundfile as sf
import time
import numpy as np
from kokoro_tts import KokoroTTS

def load_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KokoroTTS.from_pretrained("kokoro-82M").to(device)
    return model, device

def prepare_text(text):
    return text.strip()

def run_inference(model, text):
    with torch.no_grad():
        audio = model.tts(text)
    return audio

def save_audio(audio, sr, output_file):
    sf.write(output_file, audio, sr)
    return output_file

def benchmark_kokoro(text_prompt, output_file="kokoro_output.mp3"):
    times = {}
    start_total = time.time()

    start = time.time()
    model, device = load_model()
    times["load_model"] = time.time() - start

    start = time.time()
    text = prepare_text(text_prompt)
    times["prepare_text"] = time.time() - start

    start = time.time()
    audio = run_inference(model, text)
    times["inference"] = time.time() - start

    start = time.time()
    save_audio(audio, 22050, output_file)
    times["postprocess"] = time.time() - start

    times["total"] = time.time() - start_total
    return times, output_file

TEST_TEXT = "This is a benchmark test using Kokoro eighty two million model in an elaborated pipeline."
times, out_file = benchmark_kokoro(TEST_TEXT)

print(" Kokoro-82M completed.")
for k, v in times.items():
    print(f"{k:15s}: {v:.3f} sec")
print(f"Saved output to: {out_file}")

