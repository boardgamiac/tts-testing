!pip install -q melotts soundfile torch

import soundfile as sf
import time
import numpy as np
from melotts import TTS

def load_model():
    tts = TTS("melo-tts")
    return tts

def prepare_text(text):
    return text.strip()

def run_inference(tts, text):
    audio = tts.speak(text)
    return audio

def save_audio(audio, sr, output_file):
    sf.write(output_file, audio, sr)
    return output_file

def benchmark_melo(text_prompt, output_file="melo_output.mp3"):
    times = {}
    start_total = time.time()

    start = time.time()
    tts = load_model()
    times["load_model"] = time.time() - start

    start = time.time()
    text = prepare_text(text_prompt)
    times["prepare_text"] = time.time() - start

    start = time.time()
    audio = run_inference(tts, text)
    times["inference"] = time.time() - start

    start = time.time()
    save_audio(audio, 22050, output_file)
    times["postprocess"] = time.time() - start

    times["total"] = time.time() - start_total
    return times, output_file

TEST_TEXT = "This is a benchmark test using MeloTTS in a detailed modular pipeline."
times, out_file = benchmark_melo(TEST_TEXT)

print("✅ MeloTTS completed.")
for k, v in times.items():
    print(f"{k:15s}: {v:.3f} sec")
print(f"Saved output to: {out_file}")
