!pip install -q transformers soundfile torch

import torch
import soundfile as sf
import numpy as np
import time
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def load_model(model_id="facebook/f5-tts", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    return processor, model, device

def load_reference_audio(path):
    speech, sr = sf.read(path)
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)  # convert stereo to mono if needed
    return speech, sr

def prepare_inputs(processor, text, speech, sr, device):
    inputs = processor(
        text=text,
        speech=speech,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(device)
    return inputs

def run_inference(model, inputs):
    with torch.no_grad():
        generated = model.generate(**inputs)
    return generated

def decode_and_save(processor, generated, sr, output_file):
    audio = processor.batch_decode(generated, output_sampling_rate=sr)[0]
    sf.write(output_file, audio, sr)
    return output_file

def benchmark_f5(reference_file, text_prompt, output_file="f5_output.mp3"):
    times = {}
    start_total = time.time()

    start = time.time()
    processor, model, device = load_model()
    times["load_model"] = time.time() - start

    start = time.time()
    speech, sr = load_reference_audio(reference_file)
    times["load_audio"] = time.time() - start

    start = time.time()
    inputs = prepare_inputs(processor, text_prompt, speech, sr, device)
    times["prepare_inputs"] = time.time() - start

    start = time.time()
    generated = run_inference(model, inputs)
    times["inference"] = time.time() - start

    start = time.time()
    decode_and_save(processor, generated, sr, output_file)
    times["postprocess"] = time.time() - start

    times["total"] = time.time() - start_total
    return times, output_file

REFERENCE_AUDIO = "reference.mp3"
TEST_TEXT = "This is a benchmark test using F5 TTS with my cloned voice in a longer and more detailed script."

times, out_file = benchmark_f5(REFERENCE_AUDIO, TEST_TEXT)

print("✅ F5 TTS completed.")
for k, v in times.items():
    print(f"{k:15s}: {v:.3f} sec")
print(f"Saved output to: {out_file}")
