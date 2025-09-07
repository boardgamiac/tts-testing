!pip -q install "transformers>=4.36.0" accelerate soundfile bitsandbytes huggingface_hub

import os, time, torch, soundfile as sf
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "nari-labs/Dia-1.6B-0626"
CACHE_DIR = "/content/hf_cache"
USE_INT8 = False
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
torch.set_float32_matmul_precision("high")

# Pre-cache weights (avoids downloads in timing)
_ = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR, resume_download=True)

times = {}
t0 = time.time()

# Load processor + model
t = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
if USE_INT8:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        device_map="auto",
        load_in_8bit=True
    )
else:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=DTYPE
    ).to(DEVICE)
model = torch.compile(model)  # optimize graph
times["load_model"] = time.time() - t

# Prepare input
text = ["This is a Dia sixteen B speed test."]
t = time.time()
inputs = processor(text=text, padding=True, return_tensors="pt").to(DEVICE)
times["prepare_inputs"] = time.time() - t

# Warm-up (excluded from timing)
with torch.inference_mode():
    _ = model.generate(**inputs, max_new_tokens=16)

# Main inference
t = time.time()
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=96)  # ~1s audio
times["inference"] = time.time() - t

# Postprocess + save
t = time.time()
decoded = processor.batch_decode(outputs, output_type="np")
audio = decoded[0]["audio"]
sr = decoded[0]["sampling_rate"]
out_path = "/content/dia16b_output.wav"
sf.write(out_path, audio, sr)
times["postprocess"] = time.time() - t

times["total"] = time.time() - t0

print("✅ Dia-1.6B completed.")
for k, v in times.items():
    print(f"{k:15s}: {v:.3f} sec")
print(f"Saved output to: {out_path}")
