!pip -q install "transformers>=4.36.0" accelerate soundfile huggingface_hub bitsandbytes

import os, time, json, torch, soundfile as sf, numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "facebook/f5-tts"
CACHE_DIR = "/content/hf_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_8BIT = False
DTYPE = torch.float16
NUM_INFERENCE_STEPS = 8
WARMUP_TOKENS = 16
MAX_TRIES = 3
OUT_BASENAME = "/content/f5tts_fast"
REFERENCE_PATH = "/content/reference.wav"
PROMPT_TEXT = "Hello, this is an optimized F5-TTS benchmark test."

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
torch.set_float32_matmul_precision("high")

print("Prefetching model files into local cache (may take a minute first run)...")
try:
    _ = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR, resume_download=True)
    print("Prefetch complete.")
except Exception as e:
    print("Warning: snapshot_download failed:", str(e))

times = {}
t_total_start = time.time()

t = time.time()
processor = None
model = None
load_err = None
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
except Exception:
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=False)
    except Exception as e:
        load_err = f"Processor load failed: {e}"
try:
    if USE_8BIT:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            device_map="auto",
            load_in_8bit=True
        )
    else:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            torch_dtype=DTYPE
        ).to(DEVICE)
    try:
        model = torch.compile(model)
    except Exception:
        pass
except Exception as e:
    load_err = (load_err + " | Model load failed: " + str(e)) if load_err else f"Model load failed: {e}"
times["load_model"] = time.time() - t

if load_err:
    raise RuntimeError(load_err)

t = time.time()
speech = None
sr = None
if os.path.exists(REFERENCE_PATH):
    try:
        speech, sr = sf.read(REFERENCE_PATH)
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
        speech = speech.astype(np.float32)
    except Exception as e:
        print("Warning: failed to read reference audio:", e)
times["load_audio"] = time.time() - t

t = time.time()
inputs = None
try:
    if speech is not None and processor is not None:
        try:
            inputs = processor(text=PROMPT_TEXT, speech=speech, sampling_rate=sr, return_tensors="pt")
        except TypeError:
            inputs = processor([PROMPT_TEXT], sampling_rate=sr, return_tensors="pt")
    else:
        inputs = processor(text=[PROMPT_TEXT], return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(DEVICE)
except Exception as e:
    raise RuntimeError("prepare_inputs failed: " + str(e))
times["prepare_inputs"] = time.time() - t

try:
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=min(32, WARMUP_TOKENS))
except Exception:
    pass

inference_time = None
post_time = None
out_path = None
used_steps = NUM_INFERENCE_STEPS

for attempt in range(1, MAX_TRIES + 1):
    t = time.time()
    try:
        try:
            gen = model.generate(**inputs, num_inference_steps=used_steps, max_new_tokens=256)
        except TypeError:
            try:
                gen = model.generate(**inputs, num_steps=used_steps, max_new_tokens=256)
            except TypeError:
                gen = model.generate(**inputs, max_new_tokens=256)
        inference_time = time.time() - t
    except Exception as e:
        raise RuntimeError("Generation failed: " + str(e))

    t2 = time.time()
    decoded = None
    try:
        try:
            decoded = processor.batch_decode(gen, output_type="np")
            if isinstance(decoded, list) and isinstance(decoded[0], dict) and "audio" in decoded[0]:
                audio = decoded[0]["audio"]
                sr_out = decoded[0].get("sampling_rate", sr or 24000)
            elif isinstance(decoded, list) and isinstance(decoded[0], np.ndarray):
                audio = decoded[0]
                sr_out = sr or 24000
            else:
                audio = None
        except Exception:
            decoded = processor.batch_decode(gen, output_sampling_rate=sr if sr else 24000)
            if isinstance(decoded, list) and isinstance(decoded[0], dict) and "audio" in decoded[0]:
                audio = decoded[0]["audio"]
                sr_out = decoded[0].get("sampling_rate", sr or 24000)
            elif isinstance(decoded, list) and isinstance(decoded[0], np.ndarray):
                audio = decoded[0]
                sr_out = sr or 24000
            else:
                audio = None
    except Exception as e:
        audio = None

    if audio is None:
        try:
            if isinstance(gen, torch.Tensor):
                arr = gen.detach().cpu().numpy()
                if arr.ndim > 1:
                    arr = arr[0]
                audio = arr
                sr_out = sr or 24000
            else:
                with open(OUT_BASENAME + f"_raw_gen_attempt{attempt}.txt", "w") as f:
                    f.write(repr(gen))
                audio = None
                sr_out = None
        except Exception:
            audio = None
            sr_out = None

    if audio is not None:
        out_path = f"{OUT_BASENAME}_attempt{attempt}.wav"
        try:
            arr = np.asarray(audio, dtype=np.float32)
            sf.write(out_path, arr, samplerate=sr_out)
        except Exception as e:
            try:
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                sf.write(out_path, arr, samplerate=sr_out)
            except Exception as e2:
                out_path = None

    post_time = time.time() - t2
    times["inference_try_" + str(attempt)] = inference_time
    times["postprocess_try_" + str(attempt)] = post_time
    times["used_steps_try_" + str(attempt)] = used_steps

    total_elapsed = time.time() - t_total_start
    if out_path is not None and total_elapsed <= 5.0:
        break
    if attempt < MAX_TRIES:
        used_steps = max(4, int(used_steps * 0.6))
        print(f"Attempt {attempt} finished in {total_elapsed:.3f}s; retrying with {used_steps} steps...")
        continue
    else:
        break

times["total"] = time.time() - t_total_start
report = {
    "model_id": MODEL_ID,
    "device": DEVICE,
    "use_8bit": bool(USE_8BIT),
    "dtype": str(DTYPE),
    "prompt": PROMPT_TEXT,
    "reference_path": REFERENCE_PATH if os.path.exists(REFERENCE_PATH) else None,
    "output_path": out_path,
    "timings": times
}

print("=== F5-TTS benchmark report ===")
print(json.dumps(report, indent=2))
if out_path:
    print("Saved audio to:", out_path)
else:
    print("No audio saved; check logs and raw_gen files for debugging.")
