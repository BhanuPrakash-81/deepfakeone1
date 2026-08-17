# ============================================================
# COMPONENT 4 of 8 — Audio-Spoof Detection Branch
# ============================================================
# Paste each "# %% CELL" block into its own Colab cell (or run as a script
# locally). Depends on Components 1 and 2 having been run.
#
# PLAN CHANGE FROM COMPONENT 1: the original plan was AASIST (clovaai/aasist)
# + a separately-downloaded wav2vec2-xls-r-300m checkpoint. Investigating
# further: clovaai/aasist is RAW-WAVEFORM ONLY (no Wav2Vec2 front end), and
# the actual W2V+AASIST hybrid lives in a different repo
# (TakHemlata/SSL_Anti-spoofing) that requires fairseq + torch==1.8.1 --
# both a poor fit for Colab (fairseq is fragile to install/compile, and the
# torch pin conflicts with Colab's preinstalled CUDA build).
#
# Instead, this component uses a fully Hugging Face-native alternative:
# Gustking/wav2vec2-large-xlsr-deepfake-audio-classification -- a
# fine-tuned Wav2Vec2-XLSR-300M model, loadable via plain
# AutoModelForAudioClassification, no fairseq required. It's built on the
# SAME base checkpoint (facebook/wav2vec2-xls-r-300m) already downloaded in
# Component 1. Published results: 92.86% accuracy, 4.01% EER on ASVspoof2019.
#
# CONSEQUENCE: the aasist/ clone and the standalone wav2vec2-xls-r-300m
# snapshot from Component 1 are no longer needed by this pipeline (this
# model downloads its own complete weights). Component 1 has been updated
# to drop those two now-unnecessary downloads -- re-run it if you already
# did the old version and want to reclaim the disk space.
#
# WHAT THIS COMPONENT PRODUCES (consumed by Component 6, fusion):
#   For a given video_id: a video-level fake probability from the audio
#   track, or a clear "no_audio" result if Component 2 found no audio track.

# %% CELL 1 — Re-establish environment
import os
import sys

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

_raw_root = globals().get("PROJECT_ROOT", None)
if _raw_root and os.path.exists(str(_raw_root)):
    PROJECT_ROOT: str = str(_raw_root)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    PROJECT_ROOT = os.path.join(BASE_DIR, "deepfake_project")

# %% CELL 2 — Ensure a compatible transformers version
from packaging.version import Version

MIN_TRANSFORMERS_VERSION = "4.40.0"

def check_transformers_version(min_version: str = MIN_TRANSFORMERS_VERSION):
    try:
        import transformers
        current = Version(transformers.__version__)
        if current < Version(min_version):
            print(f"[warning] Installed transformers version ({current}) is below recommended {min_version}.")
    except ImportError:
        print("[warning] transformers package is not installed.")

check_transformers_version()

import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% CELL 3 — Load model + feature extractor (lazy singleton, same pattern as Component 3)
AUDIO_MODEL_NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
TARGET_SR = 16000  # what this model was trained on; Component 2 already extracts at 16kHz mono

_AUDIO_MODEL = None
_AUDIO_FEATURE_EXTRACTOR = None

def load_audio_model(model_name: str = AUDIO_MODEL_NAME):
    print(f"Loading {model_name} (downloads on first call, then cached)...")
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    return model, fe

def get_audio_model(model_name: str = AUDIO_MODEL_NAME):
    global _AUDIO_MODEL, _AUDIO_FEATURE_EXTRACTOR
    if _AUDIO_MODEL is None:
        _AUDIO_MODEL, _AUDIO_FEATURE_EXTRACTOR = load_audio_model(model_name)
    return _AUDIO_MODEL, _AUDIO_FEATURE_EXTRACTOR

# %% CELL 4 — Determine which label means "fake" from the model's own config
# Rather than hardcoding an index (same reasoning as Component 3's
# calibration step -- getting this backwards silently inverts every
# prediction), read model.config.id2label at runtime and keyword-match it.
FAKE_KEYWORDS = ("fake", "spoof", "synthetic", "ai", "generated")
REAL_KEYWORDS = ("real", "bonafide", "genuine", "human", "authentic")

_AUDIO_FAKE_INDEX = None

def determine_audio_fake_index(model) -> int:
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f"Model label map: {id2label}")

    fake_candidates = [i for i, label in id2label.items() if any(k in label for k in FAKE_KEYWORDS)]
    real_candidates = [i for i, label in id2label.items() if any(k in label for k in REAL_KEYWORDS)]

    if len(fake_candidates) == 1 and len(real_candidates) == 1 and fake_candidates[0] != real_candidates[0]:
        fake_idx = fake_candidates[0]
        print(f"Resolved FAKE_INDEX={fake_idx} from label text (unambiguous keyword match).")
    else:
        # Ambiguous -- don't guess silently. Default to index 1 (the more
        # common convention) but make this loud so it gets checked.
        fake_idx = 1
        print("WARNING: could not unambiguously determine the fake label from "
              f"id2label={id2label}. Defaulting to index 1 -- VERIFY this is "
              "correct by checking the model card before trusting predictions.")

    return fake_idx

def get_audio_fake_index(model=None) -> int:
    global _AUDIO_FAKE_INDEX
    if _AUDIO_FAKE_INDEX is None:
        model = model or get_audio_model()[0]
        _AUDIO_FAKE_INDEX = determine_audio_fake_index(model)
    return _AUDIO_FAKE_INDEX

# %% CELL 5 — Inference over a video's extracted audio
from pathlib import Path
from typing import Dict, Any, Optional

DATA_ROOT = Path(PROJECT_ROOT) / "data"
MAX_CHUNK_SECONDS = 20  # process in <=20s chunks and average -- keeps memory bounded
                         # regardless of how long an input clip is, even though
                         # Component 2 already caps ingested video length

def predict_video_audio(video_id: str, fake_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Reads PROJECT_ROOT/data/<video_id>/audio.wav (produced by Component 2)
    and returns a video-level fake probability from the audio-spoof branch.
    Returns a clear "no_audio" result (not a crash) if Component 2 found no
    audio track for this video -- Component 6 (fusion) must check this
    before trying to use the audio branch's score.
    """
    video_dir = DATA_ROOT / video_id
    meta_path = video_dir / "meta.json"
    audio_path = video_dir / "audio.wav"

    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("has_audio", False):
            return {"video_id": video_id, "has_audio": False, "audio_fake_prob": None,
                     "note": "Component 2 found no audio track for this video."}

    if not audio_path.exists():
        return {"video_id": video_id, "has_audio": False, "audio_fake_prob": None,
                 "note": f"expected audio file not found at {audio_path}"}

    model, feature_extractor = get_audio_model()
    fake_index = get_audio_fake_index(model) if fake_index is None else fake_index

    waveform, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

    if len(waveform) == 0:
        return {"video_id": video_id, "has_audio": False, "audio_fake_prob": None,
                 "note": "audio file exists but decoded to zero samples (corrupt/empty)."}

    chunk_len = MAX_CHUNK_SECONDS * TARGET_SR
    chunk_fake_probs = []

    for start in range(0, len(waveform), chunk_len):
        chunk = waveform[start:start + chunk_len]
        if len(chunk) < TARGET_SR * 0.5:  # skip trailing slivers under 0.5s
            continue

        inputs = feature_extractor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = logits.softmax(dim=-1).cpu()

        chunk_fake_probs.append(float(probs[0, fake_index]))

    if not chunk_fake_probs:
        return {"video_id": video_id, "has_audio": False, "audio_fake_prob": None,
                 "note": "audio too short to score after chunking."}

    video_fake_prob = float(np.mean(chunk_fake_probs))

    return {
        "video_id": video_id,
        "has_audio": True,
        "n_chunks": len(chunk_fake_probs),
        "chunk_fake_probs": chunk_fake_probs,
        "audio_fake_prob": video_fake_prob,
        "verdict": "FAKE" if video_fake_prob >= 0.5 else "REAL",
    }

# %% CELL 6 — Test on the video processed in Components 2 and 3
TEST_VIDEO_ID = "REPLACE_WITH_VIDEO_ID"

if TEST_VIDEO_ID != "REPLACE_WITH_VIDEO_ID":
    result = predict_video_audio(TEST_VIDEO_ID)
    print(result)
    if result.get("has_audio") is False:
        print("This video has no usable audio track -- Component 6 (fusion) "
              "should fall back to video-only branches for it.")
else:
    print("Set TEST_VIDEO_ID to a video_id folder under PROJECT_ROOT/data/.")
