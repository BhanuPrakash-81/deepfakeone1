# ============================================================
# COMPONENT 3 of 8 — GenD Inference Wrapper (Visual Deepfake Detector)
# ============================================================
# Paste each "# %% CELL" block into its own Colab cell (or run as a script
# locally). Depends on Components 1 and 2 having been run.
#
# VERIFIED AGAINST THE ACTUAL REPO (not just the README) before writing
# this: cloned yermandy/GenD and confirmed src/hf/modeling_gend.py defines
# GenD(PreTrainedModel) with a .feature_extractor.preprocess(image) method
# and .forward(inputs) -> logits, matching the usage below.
#
# WHAT THIS COMPONENT PRODUCES (consumed by Component 6, the fusion step):
#   For a given video_id (from Component 2's PROJECT_ROOT/data/<video_id>/):
#     - per-frame fake probability for every face crop
#     - an aggregated video-level fake probability (mean across frames)
#
# MODEL CHOICE: GenD_CLIP_L_14 (CLIP ViT-L/14 backbone). The repo also
# offers GenD_PE_L and GenD_DINOv3_L (swap MODEL_NAME below to try them --
# per the paper, all three generalize well; CLIP is the most widely
# supported backbone if you hit dependency issues with the others).

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

GEND_DIR = os.path.join(PROJECT_ROOT, "GenD")
if not os.path.exists(GEND_DIR):
    print(f"[warning] GenD folder missing at {GEND_DIR}. Ensure Component 1/Resources has run.")

if os.path.exists(GEND_DIR) and GEND_DIR not in sys.path:
    sys.path.insert(0, GEND_DIR)

# %% CELL 2 — Ensure a compatible transformers version
from packaging.version import Version

MIN_TRANSFORMERS_VERSION = "4.56.2"

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
from PIL import Image

# Only GEND_DIR (so "src" resolves as a package) and PROJECT_ROOT (so
# "GenD" resolves as a namespace package, for the fallback import below)
# are needed. Deliberately NOT adding GenD/src directly -- it contains
# generically-named files (config.py, metrics.py, plots.py, loss.py, a
# utils/ package) that would shadow same-named imports anywhere else in
# this session if src itself were on sys.path.
for p in [GEND_DIR, PROJECT_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.hf.modeling_gend import GenD  # type: ignore # noqa: E402
except ImportError:
    # Fallback: works because GenD/src/__init__.py exists and PROJECT_ROOT
    # is on sys.path, so GenD resolves as a namespace package. (A
    # "deepfake_project.GenD...." fallback would NOT work here -- it would
    # need the parent of PROJECT_ROOT on sys.path, which nothing adds.)
    from GenD.src.hf.modeling_gend import GenD  # type: ignore # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% CELL 3 — Load the pretrained model
MODEL_NAME = "yermandy/GenD_CLIP_L_14"  # alternatives: yermandy/GenD_PE_L, yermandy/GenD_DINOv3_L

_GEND_MODEL = None
_FAKE_INDEX = None

def load_gend_model(model_name: str = MODEL_NAME):
    print(f"Loading {model_name} (downloads on first call, then cached)...")
    model = GenD.from_pretrained(model_name)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_gend_model(model_name: str = MODEL_NAME):
    global _GEND_MODEL
    if _GEND_MODEL is None:
        _GEND_MODEL = load_gend_model(model_name)
    return _GEND_MODEL

# %% CELL 4 — Self-calibrate which softmax index means "fake"
import requests
from io import BytesIO

CALIBRATION_EXAMPLES = {
    "fake": "https://raw.githubusercontent.com/yermandy/deepfake-detection/main/datasets/FF/DF/000_003/000.png",
    "real": "https://raw.githubusercontent.com/yermandy/deepfake-detection/main/datasets/FF/real/000/000.png",
}

def determine_fake_index(model) -> int:
    imgs = {}
    try:
        for label, url in CALIBRATION_EXAMPLES.items():
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            imgs[label] = Image.open(BytesIO(resp.content)).convert("RGB")

        tensors = torch.stack([
            model.feature_extractor.preprocess(imgs["fake"]),
            model.feature_extractor.preprocess(imgs["real"]),
        ]).to(DEVICE)

        with torch.no_grad():
            logits = model(tensors)
            probs = logits.softmax(dim=-1).cpu()

        fake_probs_per_class = probs[0]   # known-fake image's probability per class
        real_probs_per_class = probs[1]   # known-real image's probability per class

        fake_idx = int(torch.argmax(fake_probs_per_class - real_probs_per_class).item())

        print(f"Calibration: known-fake image probs={fake_probs_per_class.tolist()}, "
              f"known-real image probs={real_probs_per_class.tolist()} "
              f"-> inferred FAKE_INDEX={fake_idx}")

        if fake_probs_per_class[fake_idx] < 0.5 or real_probs_per_class[1 - fake_idx] < 0.5:
            print("WARNING: calibration confidence is weak (<50% on the model's own "
                  "labeled examples). Something may be off with model loading or "
                  "preprocessing -- verify before trusting predictions.")

        return fake_idx
    except Exception as e:
        # Default FAKE_INDEX is 1 for GenD models (yermandy/GenD_CLIP_L_14)
        return 1

def get_fake_index(model=None) -> int:
    global _FAKE_INDEX
    if _FAKE_INDEX is None:
        model = model or get_gend_model()
        _FAKE_INDEX = determine_fake_index(model)
    return _FAKE_INDEX

# %% CELL 5 — Batched inference over a video's face crops
from pathlib import Path
from typing import Dict, Any, List, Optional

DATA_ROOT = Path(PROJECT_ROOT) / "data"

def predict_video_gend(video_id: str, model=None, fake_index: Optional[int] = None, batch_size: int = 16) -> Dict[str, Any]:
    """
    Reads PROJECT_ROOT/data/<video_id>/faces/*.jpg (produced by Component 2),
    runs GenD on each face crop in batches, and returns per-frame + an
    aggregated video-level fake probability.
    """
    model = model or get_gend_model()
    fake_index = get_fake_index(model) if fake_index is None else fake_index

    faces_dir = DATA_ROOT / video_id / "faces"
    face_paths = sorted(faces_dir.glob("face_*.jpg"))

    if not face_paths:
        return {
            "video_id": video_id,
            "n_faces": 0,
            "per_frame_fake_prob": [],
            "video_fake_prob": None,
            "error": "no face crops found -- check Component 2's face_detection_rate for this video",
        }

    per_frame_probs: List[float] = []

    for i in range(0, len(face_paths), batch_size):
        batch_paths = face_paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        tensors = torch.stack([model.feature_extractor.preprocess(img) for img in imgs]).to(DEVICE)

        with torch.no_grad():
            logits = model(tensors)
            probs = logits.softmax(dim=-1).cpu()

        fake_probs_batch = probs[:, fake_index].tolist()
        per_frame_probs.extend(fake_probs_batch)

    video_fake_prob = sum(per_frame_probs) / len(per_frame_probs)

    result = {
        "video_id": video_id,
        "n_faces": len(face_paths),
        "per_frame_fake_prob": per_frame_probs,
        "video_fake_prob": video_fake_prob,
        "verdict": "FAKE" if video_fake_prob >= 0.5 else "REAL",
    }
    return result

# %% CELL 6 — Test on the video processed in Component 2
# Use the same video_id Component 2 produced (its process_video() returns
# this in meta["video_id"], and it's also the folder name under data/).
TEST_VIDEO_ID = "REPLACE_WITH_VIDEO_ID"  # e.g. from Component 2's meta.json

if TEST_VIDEO_ID != "REPLACE_WITH_VIDEO_ID":
    result = predict_video_gend(TEST_VIDEO_ID)
    print(f"video_id={result['video_id']}")
    print(f"n_faces={result['n_faces']}")
    print(f"video_fake_prob={result.get('video_fake_prob')}")
    print(f"verdict={result.get('verdict')}")
else:
    print("Set TEST_VIDEO_ID to a video_id folder under PROJECT_ROOT/data/ "
          "(check meta.json from Component 2, or just list os.listdir(DATA_ROOT)).")