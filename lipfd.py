# ============================================================
# COMPONENT 5 of 8 — LipFD Lip-Sync Consistency Branch
# ============================================================
# Paste each "# %% CELL" block into its own Colab cell (or run as a script
# locally). Depends on Components 1 and 2 having been run.
#
# HONEST COMPLEXITY NOTE: unlike GenD (Component 3) and the audio-spoof
# branch (Component 4), LipFD has no from_pretrained()-style interface.
# I cloned the actual repo and read preprocess.py, models/LipFD.py,
# data/datasets.py, and validate.py to reverse-engineer the exact input
# format before writing this -- guessing at a research repo's bespoke
# preprocessing would very likely produce silently-wrong predictions.
#
# WHAT THE MODEL ACTUALLY EXPECTS (confirmed from source, not the README):
#   1. Sample N_EXTRACT=10 starting points across the video, each a
#      WINDOW_LEN=5 consecutive-frame window.
#   2. Each frame resized to 500x500, concatenated horizontally -> 500x2500.
#   3. The video's mel-spectrogram (librosa, saved+reread as a PNG -- this
#      round-trip is IN THE ORIGINAL CODE and reproduced here exactly,
#      since it affects the exact pixel values the checkpoint was trained
#      on) is sliced to the matching time window and resized to 500x2500.
#   4. Spectrogram slice stacked on TOP of the frame strip -> 1000x2500x3.
#   5. From this stacked image: three crop "zoom levels" are taken per
#      frame (full frame, then two progressively tighter fixed-PIXEL crops
#      at [28:196] and [61:163] -- NOT face-detected, just fixed pixel
#      windows, which assumes a roughly centered talking-head framing).
#   6. The whole stacked image is separately resized to 1120x1120 and run
#      through a CLIP encoder for a global feature vector.
#   7. model(crops, global_feature) -> logit -> sigmoid -> P(fake).
#
# KNOWN ORIGINAL-CODE QUIRK, REPRODUCED EXACTLY (not "fixed"):
#   data/datasets.py slices frame crops as img[:, 500:, i:i+500] for
#   i in range(5) -- this indexes columns [0:500],[1:501],[2:502],...
#   instead of the seemingly-intended [0:500],[500:1000],[1000:1500],...
#   Since the released checkpoint was trained against this exact code,
#   changing it would likely make predictions WORSE, not better -- so it
#   is reproduced here byte-for-byte rather than corrected.
#
# INTEGRATION NOTE: this branch reads directly from Component 2's
# source.mp4 + audio.wav (NOT the fixed-5fps frames/ folder), because
# LipFD's window selection needs the video's true native fps/frame_count
# to align spectrogram time slices correctly -- information Component 2's
# fixed-rate frame extraction doesn't preserve.
#
# LIMITATION WORTH KNOWING: no calibration example ships with this repo
# (unlike GenD), so the fake-label polarity below is inferred from
# data/datasets.py's label_dict (0_real -> 0, 1_fake -> 1) and validate.py's
# sigmoid-output convention, NOT empirically verified against a known
# labeled example the way Components 3 and 4 are. Sanity-check this on a
# video you know the ground truth for before trusting it.

# %% CELL 1 — Re-establish environment
import os
import sys

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    PROJECT_ROOT = "/content/drive/MyDrive/deepfake_project"
else:
    PROJECT_ROOT = os.path.join(BASE_DIR, "deepfake_project")

os.makedirs(PROJECT_ROOT, exist_ok=True)
LIPFD_DIR = os.path.join(PROJECT_ROOT, "LipFD")
if not os.path.exists(LIPFD_DIR):
    print(f"[warning] LipFD folder missing at {LIPFD_DIR}. Ensure Component 1/Resources has run.")

for p in [LIPFD_DIR, PROJECT_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# %% CELL 2 — Dependencies specific to LipFD

import torch
import cv2
import numpy as np
import librosa
from librosa import feature as librosa_feature
import matplotlib
matplotlib.use("Agg")  # headless -- no display backend needed/available in Colab
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

try:
    from models import build_model  # type: ignore # noqa: E402
except ImportError:
    # Fallback: works because PROJECT_ROOT is on sys.path, so LipFD
    # resolves as a namespace package (same reasoning as gend_interface.py's
    # GenD.src.hf fallback). A "deepfake_project.LipFD.models" fallback
    # would NOT work -- nothing puts the parent of PROJECT_ROOT on
    # sys.path, so "deepfake_project" itself is never importable.
    from LipFD.models import build_model  # type: ignore # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% CELL 3 — Download the pretrained checkpoint (Google Drive -- needs gdown)
import gdown

CKPT_DRIVE_FILE_ID = "1NPAcx0QS8N9v_9qUr-51jBaL9kGDT-cp"
CKPT_PATH = os.path.join(LIPFD_DIR, "checkpoints", "ckpt.pth")

def ensure_lipfd_checkpoint():
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    if os.path.exists(CKPT_PATH):
        print(f"[skip] checkpoint already present -> {CKPT_PATH}")
        return
    print("Downloading LipFD checkpoint from Google Drive (~GB-scale, may take a few minutes)...")
    gdown.download(id=CKPT_DRIVE_FILE_ID, output=CKPT_PATH, quiet=False)

ensure_lipfd_checkpoint()

# %% CELL 4 — Load the model (lazy singleton, same pattern as Components 3 and 4)
LIPFD_ARCH = "CLIP:ViT-L/14"  # matches validate.py's default

_LIPFD_MODEL = None

def load_lipfd_model(arch: str = LIPFD_ARCH, ckpt_path: str = CKPT_PATH):
    print(f"Building LipFD ({arch}) and loading checkpoint...")
    model = build_model(arch)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()
    model.to(DEVICE)
    return model

def get_lipfd_model():
    global _LIPFD_MODEL
    if _LIPFD_MODEL is None:
        _LIPFD_MODEL = load_lipfd_model()
    return _LIPFD_MODEL

# %% CELL 5 — Faithful reproduction of preprocess.py's spectrogram + window logic
N_EXTRACT = 10   # number of windows sampled across the video (matches preprocess.py)
WINDOW_LEN = 5   # frames per window (matches preprocess.py)
FRAME_SIZE = 500

_CLIP_NORM = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

def _get_mel_spectrogram_image(audio_path: str, tmp_path: str) -> np.ndarray:
    """
    Reproduces preprocess.py's get_spectrogram() exactly, including the
    save-then-reread-as-PNG round trip -- this affects the exact pixel
    values (via matplotlib's default colormap + imsave normalization) that
    the checkpoint was trained on, so it's kept rather than replaced with
    a more "direct" spectrogram-to-array conversion.
    """
    data, sr = librosa.load(audio_path)
    mel = librosa.power_to_db(librosa_feature.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(tmp_path, mel)
    mel_img = plt.imread(tmp_path) * 255
    return mel_img.astype(np.uint8)

def _build_stacked_windows(video_path: str, audio_path: str, tmp_dir: str):
    """
    Reproduces preprocess.py's run() logic for a SINGLE video (the original
    operates over a labeled dataset folder; this adapts it to one file).
    Returns a list of 1000x2500x3 uint8 stacked images, one per window.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_mel_path = os.path.join(tmp_dir, "mel.png")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= WINDOW_LEN + 1:
        cap.release()
        return []

    frame_idx = np.linspace(0, frame_count - WINDOW_LEN - 1, N_EXTRACT, endpoint=True, dtype=np.uint8).tolist()
    frame_idx.sort()
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]

    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in frame_sequence:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)))
        current_frame += 1
    cap.release()

    mel = _get_mel_spectrogram_image(audio_path, tmp_mel_path)
    mapping = mel.shape[1] / frame_count

    stacked_images = []
    group = 0
    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        if idx != 0:
            continue
        try:
            begin = np.round(frame_sequence[i] * mapping)
            end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
            sub_mel = cv2.resize(mel[:, int(begin):int(end)], (FRAME_SIZE * WINDOW_LEN, FRAME_SIZE))
            x = np.concatenate(frame_list[i:i + WINDOW_LEN], axis=1)
            x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
            stacked_images.append(x)
            group += 1
        except ValueError:
            continue

    return stacked_images

def _stacked_image_to_model_inputs(stacked_img: np.ndarray):
    """
    Reproduces data/datasets.py's AVLip.__getitem__ exactly, including the
    img[:, 500:, i:i+500] indexing quirk noted in the header comment.
    """
    img = torch.tensor(stacked_img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
    img_norm = _CLIP_NORM(img)

    crops = [[transforms.Resize((224, 224))(img_norm[:, 500:, i:i + 500]) for i in range(5)], [], []]
    crop_idx = [(28, 196), (61, 163)]
    for i in range(len(crops[0])):
        crops[1].append(transforms.Resize((224, 224))(
            crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
        crops[2].append(transforms.Resize((224, 224))(
            crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))

    img_full = transforms.Resize((1120, 1120))(img_norm)
    return img_full, crops

# %% CELL 6 — Inference over a video (Component 2's source.mp4 + audio.wav)
from pathlib import Path
from typing import Dict, Any

DATA_ROOT = Path(PROJECT_ROOT) / "data"

def predict_video_lipsync(video_id: str) -> Dict[str, Any]:
    """
    Reads PROJECT_ROOT/data/<video_id>/source.mp4 and audio.wav (from
    Component 2), builds LipFD's windowed spectrogram+frame inputs, and
    returns a video-level lip-sync fake probability (mean sigmoid output
    across windows). Returns has_audio=False if Component 2 found no
    audio track -- this branch cannot run without audio by definition.
    """
    video_dir = DATA_ROOT / video_id
    video_path = video_dir / "source.mp4"
    audio_path = video_dir / "audio.wav"
    meta_path = video_dir / "meta.json"

    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get("has_audio", False):
            return {"video_id": video_id, "has_audio": False, "lipsync_fake_prob": None,
                     "note": "no audio track -- lip-sync branch requires audio+video together."}

    if not video_path.exists() or not audio_path.exists():
        return {"video_id": video_id, "has_audio": False, "lipsync_fake_prob": None,
                 "note": f"missing source.mp4 or audio.wav under {video_dir}"}

    tmp_dir = str(video_dir / "_lipfd_tmp")
    stacked_images = _build_stacked_windows(str(video_path), str(audio_path), tmp_dir)

    if not stacked_images:
        return {"video_id": video_id, "has_audio": True, "lipsync_fake_prob": None,
                 "note": "video too short to build any windows (needs > WINDOW_LEN+1 frames)."}

    model = get_lipfd_model()
    window_probs = []

    with torch.no_grad():
        for stacked_img in stacked_images:
            img_full, crops = _stacked_image_to_model_inputs(stacked_img)
            img_tens = img_full.unsqueeze(0).to(DEVICE)
            crops_tens = [[t.unsqueeze(0).to(DEVICE) for t in level] for level in crops]

            features = model.get_features(img_tens)
            logit = model(crops_tens, features)[0]
            prob = torch.sigmoid(logit).flatten().item()
            window_probs.append(prob)

    video_prob = float(np.mean(window_probs))

    return {
        "video_id": video_id,
        "has_audio": True,
        "n_windows": len(window_probs),
        "window_fake_probs": window_probs,
        "lipsync_fake_prob": video_prob,
        "verdict": "FAKE (lip-sync mismatch)" if video_prob >= 0.5 else "REAL (lip-sync consistent)",
    }

# %% CELL 7 — Test on the video processed in Component 2
TEST_VIDEO_ID = "REPLACE_WITH_VIDEO_ID"

if TEST_VIDEO_ID != "REPLACE_WITH_VIDEO_ID":
    result = predict_video_lipsync(TEST_VIDEO_ID)
    print(result)
    if result.get("lipsync_fake_prob") is None:
        print("No lip-sync score produced for this video -- check the 'note' field above.")
    else:
        print("REMINDER: this branch's fake/real polarity is inferred from source, not "
              "empirically calibrated like Components 3/4. Verify on a video with known "
              "ground truth before trusting it in the fusion step (Component 6).")
else:
    print("Set TEST_VIDEO_ID to a video_id folder under PROJECT_ROOT/data/.")