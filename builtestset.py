# ============================================================
# Dataset helper — build TEST_SET/TEST_DATASET from FF++, Celeb-DF, DFD
# ============================================================
# Run this in the same session, after downloading/mounting a SAMPLE of
# these datasets somewhere accessible (Drive, local disk, etc.) -- this
# does NOT download anything itself. Point the *_DIR variables below at
# wherever your videos actually are.
#
# TYPICAL FOLDER LAYOUTS (adjust patterns below if yours differs --
# dataset mirrors sometimes reorganize slightly):
#
#   FaceForensics++ (c23):
#     <FFPP_DIR>/original_sequences/youtube/c23/videos/*.mp4        <- real
#     <FFPP_DIR>/manipulated_sequences/Deepfakes/c23/videos/*.mp4   <- fake
#     <FFPP_DIR>/manipulated_sequences/Face2Face/c23/videos/*.mp4   <- fake
#     <FFPP_DIR>/manipulated_sequences/FaceSwap/c23/videos/*.mp4    <- fake
#     <FFPP_DIR>/manipulated_sequences/NeuralTextures/c23/videos/*.mp4 <- fake
#
#   Celeb-DF v2:
#     <CELEBDF_DIR>/Celeb-real/*.mp4       <- real
#     <CELEBDF_DIR>/YouTube-real/*.mp4     <- real
#     <CELEBDF_DIR>/Celeb-synthesis/*.mp4  <- fake
#
#   DFD (distributed as part of the FF++ pipeline):
#     <DFD_DIR>/original_sequences/actors/c23/videos/*.mp4          <- real
#     <DFD_DIR>/manipulated_sequences/DeepFakeDetection/c23/videos/*.mp4 <- fake

import glob
import os
import random
from typing import List, Dict, Any, Optional

random.seed(42)  # reproducible sampling across runs

# Automatically detect dataset path on Kaggle, Google Colab, or local machine
def _detect_dataset_dir() -> Optional[str]:
    env_path = os.environ.get("DATASET_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    
    candidates = [
        "C:/Users/chimm/Downloads/archive (2)",
        "/kaggle/input/archive-2",
        "/kaggle/input/archive (2)",
        "/kaggle/input/deepfake-dataset",
        "/content/dataset",
        "/content/archive (2)",
        "./dataset",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

CUSTOM_DIR = _detect_dataset_dir()

FFPP_DIR = os.environ.get("FFPP_DIR", None)      # e.g. "/content/drive/MyDrive/datasets/FaceForensics++"
CELEBDF_DIR = (
    os.environ.get("CELEBDF_DIR") or
    os.environ.get("CELEBDF_V3_DIR") or
    os.environ.get("CELEBDF_PP_DIR")
)

# Candidate Celeb-DF directories if CELEBDF_DIR is not explicitly set via environment
if not CELEBDF_DIR:
    for c in [
        "/kaggle/input/celeb-df-v3",
        "/kaggle/input/celeb-df-pp",
        "/kaggle/input/celeb-df-v2",
        "/kaggle/input/celeb-df",
        "/content/celeb-df-v3",
        "/content/celeb-df-pp",
        "/content/celeb-df-v2",
        "/content/celeb-df",
        "./celeb-df",
    ]:
        if os.path.exists(c):
            CELEBDF_DIR = c
            break
def _detect_dfd_dir() -> Optional[str]:
    env_path = os.environ.get("DFD_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    
    user_home = os.path.expanduser("~")
    candidates = [
        os.path.join(user_home, ".cache/kagglehub/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/1"),
        os.path.join(user_home, "Downloads/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"),
        "C:/Users/chimm/.cache/kagglehub/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/1",
        "/kaggle/input/deep-fake-detection-dfd-entire-original-dataset",
        "/content/DFD",
        "/content/deep-fake-detection-dfd-entire-original-dataset",
        "./DFD",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Try kagglehub auto-resolution if available
    try:
        import kagglehub
        path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        if path and os.path.exists(path):
            return path
    except Exception:
        pass

    return None

DFD_DIR = _detect_dfd_dir()        # e.g. kagglehub dataset location or environment variable

N_REAL_PER_DATASET = 25   # stratified sample size -- keep small for free-tier storage/time
N_FAKE_PER_DATASET = 25

def _sample(paths: List[str], n: int) -> List[str]:
    if len(paths) <= n:
        return paths
    return random.sample(paths, n)

def build_test_set() -> List[Dict[str, Any]]:
    test_set = []

    if FFPP_DIR and os.path.exists(FFPP_DIR):
        real_patterns = [
            f"{FFPP_DIR}/original_sequences/youtube/c23/videos/*.mp4",
            f"{FFPP_DIR}/**/original*/**/*.mp4",
            f"{FFPP_DIR}/**/youtube/**/*.mp4",
            f"{FFPP_DIR}/real/**/*.mp4",
        ]
        real = []
        for p in real_patterns:
            real.extend(glob.glob(p, recursive=True))
        real = list(set(real))

        fake_patterns = [
            f"{FFPP_DIR}/manipulated_sequences/**/c23/videos/*.mp4",
            f"{FFPP_DIR}/**/manipulated*/**/*.mp4",
            f"{FFPP_DIR}/**/Deepfakes/**/*.mp4",
            f"{FFPP_DIR}/**/Face2Face/**/*.mp4",
            f"{FFPP_DIR}/**/FaceSwap/**/*.mp4",
            f"{FFPP_DIR}/**/NeuralTextures/**/*.mp4",
            f"{FFPP_DIR}/fake/**/*.mp4",
        ]
        fake = []
        for p in fake_patterns:
            fake.extend(glob.glob(p, recursive=True))
        fake = [f for f in list(set(fake)) if f not in real and "original" not in f.lower()]

        real_s, fake_s = _sample(real, N_REAL_PER_DATASET), _sample(fake, N_FAKE_PER_DATASET)
        print(f"FF++: found {len(real)} real, {len(fake)} fake -> sampled {len(real_s)}/{len(fake_s)}")
        test_set += [{"source": p, "label": 0, "dataset": "FF++"} for p in real_s]
        test_set += [{"source": p, "label": 1, "dataset": "FF++"} for p in fake_s]

    if CELEBDF_DIR and os.path.exists(CELEBDF_DIR):
        real = list(set(
            glob.glob(f"{CELEBDF_DIR}/Celeb-real/*.mp4") +
            glob.glob(f"{CELEBDF_DIR}/YouTube-real/*.mp4") +
            glob.glob(f"{CELEBDF_DIR}/real/*.mp4")
        ))
        
        # Scandir Celeb-synthesis recursively to capture all Celeb-DF v1 / v2 / v3 / ++ subcategories
        fake = list(set(
            glob.glob(f"{CELEBDF_DIR}/Celeb-synthesis/**/*.mp4", recursive=True) +
            glob.glob(f"{CELEBDF_DIR}/Celeb-synthesis/*.mp4") +
            glob.glob(f"{CELEBDF_DIR}/fake/*.mp4")
        ))
        
        real_s, fake_s = _sample(real, N_REAL_PER_DATASET), _sample(fake, N_FAKE_PER_DATASET)
        print(f"Celeb-DF (v1/v2/v3/++): found {len(real)} real, {len(fake)} fake -> sampled {len(real_s)}/{len(fake_s)}")
        test_set += [{"source": p, "label": 0, "dataset": "Celeb-DF"} for p in real_s]
        test_set += [{"source": p, "label": 1, "dataset": "Celeb-DF"} for p in fake_s]

    if DFD_DIR and os.path.exists(DFD_DIR):
        all_vids = list(set(
            glob.glob(f"{DFD_DIR}/**/*.mp4", recursive=True) +
            glob.glob(f"{DFD_DIR}/**/*.mov", recursive=True) +
            glob.glob(f"{DFD_DIR}/**/*.MOV", recursive=True) +
            glob.glob(f"{DFD_DIR}/**/*.MP4", recursive=True)
        ))
        
        real, fake = [], []
        for v in all_vids:
            v_lower = v.lower().replace("\\", "/")
            if any(k in v_lower for k in ["actor", "original", "real", "youtube"]):
                real.append(v)
            elif any(k in v_lower for k in ["manipulated", "deepfakedetection", "deepfake", "fake", "synthesis"]):
                fake.append(v)
            else:
                fake.append(v)

        real_s, fake_s = _sample(real, N_REAL_PER_DATASET), _sample(fake, N_FAKE_PER_DATASET)
        print(f"DFD: found {len(real)} real, {len(fake)} fake -> sampled {len(real_s)}/{len(fake_s)}")
        test_set += [{"source": p, "label": 0, "dataset": "DFD"} for p in real_s]
        test_set += [{"source": p, "label": 1, "dataset": "DFD"} for p in fake_s]

    if CUSTOM_DIR and os.path.exists(CUSTOM_DIR):
        all_custom = list(set(
            glob.glob(f"{CUSTOM_DIR}/**/*.mp4", recursive=True) +
            glob.glob(f"{CUSTOM_DIR}/**/*.mov", recursive=True) +
            glob.glob(f"{CUSTOM_DIR}/**/*.MOV", recursive=True) +
            glob.glob(f"{CUSTOM_DIR}/**/*.MP4", recursive=True) +
            glob.glob(f"{CUSTOM_DIR}/*.mp4") +
            glob.glob(f"{CUSTOM_DIR}/*.mov")
        ))
        real, fake = [], []
        for v in all_custom:
            v_lower = v.lower().replace("\\", "/")
            if any(k in v_lower for k in ["/real/", "_real", "real_", "original", "actor", "youtube"]):
                real.append(v)
            elif any(k in v_lower for k in ["/fake/", "_fake", "fake_", "manipulated", "deepfake", "synthesis"]):
                fake.append(v)
            else:
                fake.append(v)

        real_s, fake_s = _sample(real, N_REAL_PER_DATASET), _sample(fake, N_FAKE_PER_DATASET)
        print(f"Custom ({os.path.basename(CUSTOM_DIR)}): found {len(real)} real, {len(fake)} fake -> sampled {len(real_s)}/{len(fake_s)}")
        test_set += [{"source": p, "label": 0, "dataset": "Custom"} for p in real_s]
        test_set += [{"source": p, "label": 1, "dataset": "Custom"} for p in fake_s]

    if not test_set:
        print("WARNING: no dataset directories set (FFPP_DIR/CELEBDF_DIR/DFD_DIR/CUSTOM_DIR are all None), "
              "or glob patterns didn't match anything on disk -- check your folder layout "
              "against the header comment above.")

    random.shuffle(test_set)
    return test_set

# %% Build dataset list for export
TEST_SET_FROM_DATASETS = build_test_set()

if __name__ == "__main__":
    print(f"\nTotal: {len(TEST_SET_FROM_DATASETS)} videos "
          f"({sum(1 for x in TEST_SET_FROM_DATASETS if x['label']==0)} real, "
          f"{sum(1 for x in TEST_SET_FROM_DATASETS if x['label']==1)} fake)")

# Then in evaluation.py's Cell 4, either replace TEST_DATASET with this,
# or run directly:
#   evaluate_dataset(TEST_SET_FROM_DATASETS)