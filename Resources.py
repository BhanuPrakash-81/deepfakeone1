# ============================================================
# COMPONENT 1 of 7 — Environment Setup & Pretrained Resource Acquisition
# ============================================================
# Run top to bottom. Works in both Google Colab and Local environments.
#
# SITES USED IN THIS COMPONENT (verified reachable as of writing this):
#   GenD          -> https://github.com/yermandy/GenD
#   DeepfakeBench -> https://github.com/SCLBD/DeepfakeBench
#   GenConViT     -> https://github.com/erprogs/GenConViT  (HF weights linked in its README)
#   LipFD         -> https://github.com/AaronComo/LipFD
#
# PLAN CHANGE: AASIST + standalone wav2vec2-xls-r-300m removed from this
# list. clovaai/aasist turned out to be raw-waveform-only (no Wav2Vec2 front
# end); the real hybrid needs fairseq + torch==1.8.1, a poor fit for Colab.
# Component 4 (audio-spoof branch) uses a fully self-contained Hugging Face
# model instead (Gustking/wav2vec2-large-xlsr-deepfake-audio-classification)
# that downloads its own weights on first use -- nothing to clone here.

import os
import sys
import subprocess

# %% CELL 1 — Environment setup
try:
    import google.colab  # type: ignore
    IN_COLAB = True
    print("Running in Google Colab environment.")
except ImportError:
    IN_COLAB = False
    print("Running in local environment.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
if IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
    PROJECT_ROOT = "/content/drive/MyDrive/deepfake_project"
else:
    # Use a local folder in the project workspace
    PROJECT_ROOT = os.path.join(BASE_DIR, "deepfake_project")

os.makedirs(PROJECT_ROOT, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")

# %% CELL 2 — Install dependencies
def install_dependencies():
    pip_packages = [
        "transformers", "accelerate", "open_clip_torch",
        "yt-dlp", "mediapipe", "opencv-python-headless",
        "librosa", "soundfile", "huggingface_hub"
    ]
    if not IN_COLAB:
        pip_packages = ["torch", "torchvision"] + pip_packages

    import importlib.util
    pkg_map = {
        "open_clip_torch": "open_clip",
        "opencv-python-headless": "cv2",
        "yt-dlp": "yt_dlp"
    }
    missing = []
    for pkg in pip_packages:
        mod_name = pkg_map.get(pkg, pkg)
        if importlib.util.find_spec(mod_name) is None:
            missing.append(pkg)

    if missing:
        print(f"Installing missing pip dependencies: {missing}...")
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
        if not IN_COLAB and sys.prefix == sys.base_prefix:
            cmd.append("--user")
        subprocess.check_call(cmd)
    else:
        print("All required pip dependencies are already installed.")

    if IN_COLAB:
        print("Installing ffmpeg...")
        subprocess.check_call(["apt-get", "-qq", "install", "-y", "ffmpeg"])
    else:
        import shutil as _shutil
        if _shutil.which("ffmpeg") is None:
            try:
                import imageio_ffmpeg
                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
                ffmpeg_dir = os.path.dirname(ffmpeg_bin)
                os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")
                print(f"ffmpeg found via imageio-ffmpeg at {ffmpeg_bin}")
            except Exception:
                print("WARNING: 'ffmpeg' not found on PATH. Install it manually "
                      "(brew install ffmpeg / sudo apt install ffmpeg / "
                      "https://ffmpeg.org/download.html) before running Component 2 "
                      "-- frame/audio extraction will fail without it.")
        else:
            print("ffmpeg found on PATH.")

def setup_resources():
    install_dependencies()

    try:
        import torch
        print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[warning] torch not installed yet.")

    clone_if_missing("https://github.com/yermandy/GenD", f"{PROJECT_ROOT}/GenD")
    clone_if_missing("https://github.com/SCLBD/DeepfakeBench", f"{PROJECT_ROOT}/DeepfakeBench")
    clone_if_missing("https://github.com/erprogs/GenConViT", f"{PROJECT_ROOT}/GenConViT")
    clone_if_missing("https://github.com/AaronComo/LipFD", f"{PROJECT_ROOT}/LipFD")

    expected = [
        os.path.join(PROJECT_ROOT, "GenD"),
        os.path.join(PROJECT_ROOT, "DeepfakeBench"),
        os.path.join(PROJECT_ROOT, "GenConViT"),
        os.path.join(PROJECT_ROOT, "LipFD"),
    ]

    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    all_ok = True
    for path in expected:
        ok = os.path.exists(path)
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {path}")
    print("=" * 60)
    print("Ready for Component 2 (video ingestion pipeline)" if all_ok
          else "Fix missing items above before continuing")

def clone_if_missing(url: str, dest: str):
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
    print(f"[cloned] {url} -> {dest}")

# NOTE: this runs at IMPORT time, not just "if __name__ == '__main__'".
# run_local_evaluation.py does `import Resources` (not `python Resources.py`),
# so __name__ there is "Resources", never "__main__" -- a __main__ guard
# here means this setup code NEVER runs via that import, silently leaving
# GenD/DeepfakeBench/GenConViT/LipFD uncloned. That's the direct cause of
# "some [imports] are not found" downstream: gend_interface.py/lipfd.py
# expect these folders to already exist and never get told they don't.
setup_resources()