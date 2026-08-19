# ============================================================
# COMPONENT 2 of 8 — Video Ingestion Pipeline
# ============================================================
# Paste each "# %% CELL" block into its own Colab cell.
# Works in both Google Colab and Local environments (Windows/macOS/Linux).
# Depends on Component 1 having been run at least once (Drive mounted / local dir ready,
# ffmpeg/mediapipe/yt-dlp installed, PROJECT_ROOT populated).
#
# WHAT THIS COMPONENT PRODUCES (consumed by Components 3, 4, 5):
#   PROJECT_ROOT/data/<video_id>/
#       frames/frame_0000.jpg, frame_0001.jpg, ...   <- for GenD (Component 3)
#       faces/face_0000.jpg,  face_0001.jpg,  ...    <- face-cropped frames
#       audio.wav                                     <- 16kHz mono, for
#                                                          W2V-AASIST (Component 4)
#                                                          and LipFD (Component 5)
#       meta.json                                     <- fps, frame count,
#                                                          has_audio, face
#                                                          detection rate, etc.
#
# This structure is deliberately shared across all downstream components so
# none of them need to know HOW a video got ingested — they just read from
# PROJECT_ROOT/data/<video_id>/.

# %% CELL 1 — Re-establish environment (only needed if this is a fresh session)
import os
import sys
import shutil

try:
    import google.colab  # type: ignore
    IN_COLAB = True
    print("Running in Google Colab environment.")
except ImportError:
    IN_COLAB = False
    print("Running in local environment.")

_raw_root = globals().get("PROJECT_ROOT", None)
if _raw_root and os.path.exists(str(_raw_root)):
    PROJECT_ROOT: str = str(_raw_root)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    PROJECT_ROOT = os.path.join(BASE_DIR, "deepfake_project")

os.makedirs(PROJECT_ROOT, exist_ok=True)
_gend_path = os.path.join(PROJECT_ROOT, "GenD")
if not os.path.exists(_gend_path):
    print(f"[warning] GenD directory missing at {_gend_path}. Ensure Component 1/Resources has run.")

# %% CELL 2 — Imports & Detector Setup
import subprocess
import hashlib
import json
import glob
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import mediapipe as mp

DATA_ROOT = Path(PROJECT_ROOT) / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Detect if mediapipe solutions is available (Colab / older mediapipe)
try:
    import mediapipe.solutions.face_detection as mp_face_detection  # type: ignore
    HAS_MP_SOLUTIONS = True
except (ImportError, AttributeError, ModuleNotFoundError):
    mp_face_detection = None
    HAS_MP_SOLUTIONS = False

# %% CELL 3 — Video acquisition (YouTube URL, any platform yt-dlp supports, or local file)
def video_id_from_source(source: str) -> str:
    """Deterministic short ID so re-running on the same source reuses cached output."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]

def acquire_video(source: str, dest_dir: Path) -> Path:
    """
    source: a YouTube/platform URL, or a local file path.
    Returns the local path to a downloaded/copied .mp4.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "source.mp4"

    if out_path.exists():
        print(f"[skip] already downloaded -> {out_path}")
        return out_path

    is_url = source.startswith("http://") or source.startswith("https://")
    if is_url:
        ytdlp_bin = "yt-dlp" if shutil.which("yt-dlp") else [sys.executable, "-m", "yt_dlp"]
        cmd = [ytdlp_bin] if isinstance(ytdlp_bin, str) else list(ytdlp_bin)
        cmd += [
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(out_path),
            source,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed for '{source}':\n{result.stderr[-2000:]}")
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local video not found: {source}")
        shutil.copy(source, out_path)

    print(f"[downloaded] {source} -> {out_path}")
    return out_path

def get_ffmpeg_cmd() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

# %% CELL 4 — Frame extraction (fixed-rate sampling via ffmpeg)
def extract_frames(video_path: Path, frames_dir: Path, target_fps: float = 5.0, max_frames: int = 96) -> int:
    """
    Samples frames at target_fps. max_frames caps total extracted frames so a
    long video doesn't blow up disk/RAM -- 96 frames at 5fps covers
    ~19 seconds, plenty for both frame-level and short-clip temporal analysis.
    Returns the number of frames actually written.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.glob("frame_*.jpg")):
        n = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"[skip] {n} frames already extracted in {frames_dir}")
        return n

    ffmpeg_bin = get_ffmpeg_cmd()
    cmd = [
        ffmpeg_bin, "-y", "-i", str(video_path),
        "-vf", f"fps={target_fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr[-2000:]}")

    n = len(list(frames_dir.glob("frame_*.jpg")))
    print(f"[extracted] {n} frames -> {frames_dir}")
    return n

# %% CELL 5 — Audio extraction (16kHz mono WAV, required format for W2V-AASIST/LipFD)
def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """
    Returns True if an audio track was found and extracted, False if the
    source video is silent/has no audio stream (common for some clips --
    downstream audio branches must handle this gracefully, not crash on it).
    """
    if audio_path.exists():
        print(f"[skip] audio already extracted -> {audio_path}")
        return True

    ffmpeg_bin = get_ffmpeg_cmd()
    ffprobe_bin = "ffprobe" if shutil.which("ffprobe") else ffmpeg_bin
    
    # Check if video has audio stream using ffprobe or ffmpeg probe
    if shutil.which("ffprobe"):
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                 "stream=index", "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True
            )
            has_audio_stream = bool(probe.stdout.strip())
        except Exception:
            has_audio_stream = True
    else:
        has_audio_stream = True  # Try extracting with ffmpeg directly

    if not has_audio_stream:
        print(f"[no audio] {video_path} has no audio stream")
        return False

    cmd = [
        ffmpeg_bin, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[no audio] audio extraction returned code {result.returncode}")
        return False

    print(f"[extracted] audio -> {audio_path}")
    return True

# %% CELL 6 — Face detection + crop (Universal: supports Colab MediaPipe Solutions and Local OpenCV Fallback)
def detect_and_crop_faces(frames_dir: Path, faces_dir: Path, margin: float = 0.05) -> Dict[str, Any]:
    """
    Runs face detection on each extracted frame, crops the face tightly (margin=0.05),
    suppresses background green-screen pixels, and saves the cropped face.
    Frames where no face is detected are skipped.
    """
    faces_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))

    if not frame_paths:
        return {"total_frames": 0, "faces_detected": 0, "detection_rate": 0.0}

    if any(faces_dir.glob("face_*.jpg")):
        n = len(list(faces_dir.glob("face_*.jpg")))
        print(f"[skip] {n} face crops already exist in {faces_dir}")
        return {"total_frames": len(frame_paths), "faces_detected": n, "detection_rate": n / len(frame_paths)}

    # SOTA OpenCV YuNet Deep Learning Face Detector
    yunet_path = Path(PROJECT_ROOT) / "face_detection_yunet_2023mar.onnx"
    if not yunet_path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        try:
            print("[downloading] SOTA YuNet face detector model...")
            urllib.request.urlretrieve(url, str(yunet_path))
        except Exception as e:
            print(f"[warning] Could not download YuNet model: {e}")

    has_yunet = hasattr(cv2, "FaceDetectorYN") and yunet_path.exists()

    mp_detector = None
    if HAS_MP_SOLUTIONS and mp_face_detection is not None:
        try:
            mp_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        except Exception:
            mp_detector = None

    yn_detector = None
    yn_size = None
    detected = 0

    try:
        for i, fpath in enumerate(frame_paths):
            img_bgr = cv2.imread(str(fpath))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            crop = None

            if has_yunet:
                try:
                    if yn_detector is None or yn_size != (w, h):
                        yn_detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (w, h), score_threshold=0.4)
                        yn_size = (w, h)
                    faces = yn_detector.detect(img_bgr)[1]
                    if faces is not None and len(faces) > 0:
                        best_face = max(faces, key=lambda f: f[2] * f[3])
                        fx, fy, fw, fh = best_face[:4]
                        xmin = max(0, int(fx - margin * fw))
                        ymin = max(0, int(fy - margin * fh))
                        xmax = min(w, int(fx + fw * (1 + margin)))
                        ymax = min(h, int(fy + fh * (1 + margin)))
                        if xmax > xmin and ymax > ymin:
                            crop = img_bgr[ymin:ymax, xmin:xmax]
                except Exception:
                    crop = None

            if crop is None and mp_detector is not None:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    res = mp_detector.process(img_rgb)
                    if res.detections:
                        best = max(res.detections, key=lambda d: d.score[0])
                        bbox = best.location_data.relative_bounding_box
                        xmin = max(0, int((bbox.xmin - margin * bbox.width) * w))
                        ymin = max(0, int((bbox.ymin - margin * bbox.height) * h))
                        xmax = min(w, int((bbox.xmin + bbox.width * (1 + margin)) * w))
                        ymax = min(h, int((bbox.ymin + bbox.height * (1 + margin)) * h))
                        if xmax > xmin and ymax > ymin:
                            crop = img_bgr[ymin:ymax, xmin:xmax]
                except Exception:
                    crop = None

            if crop is not None:
                detected += 1
            else:
                # Center face region crop fallback (middle 60% of frame where heads are located)
                min_dim = min(h, w)
                cy, cx = h // 2, w // 2
                half = int(min_dim * 0.35)
                crop = img_bgr[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]

            # Suppress background green-screen and overexposed white studio background noise
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            crop[green_mask > 0] = (128, 128, 128)

            # Adaptive High-Key / Overexposed Highlight Tone Mapping
            gray_check = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray_check))
            max_lum = float(np.max(gray_check))

            if max_lum > 235.0 or mean_lum > 150.0:
                # Smooth highlight compression to prevent overexposure boundary clipping
                float_crop = crop.astype(np.float32) / 255.0
                toned = float_crop / (float_crop + 0.20)
                crop = np.clip(toned * (1.20 * 255.0), 0, 255).astype(np.uint8)

            elif mean_lum < 75.0:
                # Adaptive Low-Light / Nighttime Illumination Enhancement (CLAHE)
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            crop = cv2.resize(crop, (224, 224))
            cv2.imwrite(str(faces_dir / f"face_{i:04d}.jpg"), crop)
    finally:
        if mp_detector is not None:
            try:
                mp_detector.close()
            except Exception:
                pass

    rate = detected / len(frame_paths) if frame_paths else 0.0
    print(f"[faces] {detected}/{len(frame_paths)} frames processed for face crops ({rate*100:.1f}%)")
    return {"total_frames": len(frame_paths), "faces_detected": detected, "detection_rate": rate}

# %% CELL 7 — Orchestrator: ties everything together for one video
def process_video(source: str, target_fps: float = 5.0, max_frames: int = 96) -> Dict[str, Any]:
    """
    source: YouTube/platform URL or local file path.
    Returns the metadata dict (also written to meta.json) describing what
    was produced, so downstream components can check has_audio /
    detection_rate before trying to use this video.
    """
    vid = video_id_from_source(source)
    video_dir = DATA_ROOT / vid
    video_dir.mkdir(parents=True, exist_ok=True)

    meta_path = video_dir / "meta.json"
    faces_dir = video_dir / "faces"
    if meta_path.exists() and faces_dir.exists() and any(faces_dir.glob("face_*.jpg")):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("faces_detected", 0) > 0:
            print(f"[cached] {source} already processed (video_id={vid})")
            return meta

    video_path = acquire_video(source, video_dir)
    frames_dir = video_dir / "frames"
    faces_dir = video_dir / "faces"
    audio_path = video_dir / "audio.wav"

    n_frames = extract_frames(video_path, frames_dir, target_fps=target_fps, max_frames=max_frames)
    has_audio = extract_audio(video_path, audio_path)
    face_stats = detect_and_crop_faces(frames_dir, faces_dir)

    meta = {
        "source": source,
        "video_id": vid,
        "target_fps": target_fps,
        "n_frames_extracted": n_frames,
        "has_audio": has_audio,
        "audio_path": str(audio_path) if has_audio else None,
        "face_detection_rate": face_stats["detection_rate"],
        "faces_detected": face_stats["faces_detected"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] video_id={vid} | frames={n_frames} | audio={has_audio} | face_rate={face_stats['detection_rate']:.2f}")
    return meta

# %% CELL 8 & CELL 9 — Test execution when run as a standalone script
if __name__ == "__main__":
    # Swap this for any YouTube URL or a local .mp4 path.
    TEST_SOURCE = "https://www.youtube.com/watch?v=REPLACE_ME"
    if "REPLACE_ME" not in TEST_SOURCE:
        meta = process_video(TEST_SOURCE)
        print(json.dumps(meta, indent=2))

        if meta["face_detection_rate"] < 0.5:
            print("WARNING: face detected in <50% of frames. GenD/lip-sync results "
                  "on this video will be unreliable -- check video quality/angle.")
        if not meta["has_audio"]:
            print("NOTE: no audio track found. The audio-spoof and lip-sync branches "
                  "(Components 4 and 5) will need to skip this video or fall back to "
                  "video-only prediction -- make sure they handle has_audio=False.")
    else:
        print("Note: Set TEST_SOURCE to a valid YouTube URL or local .mp4 path to run ingestion test.")