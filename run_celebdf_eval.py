"""
Runner script to evaluate 10 Real YouTube videos and 10 Deepfake videos
from Celeb-DF v3 (Celeb-DF++) dataset using the multi-modal detection pipeline.

You can delete this standalone file whenever you are done!
Usage:
    python run_celebdf_eval.py
"""

import os
import sys
import random
import traceback

# 1. Environment & resources setup
print("Loading Component 1: Resources setup...")
import Resources

# 2. Video Ingestion
print("Loading Component 2: video_ingestion...")
import video_ingestion
from video_ingestion import process_video

# 3. Visual Branch (GenD)
print("Loading Component 3: gend_interface...")
import gend_interface
from gend_interface import predict_video_gend

# 4. Audio Branch
print("Loading Component 4: audio_spoof...")
import audio_spoof
from audio_spoof import predict_video_audio

# 5. Lip-sync Branch
print("Loading Component 5: lipfd...")
import lipfd
from lipfd import predict_video_lipsync

# Inject required components into globals for fusion and evaluation
globals()["process_video"] = process_video
globals()["predict_video_gend"] = predict_video_gend
globals()["predict_video_audio"] = predict_video_audio
globals()["predict_video_lipsync"] = predict_video_lipsync

# 6. Fusion
print("Loading Component 6: fusion...")
import fusion
from fusion import fuse_predictions
globals()["fuse_predictions"] = fuse_predictions

# 7. Evaluation
print("Loading Component 7/8: evaluation module...")
import evaluation
from evaluation import evaluate_dataset

CELEB_DF_V3_ROOT = r"C:\Users\chimm\Downloads\Celeb-DF++\Celeb-DF-v3"

def build_celebdf_10_samples():
    """
    Selects 10 Real YouTube videos and 10 Deepfake videos from Celeb-DF v3.
    Returns:
        List[Dict[str, Any]]: List of dicts with 'source' and 'label' (0=REAL, 1=FAKE).
    """
    dataset = []
    
    # 1. Select 10 Real YouTube videos
    youtube_real_dir = os.path.join(CELEB_DF_V3_ROOT, "YouTube-real")
    if os.path.exists(youtube_real_dir):
        real_files = sorted([os.path.join(youtube_real_dir, f) for f in os.listdir(youtube_real_dir) if f.endswith(".mp4")])
        # Pick 10 deterministically spaced samples
        if len(real_files) >= 10:
            step = len(real_files) // 10
            selected_reals = real_files[::step][:10]
        else:
            selected_reals = real_files
        
        for r_path in selected_reals:
            dataset.append({"source": r_path, "label": 0})
    else:
        print(f"[Warning] YouTube-real folder not found at {youtube_real_dir}")

    # 2. Select 10 Deepfake videos (FaceSwap & TalkingFace)
    fake_candidates = []
    
    # FaceSwap deepfakes (Celeb-DF-v2, InSwapper, SimSwap, etc.)
    faceswap_dir = os.path.join(CELEB_DF_V3_ROOT, "Celeb-synthesis", "FaceSwap", "Celeb-DF-v2")
    if os.path.exists(faceswap_dir):
        f_files = sorted([os.path.join(faceswap_dir, f) for f in os.listdir(faceswap_dir) if f.endswith(".mp4")])
        if len(f_files) >= 5:
            step = len(f_files) // 5
            fake_candidates.extend(f_files[::step][:5])
        else:
            fake_candidates.extend(f_files)

    # TalkingFace deepfakes (AniTalker, SadTalker, EchoMimic - which contain audio)
    talking_dir = os.path.join(CELEB_DF_V3_ROOT, "Celeb-synthesis", "TalkingFace", "SadTalker")
    if os.path.exists(talking_dir):
        t_files = sorted([os.path.join(talking_dir, f) for f in os.listdir(talking_dir) if f.endswith(".mp4")])
        if len(t_files) >= 5:
            step = len(t_files) // 5
            fake_candidates.extend(t_files[::step][:5])
        else:
            fake_candidates.extend(t_files)
            
    for f_path in fake_candidates[:10]:
        dataset.append({"source": f_path, "label": 1})

    return dataset

if __name__ == "__main__":
    print(f"\nConstructing test evaluation set from {CELEB_DF_V3_ROOT}...")
    dataset = build_celebdf_10_samples()
    
    real_count = sum(1 for d in dataset if d['label'] == 0)
    fake_count = sum(1 for d in dataset if d['label'] == 1)
    
    print(f"Total test samples selected: {len(dataset)} ({real_count} REAL, {fake_count} FAKE)\n")
    
    if not dataset:
        print("Error: Could not locate Celeb-DF v3 videos.")
        sys.exit(1)
        
    metrics = evaluate_dataset(dataset, threshold=0.50)
