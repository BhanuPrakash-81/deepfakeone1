"""
Runner script for Option B (Local Evaluation)
Runs all pipeline components in sequence and evaluates archive (2) dataset.
"""

import sys
import os

# 0. Ensure environment resources are initialized
print("Checking Component 1: Resources setup...")
import Resources

# 1. Video Ingestion
print("Loading Component 2: video_ingestion...")
import video_ingestion
from video_ingestion import process_video

# 2. Visual Branch (GenD)
print("Loading Component 3: gend_interface...")
import gend_interface
from gend_interface import predict_video_gend

# 3. Audio Branch
print("Loading Component 4: audio_spoof...")
import audio_spoof
from audio_spoof import predict_video_audio

# 4. Lip-sync Branch
print("Loading Component 5: lipfd...")
import lipfd
from lipfd import predict_video_lipsync

# Inject into globals so fusion.py and evaluation.py see them
globals()["process_video"] = process_video
globals()["predict_video_gend"] = predict_video_gend
globals()["predict_video_audio"] = predict_video_audio
globals()["predict_video_lipsync"] = predict_video_lipsync

# 5. Fusion
print("Loading Component 6: fusion...")
import fusion
from fusion import fuse_predictions
globals()["fuse_predictions"] = fuse_predictions

# 6. Build Dataset from archive (2)
print("Loading Component 7/8: dataset builder & evaluation...")
import builtestset
from builtestset import build_test_set, CUSTOM_DIR

test_dataset = build_test_set()
print(f"\nBuilt dataset with {len(test_dataset)} videos from {CUSTOM_DIR}.\n")

# 7. Evaluation
import evaluation
from evaluation import evaluate_dataset

if __name__ == "__main__":
    if not test_dataset:
        print("Error: No test samples found. Check CUSTOM_DIR path.")
        sys.exit(1)
    
    metrics = evaluate_dataset(test_dataset)
