"""
Runner script to test any YouTube video URL or online video link.
Usage:
    python run_youtube_video.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
"""

import sys
import os

# 1. Environment & resources setup
print("Loading Component 1: Resources setup...")
import Resources

# 2. Video Ingestion (yt-dlp YouTube downloader & face/audio extraction)
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

# Inject into globals so fusion.py and pipeline.py can access them
globals()["process_video"] = process_video
globals()["predict_video_gend"] = predict_video_gend
globals()["predict_video_audio"] = predict_video_audio
globals()["predict_video_lipsync"] = predict_video_lipsync

# 6. Fusion
print("Loading Component 6: fusion...")
import fusion
from fusion import fuse_predictions
globals()["fuse_predictions"] = fuse_predictions

# 7. Pipeline Wrapper
print("Loading Component 8: pipeline...")
import pipeline
from pipeline import analyze_video, print_report

def test_youtube_video(url: str):
    print(f"\n" + "=" * 60)
    print(f" TESTING YOUTUBE URL: {url}")
    print("=" * 60 + "\n")
    report = analyze_video(url, save_report=True)
    print_report(report)
    return report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        # Default sample YouTube URL if no argument is passed
        target_url = "https://www.youtube.com/watch?v=cQ54GDm1eL0"
        print(f"No URL specified. Testing default sample URL: {target_url}\n")
    
    test_youtube_video(target_url)
