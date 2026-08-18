# ============================================================
# Compression-aware threshold selection
# ============================================================
# Extends video_metadata_analysis.py. Run AFTER enrich_results_with_metadata()
# has attached ffprobe metadata to your evaluation results.
#
# APPROACH: instead of training separate models per compression tier
# (impractical solo/free-tier, and redundant with GenD's own
# cross-compression generalization design), this calibrates a SEPARATE
# DECISION THRESHOLD per compression bucket using the same EER-crossing
# logic evaluation.py already computes globally. Compression pushes
# scores toward "REAL" (artifacts get smoothed away), so heavily
# compressed video likely needs a LOWER threshold to maintain the same
# recall -- this finds that empirically instead of guessing.
#
# HONEST LIMITATION: this needs enough samples PER BUCKET to compute a
# meaningful threshold (a bucket with 3 videos gives you noise, not a
# calibrated threshold). Falls back to the global threshold when a
# bucket is too small, rather than pretending a tiny sample is reliable.
import os
import json
import subprocess
import shutil
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

MIN_SAMPLES_PER_BUCKET = 15  # below this, don't trust a bucket-specific threshold

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts video metadata including resolution, duration, bitrate, 
    uncompressed size estimate, and exact compression ratio (Uncompressed Size : Compressed Size).
    """
    if not os.path.exists(video_path):
        return {"error": f"File not found: {video_path}"}

    file_size_bytes = os.path.getsize(video_path)
    width, height, fps, total_frames, duration_sec = 0, 0, 0.0, 0, 0.0

    # Try extracting precise metadata via ffprobe if installed
    if shutil.which("ffprobe"):
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        width = int(stream.get("width", 0))
                        height = int(stream.get("height", 0))
                        total_frames = int(stream.get("nb_frames", 0))
                        r_fps = stream.get("r_frame_rate", "0/1")
                        if "/" in r_fps:
                            num, den = map(float, r_fps.split("/"))
                            fps = num / den if den > 0 else 0.0
                        break
                format_info = data.get("format", {})
                duration_sec = float(format_info.get("duration", 0.0))
        except Exception:
            pass

    # OpenCV fallback if ffprobe didn't get resolution/frames
    if width == 0 or total_frames == 0:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0 and total_frames > 0:
                    duration_sec = total_frames / fps
        except Exception:
            pass

    if duration_sec == 0.0 and fps > 0 and total_frames > 0:
        duration_sec = total_frames / fps

    bitrate_bps = int((file_size_bytes * 8) / duration_sec) if duration_sec > 0 else 0
    bitrate_kbps = round(bitrate_bps / 1000.0, 2)

    # Estimate uncompressed YUV420 video frame payload (1.5 bytes per pixel * total frames)
    uncompressed_bytes = int(width * height * 1.5 * total_frames) if (width > 0 and height > 0 and total_frames > 0) else 0
    compression_ratio = round(uncompressed_bytes / file_size_bytes, 2) if (file_size_bytes > 0 and uncompressed_bytes > 0) else 0.0

    return {
        "file_size_bytes": file_size_bytes,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "bitrate_bps": bitrate_bps,
        "bitrate_kbps": bitrate_kbps,
        "uncompressed_bytes": uncompressed_bytes,
        "compression_ratio": compression_ratio,  # e.g. 50.2 means 50.2:1 compression ratio
    }

def enrich_results_with_metadata(
    results: List[Dict[str, Any]], 
    project_root: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Enriches per-sample evaluation results with video metadata, including bitrate and compression ratio.
    """
    project_root = project_root or globals().get("PROJECT_ROOT", "deepfake_project")
    for item in results:
        source = item.get("source", "")
        video_id = item.get("video_id", "")

        video_path = source
        if not os.path.exists(video_path) and video_id:
            candidate = os.path.join(project_root, "data", video_id, "source.mp4")
            if os.path.exists(candidate):
                video_path = candidate

        meta = get_video_metadata(video_path)
        item["video_metadata"] = meta

    return results

def analyze_misclassifications(
    enriched_results: List[Dict[str, Any]], 
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Analyzes misclassified videos (False Negatives & False Positives) 
    and prints a detailed breakdown with compression ratio and bitrate.
    Returns the list of misclassified items sorted by highest compression ratio.
    """
    misclassified = []
    for item in enriched_results:
        gt = item.get("ground_truth", item.get("label"))
        score = item.get("predicted_score", 0.0)
        pred = 1 if score >= threshold else 0

        if pred != gt:
            item_copy = dict(item)
            item_copy["error_type"] = "False Negative (Fake missed)" if gt == 1 else "False Positive (Real flagged)"
            misclassified.append(item_copy)

    # Sort misclassified items by highest compression ratio
    misclassified.sort(
        key=lambda x: x.get("video_metadata", {}).get("compression_ratio", 0.0), 
        reverse=True
    )

    print("=" * 85)
    print(f" MISCLASSIFICATION & COMPRESSION ANALYSIS (Threshold = {threshold})")
    print(f" Total Errors: {len(misclassified)}")
    print("=" * 85)
    print(f"{'Source / Video ID':<32} {'Type':<18} {'GT':<5} {'Score':<7} {'Bitrate':<10} {'Comp. Ratio':<12}")
    print("-" * 85)

    for m in misclassified:
        vid = m.get("video_id") or os.path.basename(m.get("source", "unknown"))
        vid_short = vid[:30]
        err_type = "FN (Fake->Real)" if m["ground_truth"] == 1 else "FP (Real->Fake)"
        gt_str = "FAKE" if m["ground_truth"] == 1 else "REAL"
        score_str = f"{m.get('predicted_score', 0.0):.4f}"
        
        vm = m.get("video_metadata", {})
        kbps = f"{vm.get('bitrate_kbps', 0):.0f} kbps" if "bitrate_kbps" in vm else "N/A"
        c_ratio = f"{vm.get('compression_ratio', 0.0):.1f}:1" if "compression_ratio" in vm else "N/A"

        print(f"{vid_short:<32} {err_type:<18} {gt_str:<5} {score_str:<7} {kbps:<10} {c_ratio:<12}")

    print("=" * 85 + "\n")
    return misclassified


def compute_eer_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
    """Same FAR=FRR crossing logic as evaluation.py's compute_eer(), returns just the threshold."""
    if len(np.unique(y_true)) < 2:
        return None

    thresholds = np.linspace(0.0, 1.0, 500)
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return None

    far_list, frr_list = [], []
    for t in thresholds:
        fp = np.sum((y_scores >= t) & (y_true == 0))
        fn = np.sum((y_scores < t) & (y_true == 1))
        far_list.append(fp / negatives)
        frr_list.append(fn / positives)

    idx = np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))
    return float(thresholds[idx])

def bucket_by_bitrate(enriched: List[Dict[str, Any]], bins_kbps: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Assigns each result a bitrate bucket label. Default bins are a rough
    starting point (low/medium/high) -- adjust based on what your actual
    dataset's bitrate distribution looks like (print the raw values first
    if unsure; don't trust these defaults blindly for your specific data).
    """
    bins_kbps = bins_kbps or [500, 2000]  # <500kbps=low, 500-2000=medium, >2000=high

    for r in enriched:
        bps = r["video_metadata"].get("bitrate_bps")
        if bps is None:
            r["video_metadata"]["bitrate_bucket"] = "unknown"
            continue
        kbps = bps / 1000
        if kbps < bins_kbps[0]:
            r["video_metadata"]["bitrate_bucket"] = "low"
        elif kbps < bins_kbps[1]:
            r["video_metadata"]["bitrate_bucket"] = "medium"
        else:
            r["video_metadata"]["bitrate_bucket"] = "high"
    return enriched

def infer_label_from_filepath(path: str, base_dir: Optional[str] = None) -> int:
    """
    Infers whether a video is Real (0) or Fake (1) using its directory or file path structure.
    Ignores top-level root folder names.
    """
    if base_dir and os.path.exists(base_dir):
        try:
            rel_p = os.path.relpath(path, base_dir)
        except Exception:
            rel_p = os.path.basename(path)
    else:
        parts = os.path.normpath(path).split(os.sep)
        rel_p = os.sep.join(parts[-3:]) if len(parts) >= 3 else path

    p_lower = rel_p.lower().replace("\\", "/")
    fake_keywords = ["fake", "manipulated", "deepfake", "deepfakedetection", "synthesis"]
    real_keywords = ["real", "original", "actor", "youtube"]

    if any(k in p_lower for k in fake_keywords):
        return 1  # Fake
    elif any(k in p_lower for k in real_keywords):
        return 0  # Real
    return 1  # Default to Fake

def split_real_and_fake(dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Divides a combined dataset into separate 'real' and 'fake' lists using ground_truth, label, or file path.
    """
    real_items, fake_items = [], []
    for item in dataset:
        path = item.get("source") or item.get("video_path") or item.get("filepath") or ""
        label = item.get("ground_truth", item.get("label"))
        if label is None and path:
            label = infer_label_from_filepath(path)
        
        if label == 0:
            real_items.append(item)
        else:
            fake_items.append(item)

    return {"real": real_items, "fake": fake_items}

def calibrate_thresholds_per_bucket(
    enriched: List[Dict[str, Any]],
    bucket_key: str = "bitrate_bucket",
    global_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {bucket_value: {"threshold": float, "n_samples": int, "source": "calibrated"|"fallback_global"}}
    """
    groups: Dict[Any, List[Dict]] = {}
    for r in enriched:
        val = r["video_metadata"].get(bucket_key, "unknown")
        groups.setdefault(val, []).append(r)

    result = {}
    print(f"{'Bucket':<15}{'n':<6}{'Threshold':<12}{'Source'}")
    print("-" * 50)
    for val, items in sorted(groups.items(), key=lambda x: str(x[0])):
        n = len(items)
        if n < MIN_SAMPLES_PER_BUCKET:
            result[val] = {"threshold": global_threshold, "n_samples": n, "source": "fallback_global (too few samples)"}
            print(f"{str(val):<15}{n:<6}{global_threshold:<12.3f}fallback (n<{MIN_SAMPLES_PER_BUCKET})")
            continue

        y_true_list = []
        for r in items:
            g = r.get("ground_truth", r.get("label"))
            if g is None:
                path = r.get("source") or r.get("video_path") or ""
                g = infer_label_from_filepath(path)
            y_true_list.append(g)

        y_true = np.array(y_true_list)
        y_scores = np.array([r["predicted_score"] for r in items])
        thresh = compute_eer_threshold(y_true, y_scores)

        if thresh is None:
            result[val] = {"threshold": global_threshold, "n_samples": n, "source": "fallback_global (single class in bucket)"}
            print(f"{str(val):<15}{n:<6}{global_threshold:<12.3f}fallback (single class)")
        else:
            result[val] = {"threshold": thresh, "n_samples": n, "source": "calibrated"}
            print(f"{str(val):<15}{n:<6}{thresh:<12.3f}calibrated")

    return result

def predict_with_bucket_threshold(
    predicted_score: float,
    video_metadata: Dict[str, Any],
    bucket_thresholds: Dict[str, Dict[str, Any]],
    bucket_key: str = "bitrate_bucket",
) -> str:
    """Apply the right bucket's threshold to a new prediction instead of a flat 0.5."""
    bucket_val = video_metadata.get(bucket_key, "unknown")
    threshold = bucket_thresholds.get(bucket_val, {}).get("threshold", 0.5)
    return "FAKE" if predicted_score >= threshold else "REAL"

# %% Example usage:
#   metrics = evaluate_dataset(TEST_DATASET)
#   enriched = enrich_results_with_metadata(metrics["per_sample_results"])
#   enriched = bucket_by_bitrate(enriched)
#   bucket_thresholds = calibrate_thresholds_per_bucket(enriched, bucket_key="bitrate_bucket")
#   # Then for a new video:
#   verdict = predict_with_bucket_threshold(new_score, new_video_metadata, bucket_thresholds)