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

import numpy as np
from typing import Dict, Any, List, Optional

MIN_SAMPLES_PER_BUCKET = 15  # below this, don't trust a bucket-specific threshold

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

def infer_label_from_filepath(path: str) -> int:
    """
    Infers whether a video is Real (0) or Fake (1) using its directory or file path structure.
    """
    p_lower = str(path).lower().replace("\\", "/")
    real_keywords = ["/real/", "_real", "real_", "original", "actor", "youtube"]
    fake_keywords = ["/fake/", "_fake", "fake_", "manipulated", "deepfake", "deepfakedetection", "synthesis"]

    if any(k in p_lower for k in real_keywords):
        return 0  # Real
    elif any(k in p_lower for k in fake_keywords):
        return 1  # Fake
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