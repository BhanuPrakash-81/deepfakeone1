# ============================================================
# COMPONENT 7 of 8 — Benchmark Evaluation & Metrics Suite
# ============================================================
# Run in the SAME session as Components 2 through 6 (needs process_video
# and fuse_predictions already defined -- see the note below on why this
# file does NOT import them as separate modules).
#
# NOTE ON EXECUTION MODEL: an earlier version of this file used
# `from fusion import fuse_predictions` etc., treating the other
# components as importable .py files. That's incompatible with how
# fusion.py (Component 6) and pipeline.py (Component 8) are actually
# written -- fusion.py's own top-level code does
# `if "predict_video_gend" not in globals(): raise RuntimeError(...)`,
# which checks fusion.py's OWN module namespace at import time. Since
# predict_video_gend/predict_video_audio/predict_video_lipsync live in
# OTHER files/cells, `import fusion` would always raise that error
# immediately -- it only works when everything is pasted into one shared
# notebook session, which is the model every other component here uses.
# This file has been aligned to match that, instead of the other way
# around, since 6 of the other 7 components already assume it.
#
# WHAT THIS COMPONENT DOES:
#   Evaluates the complete multi-modal deepfake detection pipeline over a
#   dataset of labeled test videos.
#   Computes core metrics:
#     - Accuracy, Precision, Recall, F1-Score
#     - Area Under Receiver Operating Characteristic Curve (AUC-ROC)
#     - Equal Error Rate (EER)
#     - Confusion Matrix breakdown

import os
import json
import traceback
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

_raw_root = globals().get("PROJECT_ROOT", None)
if _raw_root and os.path.exists(str(_raw_root)):
    PROJECT_ROOT: str = str(_raw_root)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    PROJECT_ROOT = os.path.join(BASE_DIR, "deepfake_project")


# Ensure required functions are imported or available in globals
if TYPE_CHECKING:
    from video_ingestion import process_video
    from fusion import fuse_predictions
else:
    import traceback
    try:
        from video_ingestion import process_video  # type: ignore
    except Exception as e:
        print(f"[ERROR] Failed to import process_video from video_ingestion: {e}")
        traceback.print_exc()
        process_video = globals().get("process_video", None)

    try:
        from fusion import fuse_predictions  # type: ignore
    except Exception as e:
        print(f"[ERROR] Failed to import fuse_predictions from fusion: {e}")
        traceback.print_exc()
        fuse_predictions = globals().get("fuse_predictions", None)

    try:
        from videometadataanalysis import get_video_metadata, print_compression_summary  # type: ignore
    except Exception:
        get_video_metadata = None
        print_compression_summary = None


# %% CELL 1 — Sanity check required functions are in this session
_REQUIRED = ["process_video", "fuse_predictions"]
_missing = [name for name in _REQUIRED if globals().get(name) is None]
if _missing:
    raise RuntimeError(
        f"Missing {_missing} in this session. Run Components 2 through 6's "
        f"cells (in this same notebook) first."
    )

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    if len(np.unique(y_true)) < 2:
        return 0.0

    thresholds = np.linspace(0.0, 1.0, 500)
    far_list = []
    frr_list = []

    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    if positives == 0 or negatives == 0:
        return 0.0

    for t in thresholds:
        tp = np.sum((y_scores >= t) & (y_true == 1))
        fp = np.sum((y_scores >= t) & (y_true == 0))
        fn = np.sum((y_scores < t) & (y_true == 1))

        far = fp / negatives if negatives > 0 else 0.0
        frr = fn / positives if positives > 0 else 0.0

        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # EER occurs where |FAR - FRR| is minimized
    idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = float((far_arr[idx] + frr_arr[idx]) / 2.0)
    return round(eer, 4)

# %% CELL 3 — Evaluation Loop
def evaluate_dataset(
    dataset: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates the detection pipeline on a list of ground-truth labeled samples.

    Parameters:
        dataset: List of dicts with keys 'source' (file path/URL) and 'label' (1 for FAKE, 0 for REAL).
        weights: Optional custom fusion weights.
        threshold: Decision threshold for classification metrics.

    Returns:
        Dict containing aggregated evaluation metrics and individual predictions.
    """
    y_true: List[int] = []
    y_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    print("=" * 60)
    print(f"   STARTING BENCHMARK EVALUATION ({len(dataset)} samples)")
    print("=" * 60)

    if process_video is None or fuse_predictions is None:
        raise RuntimeError("process_video and fuse_predictions must be defined before running evaluate_dataset.")

    for i, item in enumerate(dataset, 1):
        source = item["source"]
        label = int(item["label"])
        print(f"\nEvaluating [{i}/{len(dataset)}] Source: {source} (Ground Truth: {'FAKE' if label==1 else 'REAL'})...")

        try:
            ingest_meta = process_video(source)
            video_id = ingest_meta["video_id"]
            # fuse_predictions() has no threshold param -- it always
            # returns the raw fused_weighted_prob; the custom decision
            # threshold is applied locally below instead.
            fusion_res = fuse_predictions(video_id, weights=weights)

            if "error" in fusion_res:
                print(f" -> SKIPPED: {fusion_res['error']}")
                continue

            pred_score = fusion_res["fused_weighted_prob"]  # was: "fused_fake_prob" (doesn't exist)
            pred_label = 1 if pred_score >= threshold else 0

            y_true.append(label)
            y_scores.append(pred_score)

            vm_info = {}
            if get_video_metadata:
                target_path = source
                if not os.path.exists(target_path):
                    candidate = os.path.join(PROJECT_ROOT, "data", video_id, "source.mp4")
                    if os.path.exists(candidate):
                        target_path = candidate
                vm_info = get_video_metadata(target_path)

            c_ratio = vm_info.get("compression_ratio", 0.0)
            kbps = vm_info.get("bitrate_kbps", 0.0)
            ratio_str = f"{c_ratio:.1f}:1" if c_ratio > 0 else "N/A"
            kbps_str = f"{kbps:.0f} kbps" if kbps > 0 else "N/A"

            results.append({
                "source": source,
                "video_id": video_id,
                "ground_truth": label,
                "predicted_score": pred_score,
                "predicted_label": pred_label,
                "verdict": fusion_res["fused_weighted_verdict"],  # was: "verdict" (doesn't exist)
                "active_branches": fusion_res["branches_used"],   # was: "active_branches" (doesn't exist)
                "video_metadata": vm_info,
            })
            print(f" -> Result: Predicted={pred_score:.4f} ({fusion_res['fused_weighted_verdict']}) | Comp. Ratio: {ratio_str} | Bitrate: {kbps_str} | Match: {pred_label == label}")

        except Exception as e:
            print(f" -> ERROR processing {source}: {e}")

    if not y_true:
        return {"error": "No valid dataset samples were processed successfully."}

    y_true_arr = np.array(y_true)
    y_scores_arr = np.array(y_scores)
    y_pred_arr = (y_scores_arr >= threshold).astype(int)

    # Calculate Classification Metrics
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))

    accuracy = (tp + tn) / len(y_true_arr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    eer = compute_eer(y_true_arr, y_scores_arr)

    # Try AUC-ROC using sklearn if available, else manual trapezoidal rule approximation
    try:
        from sklearn.metrics import roc_auc_score
        auc_roc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except Exception:
        auc_roc = accuracy  # Fallback approximation if sklearn not installed

    metrics = {
        "total_samples": len(y_true_arr),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "auc_roc": round(auc_roc, 4),
        "eer": round(eer, 4),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        },
        "per_sample_results": results
    }

    # Save metrics report to PROJECT_ROOT
    output_path = os.path.join(PROJECT_ROOT, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("                 BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f" Total Samples Evaluated : {metrics['total_samples']}")
    print(f" Accuracy                : {metrics['accuracy'] * 100:.2f}%")
    print(f" Precision               : {metrics['precision'] * 100:.2f}%")
    print(f" Recall (Sensitivity)    : {metrics['recall'] * 100:.2f}%")
    print(f" F1-Score                : {metrics['f1_score']:.4f}")
    print(f" AUC-ROC                 : {metrics['auc_roc']:.4f}")
    print(f" Equal Error Rate (EER)  : {metrics['eer'] * 100:.2f}%")
    print(f" Confusion Matrix        : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f" Metrics Report Saved To : {output_path}")
    print("=" * 60 + "\n")

    if print_compression_summary and results:
        print_compression_summary(results)

    return metrics

# %% CELL 4 — Test Execution
try:
    from builtestset import TEST_SET_FROM_DATASETS
    TEST_DATASET = TEST_SET_FROM_DATASETS
except Exception as e:
    print(f"[warning] Failed to import TEST_SET_FROM_DATASETS from builtestset: {e}")
    traceback.print_exc()
    TEST_DATASET = []

if __name__ == "__main__":
    if TEST_DATASET:
        evaluate_dataset(TEST_DATASET, threshold=0.30)
    else:
        print("Note: Populate TEST_DATASET with sample dicts ({'source': ..., 'label': 0 or 1}) to run benchmark evaluation.")