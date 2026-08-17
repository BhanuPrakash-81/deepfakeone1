# ============================================================
# COMPONENT 8 of 8 — End-to-End Pipeline (YouTube URL -> Verdict)
# ============================================================
# Run in the SAME session as Components 2, 3, 4, 5, 6 (needs process_video
# and fuse_predictions already defined -- this is the thin public-facing
# wrapper around everything built so far, not a new detection method).
#
# WHAT THIS ADDS ON TOP OF COMPONENT 6:
#   - A single function taking a raw URL/file path in and returning a
#     complete, human-readable report out -- no need to know about
#     video_ids, branch internals, etc.
#   - Quality flags that temper trust in the verdict: low face-detection
#     rate, branches skipped, or branch disagreement all downgrade the
#     reported confidence rather than presenting a bare "FAKE"/"REAL" that
#     hides how shaky the evidence actually was.
#   - Per-video JSON reports saved to disk, and a batch runner for
#     checking a list of sources in one go.

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Any, List

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

# %% CELL 1 — Sanity check required functions are in this session
_REQUIRED = ["process_video", "fuse_predictions"]
_missing = [name for name in _REQUIRED if globals().get(name) is None]

if _missing:
    raise RuntimeError(
        f"Missing {_missing} in this session. Run Components 2 through 6's "
        f"cells (in this same notebook) first -- this component only "
        f"wraps them, it doesn't redefine anything."
    )

PROJECT_ROOT = globals().get("PROJECT_ROOT", None)
if not PROJECT_ROOT:
    try:
        import google.colab  # type: ignore
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')
        PROJECT_ROOT = "/content/drive/MyDrive/deepfake_project"
    else:
        PROJECT_ROOT = os.path.abspath("./deepfake_project")

REPORTS_DIR = Path(PROJECT_ROOT) / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Ready.")

# %% CELL 2 — The main entry point
def analyze_video(source: str, save_report: bool = True) -> Dict[str, Any]:
    """
    source: a YouTube/platform URL or a local video file path.
    Returns a complete report: ingestion quality, per-branch scores,
    fused verdict, and a tempered confidence level that accounts for how
    much evidence was actually available (not just the raw fused number).
    """
    if process_video is None or fuse_predictions is None:
        raise RuntimeError("process_video and fuse_predictions must be defined before running analyze_video.")

    meta = process_video(source)
    fused = fuse_predictions(meta["video_id"])

    quality_flags = []
    if meta.get("face_detection_rate", 1.0) < 0.5:
        quality_flags.append(
            f"LOW FACE DETECTION RATE ({meta['face_detection_rate']*100:.0f}% of frames) "
            f"-- video branch (GenD) result is less trustworthy."
        )
    if not meta.get("has_audio", False):
        quality_flags.append("NO AUDIO TRACK -- audio-spoof and lip-sync branches were skipped entirely.")
    if "error" in fused:
        quality_flags.append(f"FUSION FAILED: {fused['error']}")
    elif fused.get("branches_disagree"):
        quality_flags.append(
            f"BRANCHES DISAGREE (spread={fused['disagreement_spread']:.2f}) "
            f"-- treat the verdict as uncertain, worth manual review."
        )
    if fused.get("branches_used") and len(fused["branches_used"]) == 1:
        quality_flags.append(
            f"ONLY ONE BRANCH PRODUCED A SCORE ({fused['branches_used'][0]}) "
            f"-- this is effectively a single-detector result, not a multimodal one."
        )

    if "error" in fused:
        confidence = "NONE"
    elif quality_flags:
        confidence = "LOW"
    elif len(fused.get("branches_used", [])) >= 2 and not fused.get("branches_disagree"):
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    report = {
        "source": source,
        "video_id": meta["video_id"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "ingestion": {
            "n_frames_extracted": meta.get("n_frames_extracted"),
            "face_detection_rate": meta.get("face_detection_rate"),
            "has_audio": meta.get("has_audio"),
        },
        "fusion": fused,
        "quality_flags": quality_flags,
        "confidence": confidence,
        "final_verdict": fused.get("fused_weighted_verdict", "UNKNOWN"),
        "final_fake_probability": fused.get("fused_weighted_prob"),
    }

    if save_report:
        out_path = REPORTS_DIR / f"{meta['video_id']}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        report["_saved_to"] = str(out_path)

    return report

# %% CELL 3 — Human-readable printer
def print_report(report: Dict[str, Any]):
    print("=" * 60)
    print(f" DEEPFAKE ANALYSIS: {report['source']}")
    print("=" * 60)
    print(f" Verdict     : {report['final_verdict']}")
    prob = report.get("final_fake_probability")
    print(f" Fake prob.  : {prob:.4f}" if prob is not None else " Fake prob.  : N/A")
    print(f" Confidence  : {report['confidence']}")
    print("-" * 60)
    print(" Branch scores:")
    for branch, score in report["fusion"].get("branch_scores", {}).items():
        print(f"   {branch:10s}: {score:.4f}" if score is not None else f"   {branch:10s}: skipped")
    if report["quality_flags"]:
        print("-" * 60)
        print(" Quality flags:")
        for flag in report["quality_flags"]:
            print(f"   - {flag}")
    print("=" * 60)

# %% CELL 4 — Batch runner for checking a list of sources
def analyze_batch(sources: List[str]) -> List[Dict[str, Any]]:
    reports = []
    for i, source in enumerate(sources):
        print(f"\n[{i+1}/{len(sources)}] Analyzing {source}...")
        try:
            report = analyze_video(source)
            print_report(report)
        except Exception as e:
            print(f"  FAILED: {e}")
            report = {"source": source, "error": str(e)}
        reports.append(report)
    return reports

# %% CELL 5 — Single-video test
TEST_SOURCE = "https://www.youtube.com/watch?v=REPLACE_ME"

if "REPLACE_ME" not in TEST_SOURCE:
    report = analyze_video(TEST_SOURCE)
    print_report(report)
else:
    print("Set TEST_SOURCE to a real URL or local file path to run the pipeline.")

# %% CELL 6 — Interactive one-off checker (optional -- run this cell, then paste a URL when prompted)
def interactive_check():
    source = input("Enter a video URL or local file path (or 'quit'): ").strip()
    while source.lower() != "quit":
        try:
            report = analyze_video(source)
            print_report(report)
        except Exception as e:
            print(f"FAILED: {e}")
        source = input("\nEnter another video URL/path (or 'quit'): ").strip()

# Uncomment to use:
# interactive_check()