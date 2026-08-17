# ============================================================
# COMPONENT 6 of 8 — Fusion (combining Components 3, 4, 5)
# ============================================================
# Run this in the SAME session/notebook as Components 3, 4, and 5 --
# it calls predict_video_gend(), predict_video_audio(), and
# predict_video_lipsync() directly rather than reloading anything, since
# those already hold their (large) models in memory via lazy singletons.
# If you get a NameError here, go run those components' cells first.
#
# WHY NOT SIMPLE AVERAGING: GenD, the audio-spoof branch, and LipFD each
# specialize in a DIFFERENT manipulation type (visual face-swap artifacts,
# voice cloning, lip-dub mismatch). A video manipulated only one of these
# ways will likely score near 0.5 (uninformative) on the other two
# branches -- naive averaging would dilute a strong, correct signal from
# the one branch that actually caught it. This component computes BOTH:
#   - a weighted average (renormalized over whichever branches actually
#     produced a score -- see WARNING below on why this matters)
#   - a max-rule score (the single highest branch score -- appropriate for
#     "OR"-style combination of complementary/orthogonal detectors)
# and surfaces both, rather than silently picking one and hiding the
# disagreement between branches.
#
# HONEST CAVEAT: the default weights below are a heuristic starting point,
# NOT empirically tuned -- none of these three branches has been validated
# against real labeled data from your target distribution yet (that's
# Component 7). Revisit these weights once you have that.

# %% CELL 1 — Sanity check that Components 3/4/5 have been run in this session
import numpy as np
from typing import TYPE_CHECKING, Dict, Any, Optional, List

# Static type checkers (Pyright/Pylance/IDEs) need explicit imports for symbol resolution
if TYPE_CHECKING:
    from gend_interface import predict_video_gend
    from audio_spoof import predict_video_audio
    from lipfd import predict_video_lipsync
else:
    import traceback
    if "predict_video_gend" not in globals():
        try:
            from gend_interface import predict_video_gend
        except Exception as e:
            print(f"[ERROR] Failed to import predict_video_gend from gend_interface: {e}")
            traceback.print_exc()

    if "predict_video_audio" not in globals():
        try:
            from audio_spoof import predict_video_audio
        except Exception as e:
            print(f"[ERROR] Failed to import predict_video_audio from audio_spoof: {e}")
            traceback.print_exc()

    if "predict_video_lipsync" not in globals():
        try:
            from lipfd import predict_video_lipsync
        except Exception as e:
            print(f"[ERROR] Failed to import predict_video_lipsync from lipfd: {e}")
            traceback.print_exc()

_REQUIRED_FUNCS = ["predict_video_gend", "predict_video_audio", "predict_video_lipsync"]
_missing = [name for name in _REQUIRED_FUNCS if name not in globals()]
if _missing:
    raise RuntimeError(
        f"Missing {_missing} in this session. Run Components 3, 4, and 5's "
        f"cells (in this same notebook/session) before running fusion -- "
        f"their model-loading is lazy but the functions themselves must "
        f"already be defined."
    )
print("Components 3, 4, 5 all present in session. Ready to fuse.")

# %% CELL 2 — Default branch weights (heuristic, see header caveat)
DEFAULT_WEIGHTS = {
    "video": 0.50,     # GenD -- always available if a face was detected,
                        # the most general-purpose signal
    "audio": 0.30,      # audio-spoof branch -- only meaningful if has_audio
    "lipsync": 0.20,    # LipFD -- only meaningful if has_audio AND the
                        # video is a talking-head-framed shot
}

DISAGREEMENT_THRESHOLD = 0.40  # flag when branches disagree by more than this

# %% CELL 3 — Fusion logic
def fuse_predictions(video_id: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Runs all three branches on video_id and combines them. Any branch that
    couldn't produce a score (no face detected, no audio track, video too
    short, etc.) is excluded from the weighted average and the remaining
    weights are renormalized to sum to 1 -- WITHOUT this renormalization,
    a missing branch would silently pull the weighted average toward 0
    (falsely "REAL") rather than reflecting only the branches that
    actually had something to say.
    """
    weights = weights or DEFAULT_WEIGHTS

    gend_result = predict_video_gend(video_id)
    audio_result = predict_video_audio(video_id)
    lipsync_result = predict_video_lipsync(video_id)

    branch_scores: Dict[str, Optional[float]] = {
        "video": gend_result.get("video_fake_prob"),
        "audio": audio_result.get("audio_fake_prob"),
        "lipsync": lipsync_result.get("lipsync_fake_prob"),
    }
    branch_raw_results = {
        "video": gend_result,
        "audio": audio_result,
        "lipsync": lipsync_result,
    }

    available = {k: v for k, v in branch_scores.items() if v is not None}
    skipped = {
        k: branch_raw_results[k].get("note", branch_raw_results[k].get("error", "no score produced"))
        for k, v in branch_scores.items() if v is None
    }

    if not available:
        return {
            "video_id": video_id,
            "error": "No branch produced a usable score for this video.",
            "skipped": skipped,
            "branch_raw_results": branch_raw_results,
        }

    # Renormalized weighted average over available branches only
    weight_sum = sum(weights[k] for k in available)
    fused_weighted = sum(weights[k] * available[k] for k in available) / weight_sum

    # Max-rule alternative
    fused_max = max(available.values())
    max_branch = max(available, key=lambda k: available[k])

    # Disagreement check
    disagreement = (max(available.values()) - min(available.values())) if len(available) > 1 else 0.0
    branches_disagree = disagreement >= DISAGREEMENT_THRESHOLD

    result = {
        "video_id": video_id,
        "branch_scores": branch_scores,             # includes None for skipped branches
        "branches_used": list(available.keys()),
        "branches_skipped": skipped,
        "fused_weighted_prob": fused_weighted,
        "fused_weighted_verdict": "FAKE" if fused_weighted >= 0.5 else "REAL",
        "fused_max_rule_prob": fused_max,
        "fused_max_rule_verdict": "FAKE" if fused_max >= 0.5 else "REAL",
        "max_rule_driven_by": max_branch,
        "branches_disagree": branches_disagree,
        "disagreement_spread": disagreement,
        "branch_raw_results": branch_raw_results,   # full detail from each component, for debugging
    }
    return result

# %% CELL 4 — Human-readable summary printer
def print_fusion_result(result: Dict[str, Any]):
    if "error" in result:
        print(f"video_id={result['video_id']}: {result['error']}")
        for k, reason in result.get("skipped", {}).items():
            print(f"  [{k}] skipped: {reason}")
        return

    print(f"video_id={result['video_id']}")
    print("-" * 50)
    for branch, score in result["branch_scores"].items():
        if score is None:
            print(f"  {branch:10s}: SKIPPED ({result['branches_skipped'].get(branch)})")
        else:
            print(f"  {branch:10s}: {score:.4f}")
    print("-" * 50)
    print(f"  Weighted fusion : {result['fused_weighted_prob']:.4f}  -> {result['fused_weighted_verdict']}")
    print(f"  Max-rule fusion : {result['fused_max_rule_prob']:.4f}  -> {result['fused_max_rule_verdict']} "
          f"(driven by '{result['max_rule_driven_by']}')")
    if result["branches_disagree"]:
        print(f"  WARNING: branches disagree by {result['disagreement_spread']:.2f} "
              f"(>= {DISAGREEMENT_THRESHOLD} threshold) -- worth manual review, "
              f"not just trusting the fused number.")
    print("-" * 50)

# %% CELL 5 — Test on the video processed in earlier components
TEST_VIDEO_ID = "REPLACE_WITH_VIDEO_ID"

if TEST_VIDEO_ID != "REPLACE_WITH_VIDEO_ID":
    result = fuse_predictions(TEST_VIDEO_ID)
    print_fusion_result(result)
else:
    print("Set TEST_VIDEO_ID to a video_id folder under PROJECT_ROOT/data/.")