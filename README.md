# Multimodal Deepfake Detector — Setup & Usage Guide

Detects deepfakes in video from YouTube or any platform, combining three
independent, pretrained, orthogonal detection signals:
- **Visual artifacts** (GenD — CLIP/DINO-based, frozen backbone)
- **Voice cloning / audio spoofing** (fine-tuned Wav2Vec2-XLSR)
- **Lip-sync mismatch** (LipFD)

Built for free-tier resources: Google Colab (T4 GPU) or a normal laptop.
No custom model training required — every component wraps an existing
pretrained model.

## File map

| File | What it does | Depends on |
|---|---|---|
| `Resources.py` | Environment setup, clones GenD/DeepfakeBench/GenConViT/LipFD | — |
| `video_ingestion.py` | Downloads video, extracts frames/audio, detects+crops faces | Resources.py |
| `gend_interface.py` | Visual deepfake branch | Resources.py, video_ingestion.py |
| `audio_spoof.py` | Audio voice-cloning branch | Resources.py, video_ingestion.py |
| `lipfd.py` | Lip-sync mismatch branch | Resources.py, video_ingestion.py |
| `fusion.py` | Combines the three branches into one score | gend_interface.py, audio_spoof.py, lipfd.py |
| `evaluation.py` | Measures accuracy against labeled test videos | fusion.py, video_ingestion.py |
| `pipeline.py` | Single `analyze_video(url)` entry point | fusion.py, video_ingestion.py |

## How to run it (Google Colab)

**Everything must be pasted into cells of the SAME notebook, in this
exact order, run top to bottom.** The components share state through
Python's `globals()` — each later file checks that the functions it
needs are already defined, and will tell you clearly which earlier file
to run if something's missing.

1. Open a new Colab notebook, set **Runtime > Change runtime type > T4 GPU**.
2. Paste `Resources.py`'s `# %% CELL` blocks in, one Colab cell per block. Run top to bottom.
   - Takes a few minutes (cloning repos, installing packages).
   - Only needs to be re-run once per Google Drive account — everything
     persists to Drive, so future sessions just need Cell 1 (mount Drive).
3. Paste `video_ingestion.py`'s cells in. Run top to bottom.
   - Edit `TEST_SOURCE` in the last cell to a real YouTube URL or file path
     to verify ingestion works before moving on.
4. Paste `gend_interface.py`'s cells in. Run top to bottom.
   - First run downloads the GenD model (~GB scale) — cached after that.
   - Watch the calibration output in Cell 4 — if it prints a weak-confidence
     warning, something's wrong with model loading; don't proceed until it's clean.
5. Paste `audio_spoof.py`'s cells in. Run top to bottom.
6. Paste `lipfd.py`'s cells in. Run top to bottom.
   - This is the most fragile component (see its header comment). The
     checkpoint download via `gdown` can occasionally fail/rate-limit on
     Google Drive — if so, the direct link is in the comment; download
     manually and place at `LipFD/checkpoints/ckpt.pth`.
   - **No automatic polarity check exists for this branch** — sanity-check
     it on a video with known ground truth before trusting it.
7. Paste `fusion.py`'s cells in. Run top to bottom.
8. **Now you have a working pipeline.** Two ways to use it:
   - **Quick check on one video**: paste `pipeline.py`'s cells in, set
     `TEST_SOURCE`, run. Or use `interactive_check()` (Cell 6) to paste
     URLs one at a time without editing code.
   - **Measure real accuracy**: paste `evaluation.py`'s cells in, fill in
     `TEST_SET`/`TEST_DATASET` with labeled examples (`{"source": url_or_path, "label": 1_or_0}`),
     run. This is what tells you whether the pipeline actually works on
     your target distribution, not just in theory.

## What "done" looks like vs. what still needs validation

**Empirically verified before you ever run it:**
- GenD's fake/real label mapping (self-calibrates against known labeled examples each session)
- Audio branch's fake/real label mapping (reads the model's own config, keyword-matched)
- All repo URLs and model IDs (checked live before being included in any component)

**NOT yet empirically verified — you need to do this:**
- LipFD's fake/real polarity (no calibration examples exist in that repo)
- The default fusion weights (`video=0.5, audio=0.3, lipsync=0.2` in `fusion.py`) — heuristic placeholders, not tuned on your data
- Whether the lip-sync and audio branches help at all on your target videos, vs. GenD alone — `evaluation.py`'s per-branch metrics table answers this directly

**Run `evaluation.py` on a real labeled test set before trusting any verdict from `pipeline.py`.** That's not optional polish — it's the only way to know if the polarity/weight assumptions above are actually correct for your use case.

## Known limitations (read before you rely on this for anything important)

- **LipFD assumes centered talking-head framing.** It doesn't do face
  detection — fixed pixel-coordinate crops. Off-center or non-portrait
  framing may not land on the mouth region correctly.
- **Audio/lip-sync branches only fire when the video has an audio track.**
  Silent videos or ones where extraction fails get GenD-only verdicts —
  `pipeline.py` flags this in `quality_flags`, but it's a real reduction
  in evidence, not a false confidence "REAL."
- **None of these models were trained on your exact target distribution**
  (real-world YouTube video, whatever platforms you're checking). All
  three were trained on academic benchmark datasets. Expect some accuracy
  drop on genuinely "in the wild" content — this is exactly the gap the
  `evaluation.py` results will reveal, and exactly the kind of finding
  that supports the "real-world generalization" evaluation-paper angle
  discussed earlier, if that's a direction you want to take this.

## If something breaks

Every component's header comment documents what it depends on and what
it hands off downstream. Start there. If a cell references a function
that's undefined, it's almost always because an earlier component's
cells weren't run in the same session — each file's Cell 1 checks for
this and tells you which one to go back to.
