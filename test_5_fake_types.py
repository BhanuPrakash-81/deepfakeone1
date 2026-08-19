"""
Enhanced Multi-Modal Evaluation Script for 20 Video Samples (Real & 5 Deepfake Types)
Combining GenConViT, GenD, FFT Frequency Artifact Analysis, Audio-Spoof, and LipFD.

Enhancements Included:
1. PIL Gaussian Pre-Smoothing on face crops to eliminate H.264 video compression ringing noise on Real faces.
2. FFT Frequency Domain High-Pass Spectral Analysis to catch seamless InSwapper grid artifacts.
3. Enhanced Adaptive Multi-Modal Fusion Formula.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

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

# Inject into globals for fusion module
globals()["process_video"] = process_video
globals()["predict_video_gend"] = predict_video_gend
globals()["predict_video_audio"] = predict_video_audio
globals()["predict_video_lipsync"] = predict_video_lipsync

# 6. Fusion
print("Loading Component 6: fusion...")
import fusion
from fusion import fuse_predictions

# 7. Robust GenConViT Model Integration with Gaussian Pre-Smoothing
print("Loading Component 7: GenConViT Model...")
_raw_root = globals().get("PROJECT_ROOT", None)
if _raw_root and os.path.exists(str(_raw_root)):
    PROJECT_ROOT = str(_raw_root)
else:
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfake_project")

DATA_ROOT = Path(PROJECT_ROOT) / "data"

GENCONVIT_DIR = os.path.join(PROJECT_ROOT, "GenConViT")
_GENCONVIT_MODEL = None
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_genconvit_model():
    global _GENCONVIT_MODEL
    if _GENCONVIT_MODEL is None:
        if os.path.exists(GENCONVIT_DIR):
            old_cwd = os.getcwd()
            os.chdir(GENCONVIT_DIR)
            if GENCONVIT_DIR not in sys.path:
                sys.path.insert(0, GENCONVIT_DIR)
            try:
                from model.config import load_config  # type: ignore
                from model.genconvit_ed import GenConViTED  # type: ignore
                
                cfg = load_config()
                cfg['model']['backbone'] = 'convnext_tiny'
                cfg['model']['embedder'] = 'swin_tiny_patch4_window7_224'
                cfg['model']['type'] = 'tiny'
                
                model = GenConViTED(cfg, pretrained=False)
                ckpt_path = os.path.join(GENCONVIT_DIR, "weight", "genconvit_ed_inference.pth")
                
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                    model_keys = set(model.state_dict().keys())
                    
                    filtered_dict = {}
                    for k, v in state_dict.items():
                        if k in model_keys and model.state_dict()[k].shape == v.shape:
                            filtered_dict[k] = v
                        else:
                            alt_k = k.replace(".head.fc.", ".head.")
                            if alt_k in model_keys and model.state_dict()[alt_k].shape == v.shape:
                                filtered_dict[alt_k] = v
                                
                    model.load_state_dict(filtered_dict, strict=False)
                    print(f"GenConViT model loaded successfully with {len(filtered_dict)} matching weights!")
                else:
                    print(f"[Warning] Weight file missing: {ckpt_path}")
                    
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                _GENCONVIT_MODEL = model
            except Exception as e:
                print(f"[Warning] Failed to load GenConViT model: {e}")
            finally:
                os.chdir(old_cwd)
    return _GENCONVIT_MODEL

def predict_video_genconvit(video_id: str) -> dict:
    """Predict video with PIL Gaussian pre-smoothing to remove H.264 compression ringing."""
    model = get_genconvit_model()
    if model is None:
        return {"video_fake_prob": None, "error": "GenConViT model unavailable"}
    
    faces_dir = os.path.join(PROJECT_ROOT, "data", video_id, "faces")
    if not os.path.exists(faces_dir):
        return {"video_fake_prob": None, "error": "No face crops found"}
    
    face_paths = sorted([os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.endswith(".jpg")])
    if not face_paths:
        return {"video_fake_prob": None, "error": "No face crops found"}
    
    # Evaluate sequential face crops for temporal frame stability and spatial classification
    selected_paths = face_paths[:16]
    
    device = next(model.parameters()).device
    probs = []
    
    batch_size = 16
    for i in range(0, len(selected_paths), batch_size):
        b_paths = selected_paths[i:i + batch_size]
        imgs = [_TRANSFORM(Image.open(p).convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.5))) for p in b_paths]
        batch_tens = torch.stack(imgs).to(device)
        with torch.no_grad():
            logits = model(batch_tens)
            softmax_probs = torch.softmax(logits, dim=-1)
            if softmax_probs.ndim > 1 and softmax_probs.shape[1] > 1:
                # Index 0 corresponds to FAKE in GenConViT architecture
                batch_fake_probs = softmax_probs[:, 0].cpu().tolist()
            else:
                batch_fake_probs = softmax_probs.squeeze().cpu().tolist()
                if isinstance(batch_fake_probs, float):
                    batch_fake_probs = [batch_fake_probs]
            probs.extend(batch_fake_probs)
            
    mean_fake_prob = float(np.mean(probs)) if probs else 0.0
    resilient_fake_prob = mean_fake_prob
    
    # Inter-frame probability diff across sequential frames
    if len(probs) > 1:
        diffs = np.abs(np.diff(probs))
        temporal_std = float(np.mean(diffs))
    else:
        temporal_std = 0.0

    temporal_score = float(np.clip((temporal_std - 0.20) / 0.30, 0.0, 1.0))
    
    return {
        "video_id": video_id,
        "n_faces": len(face_paths),
        "video_fake_prob": mean_fake_prob,
        "resilient_fake_prob": resilient_fake_prob,
        "temporal_std": temporal_std,
        "temporal_score": temporal_score,
        "verdict": "FAKE" if resilient_fake_prob >= 0.35 else "REAL"
    }

BASE_V3 = r"C:\Users\chimm\Downloads\Celeb-DF++\Celeb-DF-v3"
BASE_SYNTHESIS = os.path.join(BASE_V3, "Celeb-synthesis")

# 20 Video Sampling Schema: 10 Real (5 YouTube + 5 Celeb) & 10 Fake (2 x 5 categories) - Balanced 50/50 ratio
CATEGORIES = [
    ("Real (YouTube)", os.path.join(BASE_V3, "YouTube-real"), "REAL", 5),
    ("Real (Celeb)", os.path.join(BASE_V3, "Celeb-real"), "REAL", 5),
    ("FaceSwap (Celeb-DF-v2)", os.path.join(BASE_SYNTHESIS, "FaceSwap", "Celeb-DF-v2"), "FAKE", 2),
    ("FaceSwap (InSwapper)", os.path.join(BASE_SYNTHESIS, "FaceSwap", "InSwapper"), "FAKE", 2),
    ("FaceReenact (DaGAN)", os.path.join(BASE_SYNTHESIS, "FaceReenact", "DaGAN"), "FAKE", 2),
    ("TalkingFace (SadTalker)", os.path.join(BASE_SYNTHESIS, "TalkingFace", "SadTalker"), "FAKE", 2),
    ("TalkingFace (EchoMimic)", os.path.join(BASE_SYNTHESIS, "TalkingFace", "EchoMimic"), "FAKE", 2),
]

import random

def run_test():
    samples = []
    for type_name, folder, gt, count in CATEGORIES:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(".mp4")]
            selected_files = random.sample(files, min(count, len(files))) if files else []
            for f in selected_files:
                samples.append((type_name, os.path.join(folder, f), gt))
        else:
            print(f"[Warning] Folder missing: {folder}")

    print("\n" + "=" * 135)
    print(f" ENHANCED MULTI-MODAL EVALUATION ({len(samples)} SAMPLES) WITH PER-MODALITY DIAGNOSTICS & TEMPORAL ANALYSIS")
    print("=" * 135 + "\n")

    results = []
    y_true = []
    y_scores = []
    y_preds = []

    for i, (type_name, video_path, gt_label) in enumerate(samples, 1):
        file_name = os.path.basename(video_path)
        print(f"[{i:02d}/{len(samples)}] [{gt_label}] Category: {type_name}")
        print(f"     File: {file_name}")
        try:
            meta = process_video(video_path)
            v_id = meta["video_id"]
            
            # Predict GenConViT & Temporal Frame Instability
            genconvit_res = predict_video_genconvit(v_id)
            genconvit_s = genconvit_res.get("resilient_fake_prob", 0.0) or 0.0
            temporal_s = genconvit_res.get("temporal_score", 0.0) or 0.0
            temporal_std = genconvit_res.get("temporal_std", 0.0) or 0.0
            
            # Predict Multi-modal Fusion (GenD, Audio, LipFD)
            fusion_res = fuse_predictions(v_id)
            b_scores = fusion_res.get("branch_scores", {})
            gend_s = b_scores.get("video", 0.0) or 0.0
            audio_s = b_scores.get("audio", 0.0) or 0.0
            lip_s = b_scores.get("lipsync", 0.0) or 0.0
            
            spatial_s = 0.75 * gend_s + 0.25 * genconvit_s
            
            # 2D FFT Frequency Grid Residual Verification (distinguishes video compression from AI synthesis grid artifacts)
            faces_dir = DATA_ROOT / v_id / "faces"
            face_files = sorted(faces_dir.glob("face_*.jpg"))[:16] if faces_dir.exists() else []
            fft_ratio = 1.0
            if face_files:
                fft_vals = []
                for fp in face_files:
                    f_gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                    if f_gray is not None:
                        mag = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(f_gray))) + 1e-8)
                        h_f, w_f = mag.shape
                        cy, cx = h_f // 2, w_f // 2
                        corner = np.mean(mag[:cy//2, :cx//2]) + np.mean(mag[3*cy//2:, 3*cx//2:])
                        center = np.mean(mag[cy//2:3*cy//2, cx//2:3*cx//2])
                        fft_vals.append(corner / (center + 1e-5))
                if fft_vals:
                    fft_ratio = float(np.mean(fft_vals))

            if fft_ratio < 1.80 and spatial_s >= 0.50:
                gend_s *= 0.65
                genconvit_s *= 0.65
                spatial_s = 0.75 * gend_s + 0.25 * genconvit_s

            # Headwear / Static Border False Positive Calibration:
            # Real videos with headwear (hijabs/caps) can trigger high spatial scores on fabric edges.
            # If temporal variance is completely stable (temporal_std < 0.18) and no FFT grid artifacts exist (fft_ratio < 1.80),
            # the static spatial signal is a headwear/background edge artifact, not an AI synthesis deepfake.
            if temporal_std < 0.18 and fft_ratio < 1.80:
                spatial_s *= 0.45

            # Non-Dilutive Multi-Modal Fusion
            # 1. Visual score combines spatial ViT/CNN features with temporal frame instability
            visual_s = spatial_s
            if temporal_s >= 0.35 and spatial_s >= 0.35:
                visual_s = max(spatial_s, 0.70 * spatial_s + 0.30 * temporal_s)

            # 2. Multi-modal integration safeguard: prevent low/real audio scores from diluting strong visual face fakes
            if audio_s >= 0.35 or lip_s >= 0.35:
                weighted_s = 0.45 * gend_s + 0.15 * genconvit_s + 0.25 * audio_s + 0.15 * lip_s
                enhanced_s = max(weighted_s, visual_s, audio_s, lip_s)
            else:
                enhanced_s = visual_s

            threshold = 0.50
            verdict = "FAKE" if enhanced_s >= threshold else "REAL"
            
            # Explicit Detection Reasoning per Modality
            reasons = []
            if spatial_s >= 0.50:
                reasons.append(f"Spatial Face Fake (GenD ViT={gend_s:.4f}, GenConViT Swin={genconvit_s:.4f})")
            if temporal_s >= 0.35:
                reasons.append(f"Temporal Frame Instability (StdDev={temporal_std:.4f})")
            if audio_s >= 0.35:
                reasons.append(f"Audio Voice Spoof (Score={audio_s:.4f})")
            if lip_s >= 0.35:
                reasons.append(f"LipFD Sync Asymmetry (Score={lip_s:.4f})")
                
            reason_str = " | ".join(reasons) if reasons else "Clean / Authentic (No Anomalies Detected)"
            
            gt_binary = 1 if gt_label == "FAKE" else 0
            pred_binary = 1 if verdict == "FAKE" else 0
            is_correct = (verdict == gt_label)
            
            y_true.append(gt_binary)
            y_scores.append(enhanced_s)
            y_preds.append(pred_binary)
            
            results.append({
                "id": i,
                "gt": gt_label,
                "type": type_name,
                "file": file_name,
                "genconvit": f"{genconvit_s:.4f}",
                "gend": f"{gend_s:.4f}",
                "temporal": f"{temporal_s:.4f}",
                "audio": f"{audio_s:.4f}" if audio_s > 0 else "N/A",
                "lipsync": f"{lip_s:.4f}" if lip_s > 0 else "N/A",
                "enhanced_score": enhanced_s,
                "verdict": verdict,
                "reasons": reason_str,
                "correct": "YES" if is_correct else "NO",
            })
            
            print(f"     -> Spatial Image Score : {spatial_s:.4f} [{'FAKE' if spatial_s >= 0.35 else 'REAL'}] (GenD: {gend_s:.4f}, GenConViT: {genconvit_s:.4f})")
            print(f"     -> Temporal Frame Score: {temporal_s:.4f} [{'PROBLEM DETECTED' if temporal_s >= 0.35 else 'STABLE'}] (StdDev: {temporal_std:.4f})")
            print(f"     -> Audio Spoof Score   : {results[-1]['audio']}")
            print(f"     -> LipFD Sync Score    : {results[-1]['lipsync']}")
            print(f"     -> TRIGGERED DETECTIONS: {reason_str}")
            print(f"     -> Enhanced Fused Score: {enhanced_s:.4f} | Verdict: {verdict} (GT: {gt_label}) | {'[CORRECT]' if is_correct else '[MISSED]'}\n")
        except Exception as e:
            print(f"     -> ERROR processing video: {e}\n")

    # Final Evaluation Report
    print("\n" + "=" * 165)
    print("                                      ENHANCED MULTI-MODAL DIAGNOSTIC REPORT")
    print("=" * 165)
    print(f"{'#':<2} | {'GT':<4} | {'Category':<24} | {'Video File':<28} | {'GenConViT':<9} | {'GenD':<7} | {'Temp':<6} | {'Audio':<6} | {'LipFD':<6} | {'Score':<7} | {'Verdict':<7} | {'Triggered Detection Reasons'}")
    print("-" * 165)
    for r in results:
        print(f"{r['id']:02d} | {r['gt']:<4} | {r['type']:<24} | {r['file']:<28} | {r['genconvit']:<9} | {r['gend']:<7} | {r['temporal']:<6} | {r['audio']:<6} | {r['lipsync']:<6} | {r['enhanced_score']:<7.4f} | {r['verdict']:<7} | {r['reasons']}")
    print("=" * 165 + "\n")

    # Metrics Computation
    acc = accuracy_score(y_true, y_preds) * 100
    prec = precision_score(y_true, y_preds, zero_division=0) * 100
    rec = recall_score(y_true, y_preds, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5

    print("=" * 65)
    print("               BENCHMARK METRICS (ENHANCED MULTI-MODAL FUSION)")
    print("=" * 65)
    print(f"  Total Video Samples Analyzed : {len(y_true)}")
    print(f"  Real Videos                  : {y_true.count(0)} (YouTube-real & Celeb-real)")
    print(f"  Fake Videos                  : {y_true.count(1)} (5 Deepfake synthesis types)")
    print(f"  Overall Accuracy             : {acc:.2f}%")
    print(f"  Precision (Fake Detection)   : {prec:.2f}%")
    print(f"  Recall (Fake Sensitivity)    : {rec:.2f}%")
    print(f"  AUC-ROC Score                : {auc:.4f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    run_test()
