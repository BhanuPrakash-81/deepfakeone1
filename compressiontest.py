import os
import evaluation
from videometadataanalysis import (
    get_video_metadata,
    enrich_results_with_metadata,
    bucket_by_bitrate,
    calibrate_thresholds_per_bucket,
    predict_with_bucket_threshold,
    print_compression_summary,
    analyze_misclassifications
)

from builtestset import TEST_SET_FROM_DATASETS

# 1. Run evaluation on test dataset
test_dataset = TEST_SET_FROM_DATASETS
metrics = evaluation.evaluate_dataset(test_dataset)

# 2. Enrich results with video metadata (calculates bitrate & compression ratio)
enriched = enrich_results_with_metadata(metrics["per_sample_results"])

# 3. Categorize into low/medium/high bitrate buckets
bucketed = bucket_by_bitrate(enriched)

# 4. Output compression ratio & metadata for each video tested
print_compression_summary(bucketed)

# 5. Analyze misclassifications by compression ratio
misclassified = analyze_misclassifications(bucketed)

# 6. Calibrate per-bucket decision thresholds (EER)
thresholds = calibrate_thresholds_per_bucket(bucketed, bucket_key="bitrate_bucket")

# 7. Predict on a target video using its bucket threshold
sample_video = "new_video.mp4"
new_meta = get_video_metadata(sample_video)

if "error" not in new_meta:
    print(f"\n--- Target Video Metadata: {sample_video} ---")
    print(f"  Bitrate: {new_meta.get('bitrate_kbps', 'N/A')} kbps")
    print(f"  Compression Ratio: {new_meta.get('compression_ratio', 'N/A')}:1")
    verdict = predict_with_bucket_threshold(0.85, new_meta, thresholds)
    print(f"Prediction Verdict: {verdict}\n")
else:
    print(f"\nNote: '{sample_video}' not found on disk for single-sample prediction demo.")

