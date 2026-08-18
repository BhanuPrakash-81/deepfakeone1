import evaluation
from videometadataanalysis import (
    get_video_metadata,
    enrich_results_with_metadata,
    bucket_by_bitrate,
    calibrate_thresholds_per_bucket,
    predict_with_bucket_threshold
)

from builtestset import TEST_SET_FROM_DATASETS

# 1. Run evaluation on test dataset
test_dataset = TEST_SET_FROM_DATASETS
metrics = evaluation.evaluate_dataset(test_dataset)

# 2. Enrich results with video metadata
enriched = enrich_results_with_metadata(metrics["per_sample_results"])

# 3. Categorize into low/medium/high bitrate buckets
bucketed = bucket_by_bitrate(enriched)

# 4. Calibrate per-bucket decision thresholds (EER)
thresholds = calibrate_thresholds_per_bucket(bucketed, bucket_key="bitrate_bucket")

# 5. Predict on a new video using its bucket threshold
new_meta = get_video_metadata("new_video.mp4")
verdict = predict_with_bucket_threshold(0.85, new_meta, thresholds)
print(f"Prediction: {verdict}")
