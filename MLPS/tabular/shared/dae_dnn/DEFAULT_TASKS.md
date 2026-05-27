# DNN Defaults (STL + ADP)

This folder implements plain MLP baselines for the active non-vision task set.
All tasks use real public benchmark datasets and can be executed in STL or ADP
mode.

Default task mapping

1) Prediction / regression
   - Dataset: YearPredictionMSD
   - Input: full feature vector -> Output: 1-d float
   - Loss: MSE
   - Metric: MSE

2) Representation / embeddings
   - Dataset: Covertype
   - Input: 54-d feature vector -> Output: 7-class logits + embeddings
   - Loss: cross-entropy
   - Metric: kNN accuracy on embeddings (k=5)

3) Autoencoding / reconstruction
   - Dataset: Covertype
   - Input: 54-d feature vector -> Output: 54-d reconstruction
   - Loss: MSE
   - Metric: MSE

4) Generation (single-model)
   - Dataset: Covertype
   - Input: fixed noise vector paired with each feature vector -> Output:
     54-d reconstruction
   - Loss: MSE
   - Metric: MSE

5) Denoising / restoration
   - Dataset: Covertype
   - Input: noisy features -> Output: clean features
   - Loss: MSE
   - Metric: MSE

6) Anomaly / novelty detection
   - Dataset: Covertype (normal class vs other classes)
   - Input: 54-d feature vector -> Output: reconstruction
   - Loss: MSE
   - Metric: AUROC based on reconstruction error

7) Simulation / digital twin
   - Dataset: California Housing derived transform
   - Input: feature vector -> Output: synthetic target transform
   - Loss: MSE
   - Metric: MSE

Notes
- All tasks use the same plain MLP backbone.
- STL = fixed architecture; ADP = width/depth adaptive.
- Logging: per-run `training_log.txt`, `training_stats.csv`, and plots.
