# DNN Defaults (STL + ADP)

This folder implements plain MLP baselines for 15 non-vision tasks.
All tasks use real public benchmark datasets and can be executed in STL or ADP
mode (6 ADP variants).

Default task mapping

1) Prediction / regression
   - Dataset: YearPredictionMSD
   - Input: 90-d float vector -> Output: 1-d float
   - Loss: MSE
   - Metric: MSE, pairwise accuracy

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

7) Inverse problems / scientific modeling
   - Dataset: California Housing
   - Input: observed features -> Output: held-out features
   - Loss: MSE
   - Metric: MSE

8) Control / optimization approximation
   - Dataset: California Housing
   - Input: conditioned features -> Output: target-side features
   - Loss: MSE
   - Metric: MSE

9) Clustering / similarity
   - Dataset: Covertype
   - Input: 54-d feature vector -> Output: embedding
   - Loss: reconstruction MSE (autoencoder)
   - Metric: k-means NMI on embeddings

10) Compression / encoding
    - Dataset: Covertype
    - Input: 54-d feature vector -> Output: reconstruction
    - Loss: MSE
    - Metric: MSE, compression ratio (latent/in_dim)

11) Ranking / scoring
    - Dataset: YearPredictionMSD
    - Input: x -> Output: score
    - Loss: MSE
    - Metric: pairwise ranking accuracy

12) Multimodal mapping / fusion
    - Dataset: Covertype + scalar metadata (class parity)
    - Input: [feature vector ; parity scalar] -> Output: 7-class logits
    - Loss: cross-entropy
    - Metric: accuracy

13) Self-supervised / weakly-supervised pretext
    - Dataset: Covertype feature permutation prediction
    - Input: permuted feature vector -> Output: permutation ID
    - Loss: cross-entropy
    - Metric: accuracy

14) Simulation / digital twin
    - Dataset: California Housing derived transform
    - Input: feature vector -> Output: synthetic target transform
    - Loss: MSE
    - Metric: MSE

15) Misc modeling (noise, error, residual, bias, normalization, transfer)
    - Dataset: California Housing
    - Input: x -> Output: residual (y - baseline linear)
    - Loss: MSE
    - Metric: MSE

Notes
- All tasks use the same plain MLP backbone.
- STL = fixed architecture; ADP = width/depth adaptive (6 modes).
- Logging: per-run `training_log.txt`, `training_stats.csv`, and plots.
