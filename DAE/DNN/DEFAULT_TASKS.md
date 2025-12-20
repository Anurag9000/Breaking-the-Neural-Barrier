# DNN Defaults (STL + ADP)

This folder implements plain MLP baselines for the tasks listed in `not_accomplished.md`.
All tasks run with the same training loop and logging, and can be executed in STL or ADP
mode (6 ADP variants). These defaults are intentionally simple so the focus is on
comparative behavior rather than SOTA.

Default task mapping (datasets, I/O, losses, metrics)

1) Prediction / regression
   - Dataset: synthetic regression (sklearn.make_regression)
   - Input: 20-d float vector -> Output: 1-d float
   - Loss: MSE
   - Metric: MSE, MAE

2) Classification / decision
   - Dataset: MNIST
   - Input: 1x28x28 -> Output: 10-class logits
   - Loss: cross-entropy
   - Metric: accuracy

3) Representation / embeddings
   - Dataset: MNIST
   - Input: 1x28x28 -> Output: 10-class logits + embeddings
   - Loss: cross-entropy
   - Metric: kNN accuracy on embeddings (k=5)

4) Autoencoding / reconstruction
   - Dataset: MNIST
   - Input: 1x28x28 -> Output: 1x28x28
   - Loss: MSE
   - Metric: MSE

5) Generation (single-model)
   - Dataset: MNIST
   - Input: fixed noise vector (z=64) paired with each image -> Output: 1x28x28
   - Loss: MSE
   - Metric: MSE

6) Denoising / restoration
   - Dataset: MNIST
   - Input: noisy image -> Output: clean image
   - Loss: MSE
   - Metric: MSE

7) Anomaly / novelty detection
   - Dataset: MNIST (train: digits 0-4, test: digits 5-9)
   - Input: 1x28x28 -> Output: reconstruction
   - Loss: MSE
   - Metric: AUROC based on reconstruction error

8) Sequence / temporal modeling
   - Dataset: synthetic sine wave
   - Input: window of 20 values -> Output: next value
   - Loss: MSE
   - Metric: MSE

9) Inverse problems / scientific modeling
   - Dataset: synthetic linear system y = A x + noise
   - Input: y -> Output: x
   - Loss: MSE
   - Metric: MSE

10) Control / optimization approximation
    - Dataset: synthetic LQR mapping u = -K x
    - Input: state x -> Output: action u
    - Loss: MSE
    - Metric: MSE

11) Clustering / similarity
    - Dataset: MNIST
    - Input: 1x28x28 -> Output: embedding
    - Loss: reconstruction MSE (autoencoder)
    - Metric: k-means NMI on embeddings

12) Compression / encoding
    - Dataset: MNIST
    - Input: 1x28x28 -> Output: reconstruction
    - Loss: MSE
    - Metric: MSE, compression ratio (latent/in_dim)

13) Ranking / scoring
    - Dataset: synthetic scoring (score = w dot x + noise)
    - Input: x -> Output: score
    - Loss: MSE
    - Metric: pairwise ranking accuracy

14) Multimodal mapping / fusion
    - Dataset: MNIST + scalar metadata (digit parity)
    - Input: [flattened image ; parity scalar] -> Output: 10-class logits
    - Loss: cross-entropy
    - Metric: accuracy

15) Self-supervised / weakly-supervised pretext
    - Dataset: MNIST rotations (0/90/180/270)
    - Input: rotated image -> Output: 4-class rotation label
    - Loss: cross-entropy
    - Metric: accuracy

16) Simulation / digital twin
    - Dataset: linear dynamical system x_{t+1} = A x_t + B u_t
    - Input: [x_t ; u_t] -> Output: x_{t+1}
    - Loss: MSE
    - Metric: MSE

17) Edge / real-time inference
    - Dataset: MNIST
    - Input: 1x28x28 -> Output: 10-class logits
    - Loss: cross-entropy
    - Metric: accuracy + throughput (samples/sec) in val
    - Constraint: default max_width=32 for ADP

18) Misc modeling (noise, error, residual, bias, normalization, transfer)
    - Dataset: synthetic regression with bias
    - Input: x -> Output: residual (y - baseline linear)
    - Loss: MSE
    - Metric: MSE

Notes
- All tasks use the same plain MLP backbone (no CNNs).
- STL = fixed architecture; ADP = width/depth adaptive (6 modes).
- Logging: per-run `training_log.txt`, `training_stats.csv`, and plots.
