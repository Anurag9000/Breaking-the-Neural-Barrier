# ADP Repository Guide & Structure

This repository contains a large collection of Deep Learning models across various domains (Autoencoders, CNNs, Diffusion, Graph, LSTM/RNN, Transformers, VAEs) refactored to support **Adaptive Deep Processing (ADP)**.

## 1. Repository Structure Overview

The repository is organized by domain, then learning paradigm (Supervised, Self-Supervised, Unsupervised), and finally Models.

**Key File Types:**
*   `Original_Model.py`: The original model implementation (e.g., `cnn_res_net_v_1.py`).
*   `Original_Model_adp_width_to_depth.py`: The **ADP Wrapper**. This single file contains **ALL** ADP modes for that model.

**Example Path:**
`CNN/Supervised/Models/cnn_res_net_v_1_adp_width_to_depth.py`

## 2. ADP Modes & Running Instructions

Although the file is named `...width_to_depth.py`, it supports **all** ADP algorithms defined in the specification. You select the algorithm using the `--adp-mode` command-line argument.

### Common usage:
```bash
python <path_to_adp_wrapper.py> --adp-mode <MODE_NAME> [other options]
```

### Available Modes:

| Mode Name | Description | CLI Argument |
| :--- | :--- | :--- |
| **Width Only** | Expands only the width (neurons/channels) of layers. | `--adp-mode width_only` |
| **Depth Only** | Expands only the depth (layers) of the network. | `--adp-mode depth_only` |
| **Width to Depth** | Expands width until saturation, then adds depth. | `--adp-mode width_to_depth` |
| **Depth to Width** | Expands depth until saturation, then adds width. | `--adp-mode depth_to_width` |
| **Alt Width** | Alternates expanding width and training. | `--adp-mode alt_width` |
| **Alt Depth** | Alternates expanding depth and training. | `--adp-mode alt_depth` |

### Example Commands:

**1. Run `width_only` ADP on a ResNet:**
```bash
python CNN/Supervised/Models/cnn_res_net_v_1_adp_width_to_depth.py --adp-mode width_only --max-epochs 100 --start-width 64
```

**2. Run `depth_to_width` ADP on a Transformer:**
```bash
python Autoencoder/Supervised/Models/ae_transformer_stl_adp_width_to_depth.py --adp-mode depth_to_width --start-depth 2
```

## 3. Full List of ADP Models

(The complete list of all 500+ ADP wrapper files is available in `adp_scaffold.txt` in the root directory.)

## 4. Key Implementation Details
*   **Forward-Only Expansion**: The algorithms use a forward-only approach. Expansions are tried, and if they fail, the search continues without rolling back immediately (except for `patience_es` restoration).
*   **Global Best**: The best model found across the entire search history is saved and restored at the very end.
*   **Snapshots**: Architecture configurations (width, depth, etc.) and weights are strictly snapshotted to ensure correct restoration.

