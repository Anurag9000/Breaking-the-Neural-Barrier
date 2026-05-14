# Real Benchmark Migration Checklist

This checklist tracks remaining runnable entrypoints that still need to be moved
off synthetic, toy, or small-benchmark defaults.

## Completed

- `DAE/DNN/tasks.py` now uses real benchmark datasets instead of synthetic/MNIST placeholders.
- `CNN/utils/train_supervised_single_model.py` supports `imagefolder`.
- `utils/cnn_data.py` has ImageNet normalization and transforms.
- `Autoencoder/Self-Supervised/Models/Masking_Missing-Data_Families.py` defaults to `imagefolder`.
- `Autoencoder/Self-Supervised/Runs/run_Masking_Missing-Data.py` defaults to `imagefolder`.
- `Transformer/Supervised/Runs/run_transformer_seq_2_seq.py` now requires real TSV inputs.
- Explicit `Transformer/Supervised/*toy*` runners/models were deleted.
- `sitecustomize.py` blocks toy/demo scripts and redirects legacy CIFAR/STL/SVHN loaders to folder-backed real data.
- `sitecustomize.py` also injects sibling `Models/` directories for entrypoints under `Runs/` so model imports resolve without per-file path hacks.
- Saved run outputs under `logs/` were deleted.
- `Diffusion/Self-Supervised/Runs/runs/_common_real_image.py` and the `runs/_common_train*.py` shims now route the self-supervised diffusion leaf runners to real folder-backed image data.
- Standalone `Diffusion/Self-Supervised/Runs/run_*_cifar_10.py` diffusion scripts now use the same real-image loader path.
- `Diffusion/Supervised/Runs/runs/_common_real_image.py` now backs the image-only supervised diffusion runners, and the following leaf runners were converted off synthetic dummies: `run_convnext_diff.py`, `run_resnet_diff.py`, `run_vit_diff.py`, `run_unet_baseline.py`, `run_ddpm_denoise_unet.py`, `run_fmd_unet.py`, `run_hybrid_diffae_unet.py`, `run_lsd_unet.py`, and `run_lsd_dit.py`.
- The supervised conditional image runners `run_ddpm_classcond_unet.py`, `run_rfm_classcond_unet.py`, and `run_ddpm_segcond_unet.py` now derive their class count from real `ImageFolder` data instead of hardcoding CIFAR-10.
- `Diffusion/Supervised/Runs/run_ddpm_classcond_dit.py`, `run_ddpm_inpaint_unet.py`, `run_ddpm_sr_unet.py`, and `run_ddpm_vpred_unet.py` were rewritten to use the real-image loader path instead of CIFAR/dummy data.
- The unconditional supervised image runners `run_dit.py`, `run_swin_diff.py`, `run_unetpp.py`, and `run_tiny_mobile_diff.py` were also rewired to the same real-image loader path.
- The remaining SDE runners `run_vp_sde_supervised.py`, `run_pf_ode_supervised.py`, `run_sdsn_score_unet.py`, and `run_ssm_score_unet.py` now use the real-image loader path as well.
- `Diffusion/Supervised/Runs/run_ddpm_classcond_unet_1.py`, `run_ddpm_classify_map_unet.py`, `run_ddpm_ctrl_depth_unet.py`, `run_ddpm_ctrl_sketch_unet.py`, and `run_lsd_cond_vec_unet.py` were also switched from dummy batches to real folder-backed image data.
- `Diffusion/Supervised/Runs/run_ddpm_flow_unet.py` now uses the published FlyingChairs optical-flow benchmark.
- `Diffusion/Supervised/Runs/run_ddpm_regress_unet.py` now uses PASCAL VOC segmentation masks as a real regression target.
- `Diffusion/Supervised/Runs/run_ddpm_sem2img_unet.py` now uses PASCAL VOC segmentation labels as one-hot semantic conditioning.
- `Diffusion/Supervised/Runs/run_ddpm_textcond_unet.py` now uses COCO Captions with hashed text embeddings.
- `Diffusion/Supervised/Runs/run_ddpm_audio_spec_unet.py` now uses LibriSpeech spectrograms.
- `Diffusion/Supervised/Runs/run_ddpm_timeseries_1d_unet.py` now uses LibriSpeech waveforms as 1D sequence data.
- `Diffusion/Supervised/Runs/run_ddpm_vfi_unet3d.py` now uses UCF101 clips for video frame interpolation.
- `Diffusion/Supervised/Runs/run_ddpm_xdomain_unet.py` now uses paired real image folders for cross-domain translation experiments.
- `Diffusion/Supervised/Runs/run_unet1d_diff.py` now uses LibriSpeech waveforms as a real 1D benchmark.
- `Diffusion/Supervised/Runs/run_unet3d_diff.py` now uses UCF101 clips as a real 3D benchmark.
- `Diffusion/Supervised/Runs/run_hybrid_ct_diff.py` now uses paired real image folders instead of random CT-shaped noise.
- `Diffusion/Supervised/Runs/run_ddpm_pose_unet.py` now uses COCO keypoints instead of synthetic pose heatmaps.
- `Diffusion/Supervised/Runs/run_ddpm_med3d_unet.py` now uses UCF101 video volumes instead of random medical-shaped noise.
- `LSTM, GRU, RNN/Supervised/Runs/*` leaf runners now use real benchmark loaders rather than synthetic or CIFAR/MNIST placeholder data.
- `Transformer/Supervised/Runs/` image-wrapper runners such as `run_vit.py`, `run_deit.py`, `run_swin.py`, `run_rest.py`, `run_cait.py`, `run_nat.py`, `run_dat.py`, `run_maxvit.py`, `run_pvt_v_1.py`, `run_pvt_v_2.py`, `run_crossvit.py`, `run_levit.py`, `run_cswin.py`, `run_coatnet.py`, `run_efficientformer.py`, `run_focal.py`, `run_nest.py`, `run_mvitv_2.py`, `run_mobilevit_v_2.py`, `run_t_2_t_vit.py`, `run_crossformer.py`, `run_swin_v_2.py`, `run_vit_lite.py`, `run_beit_sup.py`, and `run_eva_sup.py` were switched from CIFAR loaders to real folder-backed image data.
- `Transformer/Supervised/Runs/run_bert_encoder.py`, `run_causal_transformer.py`, and `run_conformer_audio_cls.py` now use AG News / SpeechCommands instead of synthetic text/audio.
- `Transformer/Supervised/Runs/run_bigbird_text.py`, `run_fnet_text.py`, and `run_longformer_text.py` now use AG News instead of synthetic TSV or toy token data.
- `Transformer/Self-Supervised/Models/model_cpc_text_transformer_adp_width_to_depth.py` now uses AG News text batches instead of synthetic integer sequences.
- `Transformer/Self-Supervised/Runs/run_mae_vit.py`, `run_colorization_vit.py`, and `run_rotnet_vit.py` now use folder-backed real image data instead of CIFAR defaults.
- `Transformer/Self-Supervised/Runs/run_simclr_vit.py`, `run_vicreg_vit.py`, and `run_barlow_vit.py` now use folder-backed real image data instead of CIFAR defaults.
- `Transformer/Self-Supervised/Runs/run_simsiam_vit.py`, `run_swav_vit.py`, `run_maskfeat_vit.py`, `run_igpt_patch.py`, `run_ae_vit.py`, `run_dae_vit.py`, `run_jigsaw_vit.py`, `run_cae_vit.py`, `run_simmim_vit.py`, and `run_beit_token_vit.py` now use real folder-backed image data instead of CIFAR defaults.
- `Autoencoder/Self-Supervised/Models/Masking_Missing-Data_Families.py`, `Classical_Reconstruction__ae_ssl_core.py`, `Classical_Reconstruction__ae_ssl_core_adp_width_to_depth.py`, `ae_seq_adp_width_to_depth.py`, and `ae_predictive_adp_width_to_depth.py` now use real folder-backed image data instead of CIFAR/STL placeholders.
- `Autoencoder/Supervised/Runs/_common_real_image.py` now backs the supervised AE loaders, and `run_ae_stl_py_train_eval_undercomplete_autoencoder.py`, `run_ae_tied_stl_py_train_eval_tied_weights_autoencoder.py`, and `run_ae_denoise_stl_py_train_eval_denoising_ae_with_gaussian_noise.py` now use it instead of CIFAR loaders.
- `Autoencoder/Supervised/Runs/_common_real_image.py` now backs the remaining supervised AE runners and the matching `Models/*_adp_width_to_depth.py` smoke-test loaders, so the supervised AE family no longer depends on CIFAR placeholders.
- `DAE/Supervised/Runs/run_dae_tokenmask_text_semisup_stl.py` now uses AG News semi-supervised text data instead of toy token streams.
- `DAE/Supervised/Models/dae_tcn_seq_sup_stl_adp_width_to_depth.py` now uses the FordA time-series benchmark instead of synthetic Gaussian sequences.
- `DAE/Supervised/Models/dae_speech_spec_sup_stl_adp_width_to_depth.py` now uses SpeechCommands MFCC features with real speech labels instead of toy spectrograms.
- `DNN FOR NLP (TRY TO BEAT TRANSFORMERS DNN WITH MY ADP ALGORITHM CAN LEARN RELATIONAL MODELING DATA)/Supervised/Runs/run_nlp_ssl_stl.py` and the matching `nlp_ssl_*_adp_width_to_depth.py` files now use AG News contrastive text loaders instead of placeholder text.
- `DNN FOR NLP (TRY TO BEAT TRANSFORMERS DNN WITH MY ADP ALGORITHM CAN LEARN RELATIONAL MODELING DATA)/Unsupervised/Models/nlp_ae_common_adp_width_to_depth.py` now uses AG News CSVs and real text autoencoder batches instead of the synthetic placeholder tuple.
- Selected `LSTM, GRU, RNN/Self-Supervised/Models/*_adp_width_to_depth.py` wrappers now use FordA time-series loaders instead of dummy image-shaped tensors: `rnn_dc_adp_width_to_depth.py`, `rnn_apc_adp_width_to_depth.py`, `lstm_seq_autoencoder_adp_width_to_depth.py`, `rnn_barlow_adp_width_to_depth.py`, `rnn_vrae_adp_width_to_depth.py`, and `rnn_seqclr_adp_width_to_depth.py`.
- `LSTM, GRU, RNN/Supervised/Models/lstm_tagger_vanilla_adp_width_to_depth.py` now uses CoNLL-2003 POS tags instead of synthetic tagging data.
- `LSTM, GRU, RNN/Self-Supervised/Runs/run_rnn_dc.py`, `run_rnn_mtm.py`, `run_rnn_seqclr.py`, and `run_rnn_tid.py` now use FordA-backed real sequence data instead of toy generators.
- `LSTM, GRU, RNN/Self-Supervised/Runs/run_rnn_pl.py`, `run_rnn_triplet.py`, `run_rnn_apc.py`, `run_rnn_jigsaw_py_trainer_for_segment_jigsaw_permutation_recovery.py`, `run_rnn_vrae.py`, `run_rnn_fbc.py`, `run_rnn_sbd.py`, `run_rnn_barlow.py`, `run_rnn_ar.py`, `run_rnn_cpc.py`, `run_rnn_dsae.py`, `run_rnn_tgp_py_trainer_for_temporal_gap_speed_prediction.py`, `run_rnn_speed_py_trainer_for_speed_direction_classification.py`, `run_rnn_tov.py`, and `run_rnn_npair_py_trainer_for_n_pair_info_nce_single_gru.py` now use FordA-backed real sequence data instead of toy/generated sequences.

## Still Remaining

### LSTM, GRU, RNN

The supervised leaf runners are complete. The remaining work in this family is
the self-supervised sequence runners below.

#### Self-Supervised leaf runners using toy/generated sequences

None remain in the run-entrypoint layer.

### Transformer
The supervised image-wrapper cluster below has been moved to real folder-backed
image data in this pass. The remaining transformer work is split into the
legacy explicit CIFAR wrappers, the text/audio/seq2seq supervised runners, and
the self-supervised family.

#### Remaining supervised wrappers

- `Transformer/Supervised/Runs/run_convnext_vit_cifar_10.py`
- `Transformer/Supervised/Runs/run_deit_cifar_10.py`
- `Transformer/Supervised/Runs/run_mobilevit_cifar_10.py`
- `Transformer/Supervised/Runs/run_mlp_mixer_cifar_10.py`
- `Transformer/Supervised/Runs/run_perceiver_cifar_10.py`
- `Transformer/Supervised/Runs/run_pvt_cifar_10.py`
- `Transformer/Supervised/Runs/run_swin_cifar_10.py`
- `Transformer/Supervised/Runs/run_vit_cifar_10.py`
- `Transformer/Supervised/Runs/run_bart_style_seq_2_seq.py`
- `Transformer/Supervised/Runs/run_t_5_style_seq_2_seq.py`

#### Remaining self-supervised transformer runners

None remain in the run-entrypoint layer.

### Diffusion

The self-supervised diffusion leaf runners are now covered by the real-image
loader package in `Diffusion/Self-Supervised/Runs/runs/`; the remaining
diffusion work is the supervised family above, plus non-image modalities that
need their own public benchmark adapters.

### Autoencoder

- `Autoencoder/Supervised` is now fully migrated off CIFAR placeholders.
- `Autoencoder/Self-Supervised/Models/*_adp_width_to_depth.py` files were audited and the remaining real-image loaders are already in place.

### CNN

- `CNN/Supervised/Runs/run_cnn_*.py` leaf runners still default to CIFAR10/CIFAR100.
- `CNN/Self-Supervised/Runs/run_*.py` leaf runners still have CIFAR-based defaults or small-benchmark assumptions.
- `CNN/ADP_ResNet/run_resnet_stl.py`
- `CNN/ADP_ResNet/run_resnet_stl_grid.py`
- `CNN/Supervised/Models/*_adp_width_to_depth.py` smoke-test loaders were converted from dummy `torch.randn` batches to real CIFAR loaders.
- `CNN/Self-Supervised/Models/*_adp_width_to_depth.py` smoke-test loaders were converted from dummy `torch.randn` batches to real CIFAR loaders.

### Graph

- `Graph/Supervised/Runs/run_*.py`
- `Graph/Self-Supervised/Runs/run_*.py`
- `Graph/Supervised/Models/*_adp_width_to_depth.py`
- `Graph/Self-Supervised/Models/*_adp_width_to_depth.py`

### VAE

- `VAE/Supervised/Runs/run_*.py`
- `VAE/Self-Supervised/Runs/run_*.py`
- `VAE/Supervised/Models/*_adp_width_to_depth.py`
- `VAE/Self-Supervised/Models/*_adp_width_to_depth.py`
- `VAE/Supervised/Models/*_adp_width_to_depth.py` smoke-test loaders were converted from dummy `torch.randn` batches to real CIFAR loaders.
- `VAE/Self-Supervised/Models/*_adp_width_to_depth.py` smoke-test loaders were converted from dummy `torch.randn` batches to real CIFAR loaders.

## Current Policy

- Toy/demo scripts are blocked at startup.
- CIFAR/STL/SVHN-style image loaders are redirected to real folder-backed datasets.
- MNIST-family loaders are blocked until a real benchmark adapter exists.
- Any remaining file listed above still needs direct migration to a published benchmark or a family-specific loader.
