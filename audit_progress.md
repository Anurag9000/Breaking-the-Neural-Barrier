# Audit Progress Tracker

## Status Legend
| Symbol | Meaning |
|:---:|---|
| [ ] | Pending |
| [/] | In Progress |
| [x] | Verified & Fixed |
| [!] | Issues Found (Blocked) |

## Files to Audit

### LSTM, GRU, RNN
- [x] Supervised\Runs\run_rnn_vanilla.py (Verified: Standard Run Script)
- [x] Supervised\Models\rnn_vanilla.py (Verified: Model Definition)
- [x] Supervised\Models\rnn_vanilla_adp_width_to_depth.py (Fixed: Duplicate functions, Rebuild logic, Dummy data)
- [x] Supervised\Models\rnn_vanilla_ln.py (Verified: Model Definition)
- [x] Supervised\Models\rnn_vanilla_ln_adp_width_to_depth.py (Fixed: Wrong ModelClass, Rebuild logic, Duplicate functions)
- [x] Supervised\Models\rnn_vanilla_re_lu.py (Verified: Model Definition)
- [x] Supervised\Models\rnn_vanilla_re_lu_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, dummy data)
- [x] Supervised\Models\lstm_cls_vanilla.py (Verified: Model Definition)
- [x] Supervised\Models\lstm_cls_vanilla_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, switched to NLP synthetic data)
- [x] Supervised\Models\lstm_tagger_vanilla.py (Verified: Model Definition)
- [x] Supervised\Models\lstm_tagger_vanilla_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, added tagging data)
- [x] Supervised\Runs\run_lstm_tagger_vanilla.py (Verified: Synthetic data generation and training loop)
- [x] Supervised\Models\rnn_gru.py (Verified: Model Definition)
- [x] Supervised\Models\rnn_gru_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, dummy data)
- [x] Supervised\Models\rnn_gru_bi.py (Verified: Model Definition)
- [x] Supervised\Models\rnn_gru_bi_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, dummy data)
- [x] Supervised\Models\cnn_vgg.py (Verified: Added cfg introspection logic)
- [x] Supervised\Models\cnn_vgg_adp_width_to_depth.py (Fixed: Implemented parametric VGG cfg generation, fixed rebuild logic and data)
- [x] Supervised\Runs\run_cnn_vgg.py (Verified: Standard training loop)
- [x] Supervised\Models\model_causal_transformer.py (Verified: Added init args storage)
- [x] Supervised\Models\model_causal_transformer_adp_width_to_depth.py (Fixed: Rebuild logic, corrected ModelClass import, data loading)
- [x] Supervised\Runs\run_causal_transformer.py (Verified: Training loop)
- [x] Self-Supervised\Models\model_cpc_text_transformer.py (Verified: Added init args storage)
- [x] Self-Supervised\Models\model_cpc_text_transformer_adp_width_to_depth.py (Fixed: Rebuild logic, duplicate functions, synthetic data)

