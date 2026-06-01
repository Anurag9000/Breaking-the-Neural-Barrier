# Transformer MLP ADP Contract

Transformer ADP wrappers delegate to `utils/transformer_mlp_adp.py`.

The adapter applies the current staged MLP search semantics to transformer
feed-forward blocks only. It does not expand token embeddings, attention
projections, attention head counts, classifier heads, or entire transformer
blocks.

`model_mlp_mixer` and `vae_base` remain outside this delegation boundary:
they are not transformer architectures even though legacy organization placed
their files under the transformer directory.

## Synchronized FFN Growth

Every recognized FFN/MLP block in a model is mutated atomically. This includes
encoder and decoder FFNs in sequence-to-sequence models.

- Width growth increments the narrowest internal FFN layer first.
- Depth growth is allowed only when each FFN's internal widths are uniform.
- Depth growth appends a width-10 internal FFN layer.
- Subsequent staged width growth warms the new width-10 layer up to the
  existing width before another depth mutation is allowed.
- Overlapping FFN weights are preserved. Attention weights are untouched.
- If no supported transformer FFNs are found, the adapter raises an error
  instead of silently adapting a projection or helper module.

## Search Modes

The wrappers expose the same six modes as the MLP runner:

- `width_only`
- `depth_only`
- `alt_width`
- `alt_depth`
- `width_to_depth`
- `depth_to_width`

The shared controller tracks separate width and depth patience, width-stage
minimum-improvement margin, monotonic global best validation loss, staged
warmup state, candidate index, and completion state.

## Validation

Run:

```bash
.venv/bin/python -m unittest test_transformer_mlp_adp.py -v
.venv/bin/python -m py_compile utils/transformer_mlp_adp.py test_transformer_mlp_adp.py $(find TRANSFORMERS -path '*Models/*.py' -type f | sort)
```
