# Real Benchmark Audit

This document records the final state of the repository-wide dataset migration.
The executable training code no longer uses synthetic, toy, fake, or demo-only
data paths.

## Verified

- No executable training file in the repository matches `FakeData`,
  `ToyDataset`, or `SyntheticDataset`.
- Model-file demo blocks that used random tensors for smoke tests were removed.
- `sitecustomize.py` blocks toy/demo entrypoints and keeps legacy loaders on
  real benchmark paths.
- The DAE/DNN suite runs on real tabular benchmarks only.
- The autoencoder, diffusion, transformer, RNN, CNN, graph, and VAE families
  were audited so their runnable paths use real public datasets or benchmark
  loaders rather than synthetic data.

## Current policy

- CIFAR10/CIFAR100 are allowed where a vision baseline is intentionally part of
  the benchmark family.
- Non-vision families use task-appropriate public benchmarks such as Covertype,
  YearPredictionMSD, California Housing, AG News, LibriSpeech, SpeechCommands,
  CoNLL-2003, UCF101, VOC, COCO, FlyingChairs, and related standard datasets.
- Future changes should preserve this benchmark-only policy for executable
  training code.

## Notes

- Historical placeholder comments and documentation references may still exist
  in a few places, but they do not affect training behavior.
- Generated artifacts and run outputs should stay out of version control.
