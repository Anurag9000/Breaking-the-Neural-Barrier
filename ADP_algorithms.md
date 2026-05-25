# ADP Policy

This repository retains implementations for several historical ADP variants, but
the supported operational policy is now intentionally narrow.

## Supported ADP modes

Only these two ADP modes should be used for new experiments:

- `alt_width`
- `width_to_depth`

## Disabled-for-use legacy modes

These implementations remain in the codebase for archival and comparative
reference only, but they are no longer supported by the active CLIs and default
experiment runners:

- `width_only`
- `depth_only`
- `depth_to_width`
- `alt_depth`

## Practical consequence

- New suite runners and shared launchers default to `alt_width` and
  `width_to_depth` only.
- Many per-model scripts still contain the legacy search logic internally, but
  their argument parsers no longer advertise or accept those legacy modes.
- Historical results that used the disabled modes are preserved; this policy
  applies only to future runs.

## Rationale

This is an empirical policy decision: across the maintained experiment suites,
`alt_width` and `width_to_depth` were kept because they consistently produced
the only results worth continuing to benchmark. The other modes remain in-tree
to preserve reproducibility of older work, not because they should still be run.
