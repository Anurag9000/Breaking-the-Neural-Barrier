# ADP Policy

This repository retains implementations for several historical ADP variants, but
the supported operational policy is now intentionally narrow.

## Supported ADP modes

Only these four ADP modes should be used for new experiments:

- `alt_width`
- `alt_depth`
- `width_to_depth`
- `depth_to_width`

## Disabled-for-use legacy modes

These implementations remain in the codebase for archival and comparative
reference only, but they are no longer supported by the active CLIs and default
experiment runners:

- `width_only`
- `depth_only`

## Practical consequence

- New suite runners and shared launchers default to the supported four-mode set:
  `alt_width`, `alt_depth`, `width_to_depth`, `depth_to_width`.
- Many per-model scripts still contain the legacy search logic internally, but
  their argument parsers no longer advertise or accept those legacy modes.
- Historical results that used the disabled modes are preserved; this policy
  applies only to future runs.

## Rationale

This is an empirical policy decision: across the maintained experiment suites,
the supported operational set is now `alt_width`, `alt_depth`,
`width_to_depth`, and `depth_to_width`. The `width_only` and `depth_only`
variants remain in-tree to preserve reproducibility of older work, not because
they should still be run.
