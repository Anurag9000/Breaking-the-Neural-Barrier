Centralized home for all plain fully connected MLP code in this repo.

Layout:
- `tabular/` for tabular and mixed benchmark-task MLPs
- `image/` for image MLPs and image DAE MLPs
- `text/` for text MLPs
- `graph/` for graph inputs processed by fully connected MLPs
- `examples/` for the legacy den-style and multi-head MLP examples

Each domain has `supervised/`, `unsupervised/`, and `semi_supervised/` folders. Cross-paradigm shared files live in `shared/` only when one file serves multiple paradigms.
