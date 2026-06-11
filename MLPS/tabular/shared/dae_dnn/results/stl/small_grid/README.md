# Small STL Width Sweep Archive

This folder restores the historical small / fixed-grid loss-vs-parameters study from git history.

Recovered task family:
- `classification` (legacy label: `representation`)
- `autoencoding`
- `generation`
- `denoising`
- `anomaly`

Not recovered in this small archive:
- `simulation`
- `prediction`

Contents:
- `csv/tasks/*.csv`: archived per-task rows recovered from git history
- `csv/all_models_loss_by_task.csv`: combined task listing recovered from git history
- `csv/best_per_task.csv`: best row per task recovered from git history
- `graphs/*/*.png`: recovered loss-vs-parameters plots when present in history
- `graphs/anomaly/anomaly_loss_vs_params.png`: regenerated from the archived anomaly rows because the original plot blob was not preserved in git history

Notes:
- `classification` is the repo-facing name for the old `representation` label.
- This archive is separate from the live repeat-based ADP W2D suite.
