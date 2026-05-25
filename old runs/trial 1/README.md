## Trial 1 Archive

This folder preserves the lightweight record of the first tabular ADP/STL study.

Contents:
- `csv/all_models_loss_by_task.csv`: every archived model summary row that was available before cleanup
- `csv/tasks/*.csv`: per-task slices of the same data
- `csv/best_per_task.csv`: lowest-loss model per task in the archive
- `graphs/`: saved loss-vs-parameter plots that existed before cleanup

Heavy artifacts were intentionally removed from the active repo after this archive was created:
- checkpoints
- candidate directories
- metadata JSON
- training logs
- run-progress files
- intermediate plots inside run folders
