# DAE/DNN Search Suite Report

- Run root: `DAE/DNN/results/search_suite_20260516`
- Git commit: `62a1ab08faeece374c7fd13f1bcff8cb61ea35eb`
- Tasks completed: `['prediction']`

## Task: prediction
- Overall winner: `search` via `random` at `0.0008057808154262602`

| Method | Search arch | Search val | STL arch | STL val | Winner |
|---|---|---:|---|---:|---|
| grid | in=8 hidden=[512, 512, 512, 512, 512] out=1 bn=True | 0.0008414287003688514 | in=8 hidden=[512, 512, 512, 512, 512] out=1 bn=True | 0.2780097424983978 | search |
| random | in=8 hidden=[512, 512] out=1 bn=True | 0.0008057808154262602 | in=8 hidden=[512, 512] out=1 bn=True | 0.3298645615577698 | search |
| bayes | in=8 hidden=[512, 512, 512, 512, 512] out=1 bn=True | 0.0010810543317347765 | in=8 hidden=[512, 512, 512, 512, 512] out=1 bn=True | 0.2843468487262726 | search |
| nas | in=8 hidden=[17, 17, 17, 17] out=1 bn=True | 0.0171149093657732 | in=8 hidden=[17, 17, 17, 17] out=1 bn=True | 0.3568846881389618 | search |
