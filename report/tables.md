# Report Tables

## Model comparison

| Model | AUC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| lr | 0.9130 | 0.4751 | 0.9030 | 0.6875 | 0.3630 |
| rf | 0.8855 | 0.1857 | 0.8891 | 0.8321 | 0.1045 |
| gbt | 0.9249 | 0.4791 | 0.9021 | 0.6722 | 0.3721 |

## Tuning metrics

| Model | AUC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| gbt baseline | 0.9249 | 0.4791 | 0.9021 | 0.6722 | 0.3721 |
| gbt tuned | 0.9309 | 0.4997 | 0.9034 | 0.6692 | 0.3987 |
| rf baseline | 0.8855 | 0.1857 | 0.8891 | 0.8321 | 0.1045 |
| rf tuned | 0.9222 | 0.2967 | 0.8949 | 0.7782 | 0.1833 |

## Tuning parameters

| Model | Best params |
| --- | --- |
| gbt | maxDepth=5, maxIter=40, stepSize=0.1, subsamplingRate=1.0 |
| rf | featureSubsetStrategy=sqrt, maxDepth=10, numTrees=100 |

## Streaming latency

| Count | Avg (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- | --- | --- |
| 5000 | 1959.60 | 1252.89 | 6918.08 | 3.12 | 10143.04 |
