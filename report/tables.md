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

| Run | Count | Avg (ms) | Median (ms) | p95 (ms) | Min (ms) | Max (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| latest | 500 | 1387.57 | 1306.96 | 3144.89 | 8.91 | 3643.91 |
| gbt_test | 3000 | 1067.31 | 861.95 | 2935.24 | 2.81 | 4968.65 |
| lr_test | 3000 | 996.33 | 792.47 | 3143.01 | 4.18 | 4658.79 |
| smoke | 500 | 1387.57 | 1306.96 | 3144.89 | 8.91 | 3643.91 |
