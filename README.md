# drXAI: Scaling Time Series Classification via XAI-Driven Data Reduction

## Overview
Temporary repo for "drXAI: Scaling Time Series Classification via
XAI-Driven Data Reduction" submitted to ECML-PKDD 2026 Research Track.

## Abstract:
Explainable AI (XAI) for time series has seen significant al-
gorithmic growth, but its utility in providing measurable performance
gains for downstream tasks remains under-explored. This paper bridges
this gap by introducing drXAI, a novel methodology that repurposes XAI
attribution methods for effective data reduction in Time Series Classifica-
tion (TSC). The core challenge in modern TSC is scalability; state-of-the-
art models, such as Transformers, exhibit quadratic complexity relative
to sequence length and linear complexity relative to the number of chan-
nels. This renders them computationally prohibitive for massive datasets.
drXAI addresses this by using a fast, GPU-accelerated classifier (Hydra)
to generate local attributions (via Feature Ablation). We aggregate these
into global feature importance scores and employ an automated elbow-
cut heuristic to select the most salient features without requiring manual
thresholds. We evaluate our approach on both synthetic and real-world
univariate and multivariate datasets. On synthetic benchmarks, drXAI
successfully recovers ground-truth features where traditional baselines
fail. On real-world data, drXAI achieves 80–90% data reduction while
maintaining classification accuracy comparable to models trained on the
full dataset. Most importantly, we show that drXAI allows resource-
intensive models like ConvTran to scale to datasets that were previously
inaccessible due to memory constraints. Our results show the benefits of
using XAI not just for interpretability, but as a robust tool for feature
selection and scalability in time series analysis. All our code and data
are openly available.

## Code:
Code developed using python 3.11.14. Install libraries in requirements.txt

Executables are:
### train_classifier.py
Trains a classifier on a either original dataset or a reduced dataset.

### get_selection.py
Generates a selection of features based on the global feature importance scores.
Only tested using hydra as "explainer classifier"

### baselines.py:
Used to achieve baseline selections in our experiments. It used a different venv

## Data:
Trained models, datasets, selections and results are available through [zenodo](https://zenodo.org/records/20604796)
