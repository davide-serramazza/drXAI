# drXAI: Scaling Time Series Classification via XAI-Driven Data Reduction

## Overview
Repository for "Scaling Time Series Classification via XAI-Driven Data Reduction" accepted for AALTD workshop at
ECML-PKDD 2026.

## Abstract:
Explainable AI (XAI) for time series has seen significant algorithmic growth, but its utility in providing measurable
performance gains for downstream tasks remains under-explored. This paper bridges this gap by introducing drXAI, a novel
methodology that repurposes XAI attribution methods for effective data reduction in Time Series Classification (TSC).
The core challenge in modern TSC is scalability; state-of-the-art models, such as Transformers, exhibit quadratic
complexity relative to sequence length and linear complexity relative to the number of channels. This renders them c
omputationally prohibitive for massive datasets. drXAI addresses this by using a fast, GPU-accelerated classifier (Hydra)
to generate local attributions. We aggregate these into global feature importance scores and employ an automated
elbow-cut heuristic to select the most salient features without requiring manual thresholds.
We evaluate our approach on both synthetic and real-world univariate and multivariate datasets. On synthetic benchmarks,
drXAI successfully recovers ground-truth features where traditional baselines fail. On real-world data, drXAI achieves
80–90\% data reduction while maintaining classification accuracy comparable to models trained on the full dataset.
Most importantly, we show that drXAI allows resource-intensive models like ConvTran to scale to datasets that were
previously inaccessible due to memory constraints. Our results show the benefits of using XAI not just for
interpretability, but as a robust tool for feature selection and scalability in time series analysis.
All our code and data are openly available.

## Results

Mean accuracy of each selection (and All Features) for the 4 MTSC real-world datasets and the 3 SOTA classifiers, 
yielding 12 results in total.
![image](https://github.com/davide-serramazza/drXAI/blob/main/images/multi_acc_withRandom.png)

Mean percentage of data saved by each selection for the 4 MTSC datasets.
![image](https://github.com/davide-serramazza/drXAI/blob/main/images/multi_savedData_withRandom.png)

Looking at the two above tables, only our methodology (drXAI) is at the top both for accuracy and data reduction.

<br>

Mean accuracy of each selection (and All Features) for the 5 UTSC  datasets used and the 3 classifiers, 
yielding 15 results in total.
![image](https://github.com/davide-serramazza/drXAI/blob/main/images/uni_acc_withRandom.png)

Mean percentage of data saved of each selection for the 5 UTSC datasets.
![image](https://github.com/davide-serramazza/drXAI/blob/main/images/uni_savedData_withRandom.png)

Looking at the two above tables, only our methodology (drXAI) is at the top both for accuracy and data reduction.


## Code:
Code developed using python 3.11.14. Install libraries in requirements.txt

Executables are:
### train_classifier.py
Trains a classifier on either original dataset or a reduced dataset.

### get_selection.py
Generates a selection of features based on the global feature importance scores.
Only tested using hydra as "explainer classifier"

### baselines.py:
Used to achieve baseline selections in our experiments. It used a different virtual evitomen

## Data:
Models, datasets, selections and results are available here https://zenodo.org/records/19045954