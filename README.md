# Road Traffic Flow Forecasting in the City of Porto

This repository contains the end-to-end Machine Learning solution for predicting urban traffic congestion levels in the city of Porto. The project was developed as part of the **Data and Machine Learning** course within the **Master's in Informatics Engineering (Mestrado em Engenharia Informática)** at the **University of Minho** (January 2026).

Following the **CRISP-DM** methodology, this solution addresses severe class imbalances and overlapping decision boundaries to deliver highly reliable traffic state predictions. It was evaluated on the **ML@DAA** Kaggle competition platform, achieving a competitive performance that aligns with the top 4 elite positions in generalization capability.

---

## 📌 Project Overview & Problem Definition

Urban mobility management is a critical challenge for modern smart cities. This project approaches traffic forecasting as an **ordinal multi-class classification** task. The target variable (`average_speed_diff`), which represents the difference between theoretical free-flow speed and actual observed speed, is categorized into five distinct levels:
* **None**: Absence of congestion (fluid traffic).
* **Low**: Light traffic, minimal impact on average speed.
* **Medium**: Moderate traffic, indicating the onset of road saturation.
* **High**: Intense traffic, causing a significant reduction in circulation speed.
* **Very_High**: Severe congestion or complete standstill.

The models utilize heterogeneous data sources combining historical traffic metrics, temporal markers, and real-time weather conditions.

---

## 🛠️ Data Pipeline & Feature Engineering

The dataset contains **6,812 supervised training records** and **1,500 unlabelled test records**. To extract maximum predictive signal while eliminating noise, a robust data preparation pipeline (`Prep. B`) was established:

1.  **Feature Elimination**:
    * `city_name` was dropped due to zero variance (all records belong to "Porto").
    * `average_rain` and `average_precipitation` were removed due to redundant semantics and severe data sparsity (>90% missing values).
2.  **Categorical & Missing Value Treatment**:
    * `luminosity` was ordinal encoded: `DARK` (0), `LOW_LIGHT` (1), and `LIGHT` (2). Missing fields defaulted to `LIGHT`.
    * `average_cloudiness` missing records were filled with a standalone category (`NONE`), treating the absence of measurement as a distinct predictive signature.
    * Numerical variables with formatting errors (`average_temperature`, `average_wind_speed`) were coerced using `pd.to_numeric` and missing slots initialized to `0`, allowing tree algorithms to resolve them naturally.
3.  **Advanced Feature Engineering**:
    * **Day of Week Representation**: Extracted and encoded via One-Hot Encoding to capture distinct weekday vs. weekend behaviors.
    * **Domain Flags**: Created binary indicators for peak traffic strain:
        * `is_rush_hour`: Identifies morning and evening peak intervals (07:00–09:00 and 17:00–19:00).
        * `is_weekend`: Identifies Saturday and Sunday trends.
    * **Cyclical Encoding**: To ensure continuous temporal interpretation by the algorithms, hours and months were mapped into 2D sinusoidal space using sine and cosine transformations:
        $$x_{sin} = sin\left(\frac{2\pi \cdot x}{max(x)}\right), \quad x_{cos} = cos\left(\frac{2\pi \cdot x}{max(x)}\right)$$
4.  **Normalization**:
    * Applied `MinMaxScaler` to bind all features to a `[0, 1]` range, ensuring high-magnitude scales (e.g., atmospheric pressure) do not dominate cyclic and binary vectors.

---

## 🔬 Experimental Matrix & Modeling Insights

The modeling phase moved systematically from standard ensembles to complex specialized architectures. Unsupervised topological analysis (**K-Means with PCA 2D projection**) revealed dense class overlap and non-linear, fragmented boundaries between adjacent classes (`Medium`, `High`, and `Very_High`), guiding our algorithmic choices:

| Exp ID | Strategy / Model Architecture | CV Score (Local) | Kaggle Score | Key Takeaways & Conclusions |
|---|---|---|---|---|
| **1** | Stacking (Baseline) + SMOTE | 0.80931 | 0.80000 | **Failure**: Synthetic samples along dense boundaries introduced severe noise and degraded accuracy. |
| **2** | Stacking + MLPClassifier | 0.81700 | 0.82666 | **Overfitting**: Slight local boost, but deep learning memorized training noise due to a small dataset (~6.8k rows). |
| **4** | Heterogeneous Ensemble (KNN + NB) | 0.81547 | 0.82666 | Outperformed by pure gradient boosting architectures. |
| **5** | Stacking + Random Forest | 0.81430 | — | **Dilution**: Bagging diluted the fine-grained nuances captured by Boosting. Dropped. |
| **6** | Stacking (Reduced Depth) | 0.80828 | — | **Underfitting**: Over-simplifying tree depths led to substantial performance drops. |
| **6b**| Stacking + CatBoost | 0.81415 | 0.82666 | Redundant; increased computational overhead with zero net score gains. |
| **7** | AutoML (PyCaret - ExtraTrees) | — | — | Massive training scores indicated severe, ungeneralizable overfitting. |
| **10**| Stacking + SVM (RBF Kernel) | 0.81533 | 0.82666 | **Failure**: SVM failed to establish clear margins due to high boundary noise. |
| **11**| Stacking with Manual Weights | 0.80931 | 0.81333 | Forcing artificial weight penalization skewed the real probability distributions. |
| **12**| **One-Vs-Rest Specialists (Winning Model)** | **0.81033** | **0.82666** | **Optimal Solution**: Decomposing the 5-class challenge into 5 binary problems dramatically stabilized decision margins. |
| **13**| OVR + CalibratedClassifierCV | 0.81092 | 0.82222 | Improved output log-loss probabilities but slightly lowered classification accuracy. |

---

## 🏆 The Winning Architecture: One-Vs-Rest Specialists

Instead of utilizing a single monolithic multi-class model that struggles to draw global lines across noisy datasets, the winning architecture (**Experiment 12**) uses a **One-Vs-Rest (OVR)** wrapper. This splits the problem into **5 independent binary classifiers**, each trained exclusively to answer a single question (e.g., *"Is this instance High congestion or Not?"*).

### Production Robustness & Operational Safety:
* **High-Volume Precision**: The model correctly isolated 1,997 fluid traffic instances (`None`), minimizing false alarms for traffic management control systems.
* **Fail-Safe Operational Errors**: In the critical extreme congestion tier (`Very_High`), the model successfully flagged 387 cases. Out of 92 missed classifications, the overwhelming majority were labeled as `High`. In an Intelligent Transport System (ITS), this is considered a **safe error** because the system will still trigger congestion mitigation protocols (unlike misclassifying it as `Low`).
* **No Dispersed Errors**: Errors remained strictly confined to adjacent classes (e.g., misclassifying `Medium` as `Low`), proving that the OVR specialists accurately internalized the underlying ordinal structure of urban traffic.

---

## ⚙️ Reproducibility & Hyperparameters

To recreate the optimal architecture and pipeline results, use the configurations detailed below:

* **Meta-Strategy**: One-Vs-Rest (OVR) Classifier Framework
* **Base Estimator**: XGBoost Classifier
* **Core Hyperparameters**:
    * `max_depth`: 5
    * `learning_rate`: 0.08
    * `n_estimators`: 100
    * `random_state`: 42
* **Data Preparation Selected**: `Prep. B` (Cyclical encoding + Domain Flags + No Synthetic Resampling)

---
## 👥 Authors
Eduarda Pereira - PG61516

Gonçalo Ferreira - PG61525

Gonçalo Magalhães - PG61524

Jéssica Cunha - PG60267
