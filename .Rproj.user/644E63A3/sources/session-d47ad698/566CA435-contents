# High-Dimensional Classification Benchmark: EMNIST Data

## 📌 Project Overview
This project benchmarks **7 supervised learning algorithms** on high-dimensional image data (handwritten letters from the EMNIST dataset). The goal was to overcome the "Curse of Dimensionality" using PCA and identify the most robust classifier for character recognition.

**Data Source:** EMNIST dataset (Subset of handwritten letters D, G, O, Q).

## 🛠️ Tech Stack
* **Language:** R
* **Libraries:** `MASS` (LDA/QDA), `class` (KNN), `nnet` (Multinomial), `xgboost` (Gradient Boosting), `HDclassif` (HDDA).
* **Techniques:** Principal Component Analysis (PCA), Cross-Validation, Scree Test.

## 🔬 Experimental Design: Sample Size Impact
To evaluate the robustness of classifiers under different data regimes, we constructed two training scenarios using balanced datasets:

* **Scenario 1 (High Resource):** Training on **9,600 images** (2,400 per class). Represents an ideal setting with abundant data.
* *Scenario 2 (Low Resource):** Training on **1,600 images** (400 per class). Represents a data-scarce setting to test model stability.
* **Test Set:** A separate set of **1,600 images** was used to evaluate out-of-sample performance for both scenarios.

## 📊 Methodology & Results
We reduced the data from **784 dimensions (pixels)** to **~69 components** (explaining 90% variance)  and evaluated the following models:

| Method | Test Error (Scenario 1) | Key Observation |
|:---|:---:|:---|
| **QDA** | **0.057** | **Best Performer.** Non-linear boundaries fit the data best. |
| **Gradient Boosting** | 0.061 | Strong performance but showed signs of overfitting (0% train error). |
| **KNN (k=1)** | 0.076 | High variance; sensitive to noise in high-dimensional space. |
| **HDDA** | 0.083 | Robust and parsimonious; stable across sample sizes. |
| **Multinom (PC+PC²)** | 0.087 | Overly complex; suffered from convergence issues. |
| **LDA** | 0.109 | Linear boundaries were insufficient for this task. |

## 📈 Visualization
![Error Rate Comparison](output/method_comparison.png)
*(Comparative analysis of Training vs. Test error across all models)*

## 📂 Repository Structure
```text
EMNIST-Classification-Benchmark/
├── data/
│   └── task1.Rdata          # Raw dataset
├── scripts/
│   └── classification.R     # ML modeling script
└── output/
    └── method_comparison.png
```

## 🚀 How to Run
1.  Clone this repository.
2.  Open `scripts/classification.R` in RStudio.
3.  Run the script to reproduce the benchmark results.

---
## 👥 Credits
Based on an assignment of the course Multivariate Statistics at KU Leuven by **Team 13**.
* **Analysis Validation & Code Refactoring:** Executed independent validation of all 7 classifiers and optimized the XGBoost tuning pipeline.