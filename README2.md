# Credit Card Fraud Detection — A Cost-Based Approach to Threshold Selection

Most published work on this dataset stops at a performance score. This project continues to the question an operations team actually has to answer: **at what score do we stop a transaction, and what does running the system cost?**

Final system: **80% of fraud caught at 49% precision**, flagging 27 transactions per 10,000 for manual review.

---

## The problem

A fraud detection model does not output "fraud" or "legitimate". It outputs a probability. Converting that probability into a decision requires a threshold, and that threshold is a business decision, not a statistical one — it depends on the relative cost of letting fraud through versus blocking a legitimate customer.

This project builds the model, then makes that cost trade-off explicit.

---

## Data

[ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Anonymised European cardholder transactions over two days in September 2013.

- 284,807 transactions, 492 labelled fraud
- Features `V1`–`V28` are principal components; only `Time` and `Amount` are interpretable
- The data file is not in this repository (144 MB, above GitHub's limit). Download it from the link above into `data/`.

---

## Data integrity: 1,081 duplicate rows

The dataset card does not mention duplicates. There are 1,081 exact duplicate rows, and they are disproportionately fraud: fraud is 0.17% of the dataset but 1.7% of rows involved in duplication — roughly ten times over-represented.

Left in place, identical rows would be split across train and test, letting the model score on examples it had memorised. The inflation would land specifically on the minority class that matters.

**Decision: dropped before splitting.** This costs 19 of 492 fraud cases and buys an honest evaluation.

Working dataset: **283,726 rows, 473 fraud (0.167%)**.

---

## What the exploratory analysis decided

Full analysis and figures are in `01_eda.ipynb`. Four findings shaped the model:

**Accuracy is unusable.** Predicting "legitimate" for every transaction scores 99.83% accuracy and catches zero fraud. → **AUPRC** as the primary metric throughout.

**`Amount` has a long right tail.** Median fraud is 9.82 EUR against 22.00 for legitimate transactions, but mean fraud is *higher* (123.87 vs 88.41), pulled up by a few large cases. → **`RobustScaler`**, since mean and standard deviation are both distorted.

**`Time` encodes row order.** It is an ascending counter, and a model can learn position in the file from it — which will not generalise. Converted to hour-of-day, fraud rate peaks at 02:00 (1.45%, nine times the base rate). → **keep `Hour`, drop `Time`**.

**The signal is strong and directional.** Measured by Cohen's d rather than correlation, since PCA output is uncorrelated by construction and a heatmap shows nothing. V14 reaches 2.23 and V4 1.99, with seven components above 1.3 — well past the 0.8 convention for a large effect. Fraud is negative on V14, V12, V10 and positive on V4, V11, which later serves as a sanity check on the fitted coefficients.

---

## Method

1. Stratified 80/20 split → 226,980 train / 56,746 test (378 / 95 fraud)
2. `RobustScaler` fitted on training data only, applied to test
3. Class imbalance handled with `class_weight='balanced'` rather than resampling
4. Model comparison by 5-fold stratified cross-validation on training data
5. Threshold chosen from cross-validated training predictions
6. Test set used once, for the final numbers only

Steps 2, 4 and 5 all exist to keep the test set uncontaminated. Each is a place where a subtler version of the duplicate-row leakage can enter.

### On resampling

`SMOTE` is the conventional default for this dataset. It is not used here. Class weighting achieves the same objective without generating synthetic minority examples, and avoids the common error of resampling before splitting. Weighting was treated as the baseline; `SMOTE` was not found to be necessary.

---

## Model selection: the difference was noise

| Model | Test AUPRC | CV AUPRC (mean ± std) |
|---|---|---|
| Logistic regression | 0.673 | 0.754 ± 0.027 |
| Histogram gradient boosting | 0.716 | 0.751 ± 0.021 |

On a single test split, gradient boosting appears to win by 0.043.

Under cross-validation the gap disappears — the means differ by 0.004, less than a fifth of one standard deviation, and logistic regression is nominally ahead. The per-fold range *within* either model spans about eight percentage points, wider than the single-split gap that suggested a winner.

**Selected: logistic regression.** With performance indistinguishable, the simpler model wins on interpretability, which has regulatory weight when an automated decision affects a customer, and on serving cost.

The gradient boosting model is retained in the notebook as a documented comparison. Testing a more complex model and finding it adds nothing is a result.

### A note on ROC AUC

The two metrics disagree about which model is better:

| Model | ROC AUC | AUPRC |
|---|---|---|
| Logistic regression | 0.966 | 0.673 |
| Histogram gradient boosting | 0.955 | 0.716 |

ROC AUC exceeds 0.95 for both, implying near-perfect performance. Its false-positive-rate denominator is the full 283,253 legitimate transactions, so even 2,000 false alarms barely register. Precision uses the number of alerts actually raised, so it penalises them directly — which is what a review team experiences.

---

## Threshold selection

### The default threshold is expensive

`predict()` applies a 0.5 cutoff, implicitly assuming that a missed fraud and a blocked legitimate customer cost the same.

On the test set, 0.5 flags 1,497 transactions to catch 83 frauds — **1,414 false alarms**, 5.5% precision, seventeen inconvenienced customers for every real catch.

### Cost model

```
total cost = (missed fraud × average fraud amount) + (false alarms × cost per review)
```

Average fraud amount comes from the training data (115.94 EUR). Cost per review is an operational figure not present in the data; 5 EUR is assumed and its influence tested below.

The threshold was selected by minimising this cost over cross-validated predictions on the **training** set, giving **0.965**.

### Honest selection cost 2.4%

Choosing the threshold on the test set instead would have given 0.995 and a cost of 3,269 EUR. The honestly-chosen 0.965 costs 3,348 EUR on the same test set — a gap of 79 EUR, or 2.4%.

The cost curve is flat across a wide band of thresholds, so methodological rigour cost almost nothing in performance. What it removed was a number that would not have survived deployment.

### The recommendation is robust to the cost assumption

| Cost per review (EUR) | Optimal threshold |
|---|---|
| 1 | 0.940 |
| 2 | 0.940 |
| 5 | 0.965 |
| 10 | 0.980 |
| 20 | 0.995 |
| 50 | 0.995 |

A fiftyfold change in the assumption moves the threshold by 0.055. The recommendation does not hinge on getting the review cost right.

---

## Results

Logistic regression at threshold 0.965, test set, evaluated once:

| | |
|---|---|
| Transactions flagged | 155 of 56,746 |
| Fraud caught | 76 of 95 |
| False alarms | 79 |
| Fraud missed | 19 |
| Precision | 0.49 |
| Recall | 0.80 |

In operational terms: four of every five frauds are caught, one of every two alerts is genuine, and the review queue receives 27 transactions per 10,000 processed.

---

## Limitations

**The test set contains 95 fraud cases.** Four cases move recall by four percentage points. Small differences between configurations should not be over-read — this is precisely what the model comparison above demonstrated.

**Two days of data.** The overnight fraud pattern is suggestive, not evidence of a stable daily cycle.

**Recall ceiling.** Even at a 0.5 threshold, 12 fraud cases are never flagged. This model will not reach them.

**Review cost is assumed, not measured.** The sensitivity analysis bounds the impact of that assumption rather than removing it.

**The threshold grid stops at 0.995.** For review costs of 20 EUR and above the optimum sits at the grid edge, so the true optimum for those cases may lie above the range searched.

**Features are anonymised.** No domain feature engineering is possible, and no fraud case can be explained in business terms — only in terms of principal components.

---

## Repository

```
01_eda.ipynb          Exploratory analysis and the decisions it produced
02_modelling.ipynb    Split, model comparison, threshold selection, results
requirements.txt      Pinned dependencies
```

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download `creditcard.csv` from the Kaggle link above into `data/`, then run the notebooks in order.

---

Mutaib AlOtaibi
