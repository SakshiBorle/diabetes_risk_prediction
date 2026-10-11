# Early Diabetes Risk Prediction

A machine-learning project comparing Logistic Regression, Decision Tree,
and K-Nearest Neighbors (KNN) for symptom-based diabetes-related
classification. A Streamlit app provides an interactive interface for
entering features and viewing a model's predicted class.

> **Important:** This is an educational/research prototype, not a
> medical device. Its Positive/Negative output reflects the dataset's
> labels and must not be used to diagnose or rule out diabetes.

## Project Overview

The project uses the UCI Early Stage Diabetes Risk Prediction dataset,
containing 520 records collected in Bangladesh. Inputs include age,
gender, and symptom indicators.

Workflow: 1. Load and inspect the dataset. 2. Prepare features and
target labels. 3. Train classification models. 4. Evaluate on a held-out
test set. 5. Compare accuracy, precision, recall, and F1-score. 6. Use
saved models in a Streamlit web application.

## Model Comparison

The current notebook evaluation produced these results:

  Model                         Accuracy   Precision   Recall   F1-score
  --------------------------- ---------- ----------- -------- ----------
  Decision Tree                   94.62%      97.53%   94.05%     95.76%
  Logistic Regression             92.31%      95.12%   92.86%     93.98%
  K-Nearest Neighbors (KNN)       84.62%      95.71%   79.76%     87.01%

**Current best-performing model:** Decision Tree, based on the reported
test-set metrics. Confirm that all models were evaluated on the same
held-out test set using consistent preprocessing and without data
leakage before presenting these figures.

## Input Features

The Streamlit app expects these 12 features:

1.  `age`
2.  `gender`
3.  `polyuria`
4.  `polydipsia`
5.  `sudden_weight_loss`
6.  `weakness`
7.  `polyphagia`
8.  `genital_thrush`
9.  `visual_blurring`
10. `irritability`
11. `partial_paresis`
12. `alopecia`

Current app encoding: - Gender: `0 = Female`, `1 = Male` - Symptom
fields: `0 = No`, `1 = Yes`

Ensure these mappings and feature order match the training notebook
exactly.

## Suggested Project Structure

The app is configured to expect this structure. Adjust paths and
filenames to match your actual project.

``` text
Early Stage Diabetes Risk Prediction/
├── app.py
├── <your-notebook>.ipynb
├── data/
│   └── diabetes_data.csv
└── models/
    ├── logistic_regression_model.pkl
    ├── decision_tree_model.pkl
    └── knn_model.pkl
```

The dataset and notebook filenames may differ. The `.pkl` files must
exist at the paths configured in `app.py`.

## Requirements

-   Python 3.10 or later (Python 3.11 is suitable)
-   Streamlit
-   pandas
-   scikit-learn
-   joblib
-   Any additional libraries imported by the notebook or app

## Setup and Run

### 1. Open the project folder

Open a terminal in the directory containing `app.py`.

### 2. Install dependencies

``` bash
python -m pip install streamlit pandas scikit-learn joblib
```

Install any additional packages required by your notebook.

### 3. Check saved models

Confirm that the `models` folder contains the three `.pkl` files and
their names match the `MODEL_FILES` configuration in `app.py`. If the
saved models do not exist, train and save them from your notebook first;
do not create placeholder files.

### 4. Start the app

``` bash
python -m streamlit run app.py
```

Streamlit will print a local URL, commonly `http://localhost:8501`.

## Evaluation Metrics

-   **Accuracy:** Fraction of all test examples classified correctly.
-   **Precision:** Of examples predicted Positive, the fraction that are
    actually Positive.
-   **Recall:** Of actual Positive examples, the fraction correctly
    identified.
-   **F1-score:** Harmonic mean of precision and recall.

Accuracy alone is not enough to select a model. Review the confusion
matrix, class balance, and class-wise metrics.

## Limitations and Responsible Use

-   The dataset is small and was collected in Bangladesh; results may
    not generalize to other populations, including people in India.
-   Labels represent the dataset's Positive/Negative classification, not
    diabetes stage or severity.
-   Model probabilities are not necessarily calibrated medical risk
    estimates.
-   A Negative prediction does not rule out diabetes, and a Positive
    prediction is not a diagnosis.
-   External validation and appropriate clinical testing would be needed
    before real-world medical use.
-   Avoid entering identifiable health information into a public
    deployment without appropriate privacy and security safeguards.

## Future Improvements

-   Validate models using cross-validation and an untouched test set.
-   Review confusion matrices and class-wise metrics.
-   Confirm feature encoding and preprocessing are identical in training
    and deployment.
-   Add automated tests comparing app predictions with notebook
    predictions.
-   Evaluate calibration and external generalization before describing
    outputs as risk estimates.

## Authors and Contributions

Add project authors, supervisor, institution, and contribution details
here.

## License

Add a license if you intend to publish or share this repository. Follow
the dataset's attribution and licensing requirements.
