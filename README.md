# Heart Disease Classification

Predicting heart disease from clinical measurements, using the Kaggle [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dataset.

The project tests one question: **does splitting patients into Sex × Age subgroups and training a separate model for each group beat a single model trained on all the data?**

## Repository Structure

```
.
├── heart.csv                    # Dataset (918 patients, 11 features + target)
├── heart_disease.ipynb          # Main notebook: preprocessing, tuning, evaluation
└── data_analysis/
    └── heart_eda.html           # Exploratory data analysis report
```

## Dataset

918 rows, one per patient. The target is `HeartDisease` (1 = heart disease, 0 = normal).

| Feature | Description |
|---|---|
| `Age` | Age in years |
| `Sex` | M / F |
| `ChestPainType` | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl); contains many missing values recorded as 0 |
| `FastingBS` | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| `RestingECG` | Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) |

## Approach

### EDA
The full report is in [data_analysis/heart_eda.html](data_analysis/heart_eda.html). The main findings:

- `ST_Slope`, `ChestPainType` and `ExerciseAngina` separate the classes most strongly. A Flat/Down slope, ASY chest pain, and exercise-induced angina all point to heart disease.
- Low `MaxHR`, high `Oldpeak` and `FastingBS = 1` are also associated with heart disease.
- `RestingBP` looks similar in both classes, and `Cholesterol` has many missing values. Both are left out of the models.
- Most patients are male, and males are much more likely to have heart disease.

### Subgroup architecture
The data is split into four subgroups by Sex and by whether Age is above or below the median (54):

| Subgroup | n | Positive rate |
|---|---|---|
| Male, above median age | 405 | 0.74 |
| Male, below median age | 320 | 0.50 |
| Female, above median age | 93 | 0.39 |
| Female, below median age | 100 | 0.14 |

Each subgroup gets its own model, and Age and Sex are used only to choose the subgroup, not as features. The idea is to stop the model from leaning on "older male → disease" and missing cases in younger women.

### Modeling pipeline
1. **Preprocessing:** `OneHotEncoder` for categorical features and `StandardScaler` for numeric ones, inside an sklearn `Pipeline` so they are re-fit within every CV fold.
2. **Split:** a stratified 80/20 train/test split inside each subgroup.
3. **Hyperparameter tuning:** [Optuna](https://optuna.org/) maximizes mean F1 under `RepeatedStratifiedKFold` (3 splits × 5 repeats). Five models are tuned:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors
   - Support Vector Classifier
   - Gaussian Naive Bayes
4. **Ensembles:** soft voting and stacking (logistic-regression meta-learner), both built on the tuned base models.
5. **Champion selection:** the candidate with the best CV F1 in each subgroup is evaluated once on the held-out test set.

### Full-data benchmark
For comparison, a single model is trained on all the data with Age and Sex as ordinary features. It uses the same five models, search spaces, CV scheme and ensembles. It is also trained and tested on **exactly the same rows** as the subgroup approach (the union of the subgroup train/test splits), so the two can be compared directly.

## Results

On the held-out test set (184 rows):

| Approach | Test F1 | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Per-subgroup (pooled champions) | 0.844 | 0.832 | — |
| Full-data (Age + Sex as features) | 0.834 | 0.821 | 0.88 |

Per subgroup, on the same test rows:

| Subgroup | n_test | Subgroup champion | Subgroup F1 | Full-data F1 |
|---|---|---|---|---|
| Male, above median | 81 | KNN | 0.883 | 0.874 |
| Male, below median | 64 | Random Forest | 0.806 | 0.806 |
| Female, above median | 19 | Random Forest | 0.545 | 0.545 |
| Female, below median | 20 | Soft Voting | 1.000 | 0.857 |

### Takeaways
- On this dataset, subgrouping gives **no meaningful advantage** over a single full-data model.
- All models score within a narrow band (0.87–0.89 CV F1 on the full data), so the choice of algorithm is not the bottleneck. KNN was the full-data champion.
- Voting and stacking did not beat the best single model on the full data.
- The female subgroups have only ~20 test rows each, so their per-group scores are unreliable. Splitting the data spreads the few female positive cases even thinner, and that is the main cost of the subgroup approach.

I still think splitting by age and sex is worthwhile, even though these results don't support it. The main limit is dataset size, and a larger dataset could show the benefit.

### Limitations and next steps
- Results come from a single train/test split. Trying several splits or nested CV would give more reliable estimates.
- `Cholesterol` could be kept by imputing the missing values (e.g. median plus a missing-value indicator) instead of dropping the column.
- Tuning the decision threshold could improve F1.

## Getting Started

```bash
pip install pandas numpy scikit-learn optuna matplotlib jupyter
jupyter notebook heart_disease.ipynb
```

Run the cells from top to bottom. The full-data benchmark reuses the subgroup splits, so it needs the earlier cells to have run first.
