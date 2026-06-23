# Task 4 — Predicting Insurance Claim Amounts

**DevelopersHub Corporation — Data Science & Analytics Internship**

## Objective
Estimate medical insurance claim amounts (`charges`) from personal attributes using
Linear Regression, and evaluate the model with MAE and RMSE.

## Approach
1. Loaded the Medical Cost Personal Dataset (1,338 records: age, sex, BMI, children,
   smoker, region, charges).
2. Checked for missing values and duplicates (none found; data was already clean).
3. Explored how smoking status, age, and BMI relate to charges via box plots, scatter
   plots, and a correlation heatmap.
4. Built a `scikit-learn` pipeline that one-hot encodes the categorical columns
   (`sex`, `smoker`, `region`) and fits a `LinearRegression` model on an 80/20 train/test
   split.
5. Evaluated the model using MAE, RMSE, and R², and visualized actual vs. predicted charges.

## Results & Insights
- **MAE: $4,177.05 | RMSE: $5,956.34 | R²: 0.807** on the held-out test set.
- **Smoking status is by far the strongest driver of insurance cost** — smokers form a
  visibly separate, much higher-cost band regardless of age or BMI.
- Among smokers specifically, a higher BMI sharply increases predicted charges, suggesting
  smoking and obesity compound each other's cost impact rather than simply adding up.
- The model underestimates the very highest-cost smoker cases, since that relationship is
  non-linear — a tree-based model (e.g., Random Forest or Gradient Boosting) would likely
  improve on this further.
- **Business takeaway:** smoking-cessation and weight-management programs could materially
  reduce a bank/insurer's expected claims exposure.

## Files
- `Task4_Insurance_Regression.ipynb` — full notebook with code, outputs, and explanations
- `insurance.csv` — dataset used
- `images/` — exported chart images
