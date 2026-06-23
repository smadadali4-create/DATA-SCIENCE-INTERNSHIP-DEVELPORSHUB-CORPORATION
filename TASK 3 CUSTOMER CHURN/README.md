# Task 3 — Customer Churn Prediction (Bank Customers)

**DevelopersHub Corporation — Data Science & Analytics Internship**

## Objective
Identify bank customers who are likely to churn (leave the bank) using a supervised
classification model, and understand which features drive that decision.

## Approach
1. Loaded the Churn Modelling dataset (10,000 bank customers, 14 columns).
2. Dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`).
3. Encoded categorical features: `Gender` via label encoding, `Geography` via one-hot
   encoding.
4. Explored churn patterns across geography, gender, age, and balance via count plots and
   box plots — found Germany has a notably higher churn rate, and churners tend to be older.
5. Scaled numeric features and trained a `RandomForestClassifier` (`class_weight="balanced"`
   to account for the ~80/20 class imbalance) on an 80/20 train/test split.
6. Evaluated with accuracy, a classification report, and a confusion matrix; extracted and
   plotted feature importances.

## Results & Insights
- **Accuracy: 86.1%** on the test set, with 77% precision / 45% recall on the churned
  (minority) class.
- **Age, balance, and number of products held** are the most important predictors of churn,
  followed by estimated salary and credit score.
- **Geography matters**: customers in Germany churn at a noticeably higher rate than those
  in France or Spain.
- **Business takeaway:** retention efforts should be focused on older, single-product
  customers — especially in Germany — before they show signs of leaving. Recall on the
  churn class (45%) shows there's room to improve detection further, e.g. with
  hyperparameter tuning, SMOTE oversampling, or gradient boosting models.

## Files
- `Task3_Churn_Classification.ipynb` — full notebook with code, outputs, and explanations
- `Churn_Modelling.csv` — dataset used
- `images/` — exported chart images
