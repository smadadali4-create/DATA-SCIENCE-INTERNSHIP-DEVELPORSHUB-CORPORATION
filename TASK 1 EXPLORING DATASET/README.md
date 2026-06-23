# Task 1 — Exploring and Visualizing the Iris Dataset

**DevelopersHub Corporation — Data Science & Analytics Internship**

## Objective
Understand how to read, summarize, and visualize a simple dataset using pandas, matplotlib,
and seaborn.

## Approach
1. Loaded the Iris dataset (150 samples, 3 species) with pandas and inspected its structure
   using `.shape`, `.columns`, `.head()`, `.info()`, and `.describe()`.
2. Checked for missing values and class balance (confirmed clean and balanced — 50 samples
   per species).
3. Built a scatter plot of petal length vs. petal width (and sepal length vs. sepal width)
   colored by species to look at how well the species separate visually.
4. Built histograms (with KDE overlays) of all four numeric features to study their
   distributions.
5. Built box plots per species for all four features, plus an IQR-based outlier check, to
   examine spread and detect outliers.
6. Added a correlation heatmap as a bonus summary view.

## Results & Insights
- The dataset is clean, balanced, and contains very few outliers.
- **Petal length and petal width** are the most discriminative features — *setosa* separates
  almost perfectly from the other two species on these alone, while *versicolor* and
  *virginica* overlap somewhat more.
- Petal measurements are strongly correlated with each other and with sepal length, hinting
  that fewer than four features could be sufficient for a species classifier.
- Sepal width is the only feature close to a simple, single-peaked (normal-ish) distribution;
  the other three features show multi-modal patterns driven by the three species being mixed
  together.

## Files
- `Task1_Iris_EDA.ipynb` — full notebook with code, outputs, and explanations
- `iris.csv` — dataset used
- `images/` — exported chart images
