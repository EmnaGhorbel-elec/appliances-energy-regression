# Appliances Energy Prediction: Linear & Ridge Regression

Predicting household appliance energy consumption (Wh) from indoor/outdoor
temperature, humidity, and weather measurements. Linear and ridge regression
are implemented **from scratch with NumPy**, with exploratory data analysis
and a chronological train/test split.

> The notebook's text and plot titles are in French.

## Dataset

[UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
(Candanedo et al., 2017): about 19,700 measurements at 10-minute intervals,
Jan–May 2016. The target is `Appliances` (energy use in Wh). The CSV is
included in `data/`.

## What the notebook covers

1. **Exploratory analysis**: time series, target distribution, temperature
   and humidity signals, correlation matrix, scatter plots of the most
   correlated variables.
2. **Linear regression**: gradient descent vs. the analytical solution
   (normal equation), cost curve, effect of the learning rate.
3. **Ridge regression**: analytical solution, search over λ to balance
   train and test R².

Data is split chronologically (70% train / 30% test), and normalization
statistics are computed on the training set only to avoid leakage.

## Results

| Model | Train R² | Test R² | Test RMSE (Wh) |
|-------|----------|---------|----------------|
| Linear (analytical / normal equation) | 0.1718 | 0.0843 | 87.63 |
| Linear (gradient descent, α=0.01) | 0.1699 | 0.0729 | 88.18 |
| **Ridge (best λ ≈ 1351)** | 0.1536 | **0.1029** | **86.74** |
| Top-5 weights (collinear features) | 0.0789 | -0.2768 | 103.48 |
| Top-10 correlated features | 0.1280 | 0.0753 | 88.06 |

**Takeaways**
- Gradient descent converges to essentially the same solution as the normal equation (R² 0.1699 vs 0.1718), which confirms the implementation.
- Ridge gives the best test performance (R² = 0.103, RMSE = 86.7 Wh). Regularization trades a little train fit for better generalization.
- A linear model explains only about 10% of the variance in appliance consumption, so the relationship is strongly nonlinear. Next steps would be lag features, time-of-day features, or nonlinear models (e.g. random forest, gradient boosting).
- Selecting features from the largest |θ| values backfires (negative test R²) because those features are collinear.

## Run it

```bash
git clone https://github.com/EmnaGhorbel-elec/appliances-energy-regression.git
cd appliances-energy-regression
pip install -r requirements.txt
jupyter notebook ML_Notebook.ipynb
```

## Tech

Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn (preprocessing and metrics only).
