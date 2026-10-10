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

| Model | Train R² | Test R² |
|-------|----------|---------|
| Linear (gradient descent) | | |
| Linear (analytical) | | |
| Ridge (best λ = ) | | |

## Run it

```bash
git clone https://github.com/EmnaGhorbel-elec/appliances-energy-regression.git
cd appliances-energy-regression
pip install -r requirements.txt
jupyter notebook ML_Notebook.ipynb
```

## Tech

Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn (preprocessing and metrics only).
