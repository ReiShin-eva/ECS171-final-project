# The Gold — ECS171 Final Project

Predicting gold prices using macroeconomic indicators via neural network time-series analysis.

**Team:** Maxwell Chen, Rundong Hu, Logan Tadano, Kushagra Sehgal, Bryan Rahardja 
**Course:** ECS171 — Machine Learning, UC Davis

---

## Overview

This project applies a feedforward neural network to study the relationship between gold prices and three global macroeconomic indicators: the US Dollar Index (DXY), crude oil prices, and inflation rate. A key focus is whether model anomalies (large residuals) correspond to real-world events such as wars or financial crises.

---

## Dataset

**Source:** [Gold Price and Global Economic Indicators (2015–2026) — Kaggle](https://www.kaggle.com/datasets/aminasalamt/gold-price-and-global-economic-indicators15-26)

Download the CSV from Kaggle and place it in the root of this repository before running the notebook. The file is excluded from version control via `.gitignore`.

| Column | Description |
|---|---|
| `Date` | Trading day |
| `Gold_Price_XAU_USD` | Gold closing price per oz (USD) |
| `US_Dollar_Index_DXY` | USD value relative to basket of foreign currencies |
| `Crude_Oil_Price` | Brent/WTI crude oil price |
| `Inflation_Rate_Pct` | Estimated global inflation percentage |

4,137 daily records spanning 2015–2026.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ecs171-gold-prediction.git
cd ecs171-gold-prediction
```

### 2. Download the dataset

Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/aminasalamt/gold-price-and-global-economic-indicators15-26), download the CSV, and place it in the project root:

```
ecs171-gold-prediction/
├── gold_price_data.csv     ← place it here
├── notebook.ipynb
└── README.md
```

### 3. Install dependencies

Run the first cell of the notebook, or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy torch
```

Most dependencies come pre-installed on Google Colab. PyTorch is the largest download (~700MB) and may take a few minutes.

### 4. Run the notebook

Open `notebook.ipynb` in VSCode or Jupyter and run all cells top to bottom.

> **Before submitting:** use `Restart & Run All` to verify the notebook runs cleanly from a fresh kernel state.

---

## Project Structure

```
ecs171-gold-prediction/
├── notebook.ipynb       # Main notebook (all sections)
├── README.md
└── .gitignore
```

---

## Methods

- **Model:** Feedforward neural network (3 inputs → hidden layers → 1 output)
- **Activation:** Leaky ReLU
- **Loss:** Mean Squared Error (MSE)
- **Normalization:** Min-Max scaling (fit on training set only)
- **Split:** 90% train / 10% test, chronological order
- **Hyperparameter tuning:** TimeSeriesSplit cross-validation
