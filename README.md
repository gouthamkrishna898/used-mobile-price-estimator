# Used & Refurbished Mobile Price Estimator

## Project Overview
This project uses machine learning to estimate the resale price of used and refurbished smartphones based on their specifications, age, usage, and condition.

## Features
- Smartphone brand and operating system selection
- RAM and storage capacity
- Screen size and battery capacity
- Main and selfie camera specifications
- 5G support
- Phone age and days used
- Estimated resale price with a bar chart

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Machine Learning Models
The project compares three regression algorithms:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Model performance is evaluated using MAE, RMSE, and R² scores. The best-performing model is saved as `price_model.pkl`.

## Dataset
The project uses a synthetically generated dataset for educational and demonstration purposes. Predictions are not verified real-world market prices.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python make_dataset.py
```

Train and evaluate the models:

```bash
python model.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

## Disclaimer
The predicted price is an estimate and is not a guaranteed resale or market price.
