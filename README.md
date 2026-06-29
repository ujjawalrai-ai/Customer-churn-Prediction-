# Customer Churn Prediction

## Overview

This project predicts whether a customer is likely to leave a telecom service company using Machine Learning. The objective is to help businesses identify customers who are at risk of churning and take preventive actions.

---

## Features

* Data preprocessing and cleaning
* Missing value handling
* Conversion of object data types to numerical values
* Feature encoding
* Feature scaling using StandardScaler
* Logistic Regression model training
* Customer churn prediction dashboard using Streamlit
* Interactive user interface for real-time predictions

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## Data Preprocessing

The following preprocessing techniques were applied:

* Handled missing values in the dataset.
* Converted object data types to numeric values using `pd.to_numeric()`.
* Encoded categorical variables into numerical values.
* Standardized feature values using `StandardScaler`.
* Split the dataset into training and testing sets.

---

## Machine Learning Model

**Algorithm Used:** Logistic Regression

The model was trained on preprocessed customer data to predict whether a customer will churn.

---

## Model Performance

* Accuracy Score: **70%**

The model achieved an accuracy of approximately **0.70**, indicating good predictive performance for customer churn classification.

---

## Project Structure

Customer-Churn-Prediction/

├── churn_prediction.ipynb

├── churnpred.py

├── 7_logistic_model.pkl

├── screenshots/
  └── Result.png
  └── Dashboard.png

├── requirements.txt

└── README.md

---

## Dashboard

The Streamlit dashboard allows users to:

* Enter customer information.
* Predict customer churn in real time.
* View prediction results through an interactive interface.

---

## How to Run

1. Clone the repository.
2. Install the required packages:

pip install -r requirements.txt

3. Run the Streamlit application:

streamlit run churnpred.py

---

## Future Improvements

* Improve model accuracy using advanced algorithms.
* Add feature importance visualization.
* Deploy the application online.
* Compare multiple classification models.

---

## Author

Ujjawal Rai
