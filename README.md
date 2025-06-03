## Linear Regression from Scratch using NumPy

This project demonstrates how to implement **Linear Regression** from scratch using **only NumPy**, without relying on machine learning libraries like Scikit-learn. A real-world dataset (Boston Housing) is used to predict housing prices based on the average number of rooms per dwelling.

---

##  Overview

-  **Linear Regression** is a fundamental algorithm in machine learning used for predicting a continuous output.
-  This implementation uses the **Normal Equation** approach to compute optimal model parameters.
-  A simple visualization is provided to compare actual data points with the predicted regression line.

---

## What You’ll Learn

- How to use **NumPy** for matrix operations in ML
- How to implement a linear regression model from scratch
- How to apply it on a **real-world dataset**
- How to visualize results using **Matplotlib**

---

## Dataset

- **Source:** [Boston Housing Dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv)
- **Target Variable:** `medv` – Median value of owner-occupied homes in $1000's
- **Feature Used:** `rm` – Average number of rooms per dwelling

---

##  How It Works

1. **Load Data** using `pandas`
2. **Extract Features** (X) and **Target** (y)
3. **Add Bias Term** to the feature matrix
4. **Apply Normal Equation** to compute optimal `theta`
5. **Predict Values** using learned parameters
6. **Visualize** the original data and regression line

---

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

Install them using:

```bash
pip install numpy pandas matplotlib
