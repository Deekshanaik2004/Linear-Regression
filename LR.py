import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Step 2: Select features (X) and target (y)
X = df[['rm']].values  # average number of rooms per dwelling
y = df['medv'].values  # median value of owner-occupied homes in $1000's

# Step 3: Add bias (intercept) column to X
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance

# Step 4: Compute theta using Normal Equation
# θ = (XᵀX)^(-1) Xᵀy
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Step 5: Make predictions
y_pred = X_b.dot(theta_best)

# Step 6: Plot
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Number of Rooms (rm)')
plt.ylabel('Median Value ($1000s)')
plt.legend()
plt.title('Linear Regression from Scratch')
plt.show()

# Step 7: Print theta
print("Intercept:", theta_best[0])
print("Slope:", theta_best[1])
