from sklearn.linear_model import LinearRegression
from joblib import dump
import numpy as np

# Fake training data
X = np.array([[100], [500], [1000], [1500], [2000]])
y = np.array([100000, 200000, 300000, 400000, 500000])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
dump(model, 'linear_regression_model.joblib')