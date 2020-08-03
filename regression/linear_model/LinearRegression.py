import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This class will perform linear regression with Ordinary Least Squares.
# Article for single variable OLS: https://towardsdatascience.com/understanding-the-ols-method-for-simple-linear-regression-e0a4e8f692cc
# Article for multivariable OLS: https://medium.com/analytics-vidhya/multivariate-linear-regression-from-scratch-using-ols-ordinary-least-square-estimator-859646708cd6

class LinearRegression():
    coef_ = np.array([]) # Estimated coefficients for the OLS problem
    
    # Fit using the training data
    def fit(self, X,y):
        transpose_X = np.transpose(X) # Find the transpose of X
        
        lhs = np.linalg.inv(np.dot(transpose_X,X)) # Find the inverse of X.T * X. Problem if X.T * X is not singular (not invertible)
        rhs = np.dot(transpose_X,y) # Find X.T * y.
        
        # Calculates (inv(X.t*X))*(X*y). You get this equation by deriving the squared error equation and setting it to zero, 
        # which will lead to a minimal error since it's a quadratic equation.
        self.coef_ = np.dot(lhs,rhs) 

    def get_params(self, deep=True):
        return self.coef_
    
    # Use the coef_ calculated fitting to predict y
    def predict(self, X):
        return np.dot(X,self.coef_)

    # Calculate the coefficient of determination.
    # The coefficient of determination (r^2) is a value between 0 and 1. 
    # The closer to 1, the better your model and the more correlation there is between you independent and dependent variable
    def score(self, X, y, sample_weights = None):
        predictions = self.predict(X)
        squared_error_regression_line = np.square(np.subtract(y,predictions)).sum()
        total_squared_error = np.square(np.subtract(np.mean(y),predictions)).sum()
        print((squared_error_regression_line/total_squared_error))
        return 1 - (squared_error_regression_line/total_squared_error)

    def set_params(self, **params):
        return None
    
    def add_bias(self,X):
        num_rows, num_columns = X.shape
        ones = np.ones((num_rows,1))
        return np.concatenate((ones, X), axis=1)

# TODO fix testing and double check implementation
# *********** MAIN ***********
lr = LinearRegression()
df = pd.read_csv('../../data/car_data.csv')

# Get first 50 rows for training
X_train = lr.add_bias(df.iloc[:292,1:8]) 
y_train = df.iloc[:292,0]

# Get second 50 rows for training
X_test = lr.add_bias(df.iloc[292:,1:8])
y_test = df.iloc[292:,0]

# Train model
lr.fit(X_train,y_train)

# Predict
predictions = lr.predict(X_test)

# Draw 
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for Test data',fontsize=16)
plt.grid(True)
plt.plot(y_test , color='purple')
plt.plot(predictions , color='red')
plt.show()