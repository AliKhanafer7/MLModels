import numpy as np
# This class will perform linear regression with Ordinary Least Squares. The most pertinent information I got while researching this fitting technique came from
# https://towardsdatascience.com/understanding-the-ols-method-for-simple-linear-regression-e0a4e8f692cc

class LinearRegression():
    coef_ = [] # Estimated coefficients for the linear regression problem. coef_[0] holds the value of the slope while coef_[1] holds the value of the y intercept
    def fit(self, X,y):
        self.coef_ += [self.get_slope(X,y)]
        self.coef_ += [self.get_intercept(X,y,self.coef_[0])]

    # Calculates the slope of the linear regression problem. The equation used is one that will minimize the value of the slope and give us the smallest error.
    # We get this equation when deriving the equation of the squared error and setting its value to 0. See the above article for further explanation.
    def get_slope(self,X, y):
        avg_X = sum(X)/len(X)
        avg_y = sum(y)/len(y)
        numerator = 0
        denomenator = 0
        for i in range(len(X)):
            numerator += (X[i] - avg_X)*(y[i] - avg_y)
        for i in range(len(X)):
            denomenator += (X[i] - avg_X)*(X[i] - avg_X)
        return numerator/denomenator

    # Calculates the intercept of the linear regression problem. The equation used is one that will minimize the value of the intercept and give us the smallest error.
    # We get this equation when deriving the equation of the squared error and setting its value to 0. See the above article for further explanation.
    def get_intercept(self, X, y, m):
        avg_X = sum(X)/len(X)
        avg_y = sum(y)/len(y)
        return avg_y - m*avg_X

    def get_params(self, deep=True):
        return self.coef_
    
    def predict(self, X):
        predictions = []
        for xi in X:
            predictions += [self.coef_[0]*xi + self.coef_[1]]
        return predictions 

    def score(self, X, y, sample_weights = None):
        return None

    def set_params(self, **params):
        return None

lr = LinearRegression()
lr.fit([0,1,2],[0,1,2])
print(lr.coef_)