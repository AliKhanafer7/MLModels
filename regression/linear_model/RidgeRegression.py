import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# This class will perform Ridge regression.
# Ridge is basically the same as OLS except it adds an error to the features in an attempt to reduce bias
# Article for Ridge regression: https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
class Ridge():
    coef_ = np.array([]) # Estimated coefficients
    alpha = 0.1
    def fit(self, X, y):
        transpose_X = np.transpose(X) # Find the transpose of X
        error = np.add(self.alpha,np.identity(transpose_X.shape[0]))
        lhs = np.linalg.inv(np.add(np.dot(transpose_X,X),error)) # Find the inverse of X.T * X and add the error. Problem if X.T * X is not singular (not invertible)
        rhs = np.dot(transpose_X,y) # Find X.T * y.
        
        # Calculates (inv(X.t*X))*(X*y). You get this equation by deriving the squared error equation and setting it to zero, 
        # which will lead to a minimal error since it's a quadratic equation.
        self.coef_ = np.dot(lhs,rhs) 

    def get_params(self):
        return self.coef_

    def predict(self, X):
        return np.dot(X,self.coef_)

    # Calculate the coefficient of determination.
    # The coefficient of determination (r^2) is a value between 0 and 1. 
    # The closer to 1, the better your model and the more correlation there is between you independent and dependent variable
    # Note: With the testing data I get a negative value, this shows me that my model needs more training
    def score(self, predictions, y, sample_weights = None):
        residual_sum_squares = np.sum(np.square(np.subtract(y,predictions)))
        total_sum_squares = np.sum(np.square(np.subtract(y,np.mean(y))))
        return 1 - (residual_sum_squares/total_sum_squares)

    # Calculates the mean squared error. I copied this directly from Imtiaz Ul Hassan article linked above.
    # I included this because it's interesting seeing the difference between how the coefficient of determination
    # and the msee. The MSEE will give you a feeling that your model is well trained while R^2 will give you the feeling
    # that your model is poorly trained
    def msee(self, actual, predicted):
        sum_error = 0.0
        for i in range(len(actual)):
            prediction_error = predicted[i] - actual[i]
            sum_error += (prediction_error ** 2)
            mean_error = sum_error / float(len(actual))
        return mean_error

    # Add column of 1s for the intercept b0
    def add_bias(self,X):
        num_rows, num_columns = X.shape
        ones = np.ones((num_rows,1))
        return np.concatenate((ones, X), axis=1)

# *********** MAIN ***********
lr = Ridge()
df=pd.read_csv('../../data/car_data.csv')
we=df.to_numpy()
we=we[:,0:8]
we=we.astype(np.float64)
df.head()

xtrain=we[:292,1:8]
ytrain=we[:292,0]
xtest=we[292:,1:8]
ytest=we[292:,0]


x_train=lr.add_bias(xtrain)
lr.fit(x_train,ytrain)
predictions = lr.predict(x_train)
train_error=lr.score(predictions,ytrain)
print('Training  Error for Multivariable regression is {}'.format(train_error))

print('\n\n')


x_train=lr.add_bias(xtrain)
x_test=lr.add_bias(xtest)
b=lr.fit(x_train,ytrain)
predictions = lr.predict(x_test)
test_error=lr.score(predictions,ytest)
print('Testing Error for Multivariable regression is {}'.format(test_error))

x_train=lr.add_bias(xtrain)
x_test=lr.add_bias(xtest)
b=lr.fit(x_train,ytrain)
test_predict=lr.predict(x_test)
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for Test data',fontsize=16)
plt.grid(True)
plt.plot(ytest , color='purple')
plt.plot(test_predict , color='red'  )
plt.show()