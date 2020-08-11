import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This class will perform linear regression with Ordinary Least Squares.
# Article for single variable OLS: https://towardsdatascience.com/understanding-the-ols-method-for-simple-linear-regression-e0a4e8f692cc
# Article for multivariable OLS: https://medium.com/analytics-vidhya/multivariate-linear-regression-from-scratch-using-ols-ordinary-least-square-estimator-859646708cd6

class LinearRegression():
    coef_ = np.array([]) # Estimated coefficients
    
    # Fit using the training data
    def fit(self, X,y):
        X = np.array(X)
        y = np.array(y)
        transpose_X = np.transpose(X) # Find the transpose of X
        lhs = np.linalg.inv(np.dot(transpose_X,X)) # Find the inverse of X.T * X. Problem if X.T * X is not singular (not invertible)
        rhs = np.dot(transpose_X,y) # Find X.T * y.
        
        # Calculates (inv(X.t*X))*(X*y). You get this equation by deriving the squared error equation and setting it to zero, 
        # which will lead to a minimal error since it's a quadratic equation.

        self.coef_ = np.dot(lhs,rhs) 

    def get_params(self):
        return self.coef_
    
    # Use the coef_ calculated fitting to predict y
    def predict(self, X):
        result = np.dot(X,self.coef_)
        return np.dot(X,self.coef_)

    # Calculate the coefficient of determination.
    # The coefficient of determination (r^2) is a value between 0 and 1. 
    # The closer to 1, the better your model and the more correlation there is between you independent and dependent variable
    # Note: With the testing data I get a negative value, this shows me that my model needs more training
    def score(self, predictions, y):
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
    
    def remove_categorical_cols(self, df):
        types = df.dtypes
        cols = df.columns
        result = []
        for i in range(len(types)):
            if(types.iloc[i] in ['int64','float64']):
                result = result + [cols[i]]

        return df[result]
            

# *********** MAIN ***********
# *********** House data testing ***********
lr = LinearRegression()
training_df = pd.read_csv('../../data/house_data.csv')
data = lr.remove_categorical_cols(training_df)
columns_means = data.mean()
data = data.fillna(columns_means)

# Prepare training and testing data
x_train = lr.add_bias(data.iloc[:1000,1:6]) # Use first 1000 rows for training
y_train = data.iloc[:1000,37:]
x_test = lr.add_bias(data.iloc[1001:,1:6]) # Use rest for testing
y_test = data.iloc[1001:,37:]

# Test using training data
lr.fit(x_train,y_train)
predictions = lr.predict(x_train)
r_squared = lr.score(predictions,y_train)
print('Coefficient of determination for Multivariable regression on house training data is {}'.format(r_squared.iloc[0]))

# Draw comparison between predicted data and real value
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for house training data',fontsize=16)
plt.xlabel('House Id')
plt.ylabel('House price')
plt.grid(True)
y_line, = plt.plot(y_train , color='purple')
y_line.set_label('Actual')
predictions_line, = plt.plot(predictions , color='red'  )
predictions_line.set_label('Predictions')
plt.legend()
plt.show()

# Test using testing data
predictions = lr.predict(x_test)
train_error = lr.score(predictions,y_test)
print('Training Error for Multivariable regression on house testing data is {}'.format(train_error.iloc[0]))

# Draw comparison between predicted data and real value
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for house test data',fontsize=16)
plt.grid(True)
y_line, = plt.plot(y_test.to_numpy(), color='purple')
y_line.set_label('Actual')
predictions_line, = plt.plot(predictions , color='red')
predictions_line.set_label('Predictions')
plt.legend()
plt.show()

print('\n\n')

# *********** Car data testing ***********

# Prepare data
lr = LinearRegression()
training_df = pd.read_csv('../../data/car_data.csv')
data = lr.remove_categorical_cols(training_df)
columns_means = data.mean()
data = data.fillna(columns_means)

xtrain=data.iloc[:292,1:8]
ytrain=data.iloc[:292,0]
xtest=data.iloc[292:,1:8]
ytest=data.iloc[292:,0]

# Test using training data
x_train=lr.add_bias(xtrain)
lr.fit(x_train,ytrain)
predictions = lr.predict(x_train)
r_squared=lr.score(predictions,ytrain)
print('Coefficient of determination for Multivariable regression on the car training data is {}'.format(r_squared))

# Draw comparison between predicted data and real value
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for Test data',fontsize=16)
plt.grid(True)
y_line, =plt.plot(ytrain.to_numpy() , color='purple')
y_line.set_label('Actual')
predictions_line, = plt.plot(predictions , color='red'  )
predictions_line.set_label('Predictions')
plt.legend()
plt.show()

# Test using testing data
x_test=lr.add_bias(xtest)
predictions = lr.predict(x_test)
r_squared=lr.score(predictions,ytest)
print('Coefficient of determination for Multivariable regression on car testing data is {}'.format(r_squared))

# Draw comparison between predicted data and real value
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for Test data',fontsize=16)
plt.grid(True)
y_line, = plt.plot(ytest.to_numpy() , color='purple')
y_line.set_label('Actual')
predictions_line, = plt.plot(predictions , color='red'  )
predictions_line.set_label('Predictions')
plt.legend()
plt.show()