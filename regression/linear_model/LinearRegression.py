class LinearRegression(fit_intercept = True, normalize = False, copy_X = True, n_jobs = None):

    def fit(self, X,y):
        '''
        Fit linear model
            
            Parameters:
                X (array): Training data
                y (array): Target values. Will be cast to X's dtype if necessary
            Returns:
                self: returns an instance of self
        
        '''
        return None
    
    def get_params(self, deep=True):
        '''
        Get parameters for this estimator

            Parameters:
                deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.
            Returns:
                Parameter names mapped to their values.
        '''
        return None
    
    def predict(self, X):
        '''
        Predict using the linear model.

            Paramters:
                X (array): Samples
            Returns:
                C: Returns predicted values
        '''
        return None

    def score(self, X, y, sample_weights = None):
        return None

    def set_params(self, **params):
        return None

