# MLModels
MLModels is my take at recreating (or somewhat so) the models provided by [scikit-learn](https://scikit-learn.org/stable/)
## Prerequisite
1. Python 3
2. Pipenv
## Installation
This project uses [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) to manage all its packages. Run the following command from the root directory to ensure you've got all the right packages
```
pipenv install
```
Then access your newly created virtual environment and run all the code from within
```
pipenv shell
```
## Usage
The code is seperated into three parts:
1. Regression models (Ex. Linear Regression)
2. Classification models (Ex. Ridge Regression and classification)
3. Clustering models (Ex. k-means)

Running any of the scripts included in these folders will graph the difference between the predicted data and the actual data.

### Demo
This demo will walk you through running my implementation of Oridnary Least Squares (OLS).

1. Navigate to the right directory 
``` 
$ cd regression/linear_model/ 
```
2. Run my script
```
$ python LinearRegression.py
```
3. At this point, two things should happen
   1. A graph should appear showing the difference between the predicted results and the actual value
   2. The model's score should be logged in the consol

## Contributing
Contributions and comments are accepted and welcomed. All new changes must be done in a seperate branch and a pull request must be submitted to be reviewed before it can be merged to master

You'll also notice a significant amount of comments. My goal is to be able to look back at this project and understand the thought process that went into every line of code. Those comments are a way of doing so.