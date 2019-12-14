### CS-433: Machine learning - Project 1 ReadMe - BIZEUL Alice, DERVAUX Juliane, SIERRO Florent
## Team: "BuzzLastyear" 


### run.py
This file is used in order to produce predictions under a .csv format.
In order to customize the submission, first upload the data (training set and test set) stored in a "data" folder, and complete the basic hyperparameters of the model in the fourth cell (lambda and polynomial expansion degree). For each subset, it is possible to fine-tune the following hyperparameters:
<br/>
	- The undefined feature DER_mass_MMC can be set to the median/mean of the feature, or all the samples can be deleted. <br/> 
	- Outliers can be removed while choosing the threshold value, as a multiple of the standard deviation. <br/>
	- Augmentation of features, i.e. XiXj, can be activated or not (where Xi and Xj are almost uncorrelated features). <br/>
	- Balance the class distribution since in the train set it is unbalanced. <br/>

To get the output submission, just run the file, and a 'BuzzLastyear.csv' (i.e. the name of our team) will be created in the data folder.


### implementations.py

As requested, this file contains the 6 following optimization functions: 
**least_squares_GD**(y, tx, initial w, max iters, gamma), **least_squares_SGD**(y, tx, initial w, max iters, gamma), **least_squares**(y, tx), **ridge_regression**(y, tx, lambda ), **logistic_regression**(y, tx, initial w, max iters, gamma), **reg_logistic_regression**(y, tx, lambda , initial w, max iters, gamma)
are in this single python file.


### helpers_*.py

Three helpers files (helpers_data.py, helpers_visualization.py, helpers_optimization.py) contain our own functions. All these files are imported in the run.py
1. helpers_data.py:		Contains the implementations to load the data, split the dataset, explore the features characteristcs and implement feature engineering.
2. helpers_visualization.py: 	Contains the implementations to plot all our figures from our analysis such as histograms, boxplots, etc. 
3. helpers_optimization.py:	Contains the implementations to perform the optimization of the different models, such as cross validation, computation of the loss and gradients, performance assessment, etc. 


## Jupiter Notebooks:

1. Exploratory data analysis.ipynb

This file explore the data set. It computes the number of samples, the number of features, look for NaN values and undefined. 
It plots some histograms to see the distributions of the features, to know if they are continuous or categorical, if they have lots of outliers.
It also plots the correlation matrix of each subset to see the correlation among the features.

2. SS*.ipynb

Here are 4 files (one for each subset) where the optimisation of the hyperparameters (i.e. lambda and degree) of our final model, which uses ridge regression, are done.


3. least-square_ss*.ipynb

Here a 4 files testing the least squares method, to check for singular matrix
