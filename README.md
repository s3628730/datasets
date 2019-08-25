# datasets
A collection of public datasets for supervised machine learning research. 
The conventions with the datasets are as follows:
1. All datasets are in CSV format.
2. All datasets have header rows.
3. The target variable is always the last column.
4. All numeric nominal features have been encoded as strings.
5. Any constant columns have been removed. 
6. Any row ID-like columns have been removed.
7. Watch out for any possible missing values in the descriptive features.

A sample Python script named **"prepare_dataset_for_modeling.py"** has also been included for loading these datasets and preparing them for model fitting.

####################################################

Description of these datasets can be found in the "github_datasets_desc" Notebook file:

https://github.com/vaksakalli/datasets/blob/master/github_datasets_desc.ipynb
