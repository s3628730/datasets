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

####################################################

## arrhythmia.csv
Arrhythmia dataset from UCI. The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify it in one of the 16 groups. 

**Problem type:** Multinomial classification (16 classes).

**Size:** 279 features, 452 instances.

## breast_cancer_wisconsin.csv
The Breast Cancer Wisconsin dataset from UCI. The dataset contains cell biopsy results for cancer screening. The objective is to predict whether a given observation is benign or malignant.

**Problem type:** Binary classification (B: benign, M: malignant).

**Size:** 30 features, 569 instances.

## diamonds.csv
The diamonds dataset from the ggplot2 R library. The dataset contains information on diamonds including carat (numeric), clarity (ordinal), cut (ordinal), and color (ordinal). The objective is to predict the price of a diamond.

**Problem type:** Regression.

**Size:** 10 features, 53940 instances.

## sonar.csv
The sonar dataset from UCI. The dataset contains sonar signal information for cylinder-shaped objects obtained from a variety of different aspect angles. The objective is to predict whether a given observation is a rock or a metal.

**Problem type:** Binary classification (1: metal, -1: rock).

**Size:** 60 features, 208 instances.

## vehicle.csv 
The vehicle dataset from UCI. The dataset contains silhouette information as one of four types of vehicle, using a set of features extracted from the silhouette where the vehicles are viewed various angles. The objective is to predict the type of vehicle.

**Problem type:** Multinomial classification (4 classes: Opel, Saab, bus, van).

**Size:** 18 features, 846 instances.

