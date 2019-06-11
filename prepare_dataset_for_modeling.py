import pandas as pd
import os
import io
import requests
import ssl
from sklearn import preprocessing

def prepare_dataset_for_modeling(dataset_name, n_obs=-1):
    """
    :param dataset_name: name of dataset to be read from the github
    :param n_obs: how many observations to sample (if > 0)
    :return: clean x and y NumPy arrays that are ready for model fitting
    """

    # read in the CSV file in to a Pandas data frame
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context
    github_location = 'https://raw.githubusercontent.com/vaksakalli/datasets/master/'
    dataset_url = github_location + dataset_name
    df = pd.read_csv(io.StringIO(requests.get(dataset_url).content.decode('utf-8')), header=0)

    # drop missing values if there are any
    df = df.dropna()

    # sample a smaller subset if n_obs > 0
    if n_obs > 0:
        df = df.sample(n=n_obs, replace=False, random_state=123)

    y = df.iloc[:, -1]  # last column is y (target feature)
    x = df.iloc[:, :-1]  # everything else is x (set of descriptive features)

    # get all columns that are strings
    # these are assumed to be nominal categorical
    categorical_cols = x.columns[x.dtypes == object].tolist()

    # if a nominal feature has only 2 levels
    # define a single binary variable for it
    for col in categorical_cols:
        n = len(x[col].unique())
        if n == 2:
            x[col] = pd.get_dummies(x[col], drop_first=True)

    # use one-hot-encoding for categorical features with >2 levels
    x = pd.get_dummies(x)

    # scale x between 0 and 1
    x = preprocessing.MinMaxScaler().fit_transform(x)
    # label-encode y
    y = preprocessing.LabelEncoder().fit_transform(y)

    return x, y

# x, y = prepare_dataset_for_modeling('us_census_income_data.csv')

