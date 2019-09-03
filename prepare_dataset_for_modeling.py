import pandas as pd
import os
import io
import requests
import ssl
from sklearn import preprocessing
from sklearn.utils import shuffle


def prepare_dataset_for_modeling(dataset_name, n_obs=-1, random_state=999, drop_const_columns=True):
    """
    :param dataset_name: name of dataset to be read from github
    :param n_obs: how many observations to sample (if > 0)
    :param random_state: seed for sampling observations
    :param drop_const_columns: drop constant-value columns (after any sampling) (if True)
    :return: x and y NumPy arrays ready for model fitting
    """

    # read in the CSV file into a Pandas data frame
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
        df = df.sample(n=n_obs, replace=False, random_state=random_state)

    # if needed, you can drop unique-value columns as below
    # ### df = df.loc[:, df.nunique() < df.shape[0]]
    
   if drop_const_columns:
        # drop constant columns
        df = df.loc[:, df.nunique() > 1]

    # last column is y (target feature)
    y = df.iloc[:, -1]
    # everything else is x (set of descriptive features)
    x = df.iloc[:, :-1]

    # get all columns that are strings
    # these are assumed to be nominal categorical
    categorical_cols = x.columns[x.dtypes == object].tolist()

    # if a nominal feature has only 2 levels:
    # encode it as a single binary variable
    for col in categorical_cols:
        n = len(x[col].unique())
        if n == 2:
            x[col] = pd.get_dummies(x[col], drop_first=True)

    # for categorical features with >2 levels: use one-hot-encoding
    # below, numerical columns will be untouched
    x = pd.get_dummies(x)

    # scale x between 0 and 1
    x = preprocessing.MinMaxScaler().fit_transform(x)
    # label-encode y
    y = preprocessing.LabelEncoder().fit_transform(y)
    # shuffle data at the end
    x, y = shuffle(x, y, random_state=321)

    return x, y


# ## how to run this script
x, y = prepare_dataset_for_modeling('us_census_income_data.csv')

