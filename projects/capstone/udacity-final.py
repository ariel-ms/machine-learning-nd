# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
clear = lambda: os.system('cls')
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
petID = np.asarray(test.PetID)

# Print number of nan values
# print(train.isnull().sum().sum())
# print(test.isnull().sum().sum())

# for var in list(train):
#     print(var + ": " + str(train[[var]].isnull().sum()))

# for var in list(test):
    # print(var + ": " + str(test[[var]].isnull().sum()))

# Get the breeds dictionary
breeds = pd.read_csv('../input/breed_labels.csv')
breeds_dict = breeds.to_dict()["BreedName"]

# Add key 307 to value Mixed Breed - preprocessing step
breeds_dict[307] = 'Mixed Breed'

# Get colors dataframe and translate it into a dictionary
colors = pd.read_csv("../input/color_labels.csv")
colors_dict = colors.to_dict()["ColorName"]

#Get state dictionary
states = pd.read_csv("../input/state_labels.csv")
states_dictionary = states.to_dict()
states_dict = {value:states_dictionary["StateName"][key] for key, value in states_dictionary["StateID"].items()}

# Separate variables according to their type
continous_variables = ["Age", "Quantity", "PhotoAmt", "Fee"]
categorical_variables = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",\
                         "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized",\
                         "Health", "AdoptionSpeed", "State", 'RescuerID']

# # load json data
import json
from pprint import pprint

# returns the sentiment score from the metadata files
def get_sentiment_dict(directory_str):
    
    directory = os.fsencode(directory_str)
    sentiment_dict = {}
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        with open(directory_str + '/' + filename) as json_file:
            data = json.load(json_file)
            sentiment_dict[filename[:filename.index(".json")]] = data['documentSentiment']["score"]
    return sentiment_dict

sentiment_dict_train = get_sentiment_dict('../input/train_sentiment')
sentiment_dict_test = get_sentiment_dict('../input/test_sentiment')

# Add sentiment score to train sets
train['SentimentScore'] = train['PetID'].map(sentiment_dict_train)
test['SentimentScore'] = test['PetID'].map(sentiment_dict_test)

### DATA PREPROCESSING FOR VISUALIZATION ###
continous_dataframe = train[continous_variables]
categorical_dataframe = train[categorical_variables]

# map colors to their respective string
categorical_dataframe['Color1'] = categorical_dataframe['Color1'].map({0: 'Black', 1: 'Brown', 2: 'Golden', 3: 'Yellow', 4: 'Cream', 5: 'Gray', 6: 'White', 7: 'Brown'})
categorical_dataframe['Color2'] = categorical_dataframe['Color2'].map({0: 'Black', 1: 'Brown', 2: 'Golden', 3: 'Yellow', 4: 'Cream', 5: 'Gray', 6: 'White', 7: 'Black'})
categorical_dataframe['Color3'] = categorical_dataframe['Color3'].map({0: 'Black', 1: 'Brown', 2: 'Golden', 3: 'Yellow', 4: 'Cream', 5: 'Gray', 6: 'White', 7: 'Black'})

# map categorical breeds id to their respective name 
categorical_dataframe['Breed1'] =  categorical_dataframe['Breed1'].map(breeds_dict)
categorical_dataframe['Breed2'] =  categorical_dataframe['Breed2'].map(breeds_dict)

# map state key to their respective key
categorical_dataframe['State'] = categorical_dataframe['State'].map(states_dict)

# join continous and categorical dataframe into one
join_dataframe = continous_dataframe.join(categorical_dataframe).join(train['PetID'])

# add sentiment score to join dataframe
join_dataframe['SentimentScore'] = join_dataframe['PetID'].map(sentiment_dict_train)

# add sentiment score to continous dataframe
continous_dataframe['SentimentScore'] = join_dataframe['SentimentScore']

# plot pair plot of continous data frame
# sns.pairplot(continous_dataframe.join(train['AdoptionSpeed']), vars=["Age", "PhotoAmt", "SentimentScore", "Fee"], hue="AdoptionSpeed")

# Print boxplots of continous variables by AdoptionSpeed
# for col in list(continous_dataframe):
#     join_dataframe.boxplot(column=[col], by='AdoptionSpeed')

# Print distribution plots of continous variables
# y_vars = list(continous_dataframe)

# def plot_pdf(fig_size, x_var, y_vars, data, font_size = 30):
#     for i in range(fig_size[1]):
#         g = sns.FacetGrid(data, hue="AdoptionSpeed", height=5) \
#           .map(sns.distplot, y_vars[i], ) \
#           .add_legend()
#         g.fig.suptitle("Distribution plot of " + str(y_vars[i]), size = "13")
#     plt.show()

# plot_pdf((1, len(y_vars)), "AdoptionSpeed", y_vars, join_dataframe)

# Plot violin plots
# for col in list(continous_dataframe):
#     sns.violinplot(x = "AdoptionSpeed", y = col, data = join_dataframe)
#     plt.show()


# Plot categorical variable histograms
# categorical_dataframe['Type'].value_counts().plot(kind='bar', title = "Type of animal (1 = Dog, 2 = Cat)", grid = True)
# join_dataframe.boxplot(column = ['Age'], by='AdoptionSpeed')

# for categorical in categorical_dataframe:
#     if (categorical != "Breed1" and categorical != "Breed2" and categorical != "RescuerID"):
#         ax = categorical_dataframe[categorical].value_counts().plot(kind='barh', title = categorical, grid = True)
#         for i, v in enumerate(categorical_dataframe[categorical].value_counts()):
#             ax.text(v + 5, i, str(v), fontweight='bold')
#         plt.show()

### DATA PREPROCESSING FOR MODELING ###
X = pd.concat([train, test], ignore_index = True, sort = False)

print(X.isnull().sum().sum())

# Normalize
X[continous_variables] = X[continous_variables] = (X[continous_variables] - X[continous_variables].min()) / (X[continous_variables].max() - X[continous_variables].min())

# Standarize
X[continous_variables] = X[continous_variables] = (X[continous_variables] - X[continous_variables].mean()) / X[continous_variables].std()

# Fill Nans of Sentiment description
X['SentimentScore'].fillna(0, inplace=True)

# Fill NaN of Description column
X['Description'] = X['Description'].replace(np.nan, '', regex=True)

# # Perform TFIDF
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD

# vectorizer = TfidfVectorizer(stop_words = 'english')
# results = vectorizer.fit_transform(X.Description)

# svd = TruncatedSVD(n_components = 5, random_state = 42)
# resul_col = svd.fit_transform(results)
# resul_col = pd.DataFrame(resul_col)
# resul_col = resul_col.add_prefix('TFIDF_{}_'.format('Description'))
# X = pd.concat([X, resul_col], axis=1)

# variables to drop from both data sets
vars_to_drop = ['Name', 'RescuerID', 'PetID']

# Drop selected Columns (before this the tdfid step should be done )
X = X.drop(vars_to_drop, axis = 1)

# Divide dataframes
X_train = X.loc[np.isfinite(X.AdoptionSpeed), :]
X_test = X.loc[~np.isfinite(X.AdoptionSpeed), :]

X_test = X_test.reset_index()

# Perform TFIDF in X_train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

vectorizer = TfidfVectorizer(stop_words = 'english')
results = vectorizer.fit_transform(X_train.Description)

svd = TruncatedSVD(n_components = 1, random_state = 42)
resul_col = svd.fit_transform(results)
resul_col = pd.DataFrame(resul_col)
resul_col = resul_col.add_prefix('TFIDF_{}_'.format('Description'))
X_train = pd.concat([X_train, resul_col], axis=1)
X_train = X_train.drop('Description', axis =1)

vectorizer = TfidfVectorizer(stop_words = 'english')
results = vectorizer.fit_transform(X_test.Description)

svd = TruncatedSVD(n_components = 1, random_state = 42)
resul_col = svd.fit_transform(results)
resul_col = pd.DataFrame(resul_col)
resul_col = resul_col.add_prefix('TFIDF_{}_'.format('Description'))
X_test = pd.concat([X_test, resul_col], axis=1)

X_test = X_test.drop(['AdoptionSpeed', 'Description', 'index'], axis = 1)

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]

from sklearn.model_selection import StratifiedKFold

# Train xgboost
# import xgboost as xgb
# def train_XGBoost(train, test, params):
#     kfolds = 10
#     folds = StratifiedKFold(n_splits = kfolds, shuffle = True, random_state = 42)
#     output_df = np.zeros((test.shape[0], kfolds))
#     col = 0
    
#     for train_index, val_index in folds.split(train, train['AdoptionSpeed'].values):
#         X_train = train.iloc[train_index, :] 
#         X_val = train.iloc[val_index, :]
        
#         y_train = X_train['AdoptionSpeed'].values
#         y_val = X_val['AdoptionSpeed'].values
        
#         X_train = X_train.drop(['AdoptionSpeed'], axis = 1)
#         X_val = X_val.drop(['AdoptionSpeed'], axis = 1)
        
#         dm_train = xgb.DMatrix(data = X_train, label = y_train, feature_names = X_train.columns)
#         dm_val = xgb.DMatrix(data = X_val, label = y_val, feature_names = X_test.columns)
        
#         watchlist = [(dm_train, 'train'), (dm_val, 'validation')]
        
#         model = xgb.train(dtrain=dm_train, num_boost_round=30000, evals = watchlist, \
#                          early_stopping_rounds=500, verbose_eval=1000, params=params)
        
#         predictions_test = model.predict(xgb.DMatrix(test, feature_names=test.columns), ntree_limit = model.best_ntree_limit)
#         output_df[:, col] = predictions_test
#         col += 1
#     return model, output_df

# params = { 'eval_metric': 'rmse', 'seed': 42, 'silent': 1}
# model, output = train_XGBoost(X_train, X_test, params)
# output = output.mean(axis=1)

# output = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit = model.best_ntree_limit) # esta linea ya no

# Train lightgbm algorithm
import lightgbm as lgb
def train_LGBM(train, test, params):
    kfolds = 10
    folds = StratifiedKFold(n_splits = kfolds, shuffle = True, random_state = 42)
    col =0
    output_df = np.zeros((test.shape[0], kfolds))
    
    for train_index, val_index in folds.split(train, train['AdoptionSpeed'].values):
        X_train = train.iloc[train_index, :] 
        X_val = train.iloc[val_index, :]
        
        y_train = X_train['AdoptionSpeed'].values
        y_val = X_val['AdoptionSpeed'].values
        
        X_train = X_train.drop(['AdoptionSpeed'], axis = 1)
        X_val = X_val.drop(['AdoptionSpeed'], axis = 1)
        
        d_train = lgb.Dataset(X_train, label=y_train)
        d_valid = lgb.Dataset(X_val, label=y_val)
        
        watchlist = [d_train, d_valid]
        model = lgb.train(params,
                          train_set=d_train,
                          valid_sets=watchlist,
                          verbose_eval=params['verbose_eval'],
                          early_stopping_rounds=100)
        predictions_test = model.predict(test, num_iteration=model.best_iteration)
        output_df[:, col] = predictions_test
        col += 1
    return model, output_df
    
params = {
    'application': 'regression',
    'mertic': 'rmse',
    'metric': 'auc',
    'boosting': 'gbdt',
    'num_leaves': 50,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose_eval': 50
}

model, output = train_LGBM(X_train, X_test, params)
output = output.mean(axis=1)

df = pd.DataFrame(data = {'PetID': petID, 'AdoptionSpeed': output})
df.AdoptionSpeed = df.AdoptionSpeed.astype('int32')

# df.AdoptionSpeed.value_counts()
df.to_csv('submission.csv', index = False)

# op=pd.DataFrame(data={'PassengerId':test['PassengerId'],'Survived':model.predict(testdf)})
# op.to_csv('KFold_XGB_GridSearchCV_submission.csv',index=False)