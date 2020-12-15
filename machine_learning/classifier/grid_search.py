import numpy as np
import pandas as pd
from time import time
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_json('output_user_factors.json', orient='index')  # read in json file

new_df = df['factors'].apply(pd.Series) # get a column for every factor in factors list
new_df = new_df.rename(columns = lambda x : 'factor' + '_' + str(x)) # rename every created columns

df.drop('factors', axis=1, inplace=True)    # drop the original factors column

df = pd.concat([df, new_df], axis=1)    # add the new factor columns to the origional dataframe

X = df.drop('label', axis=1).drop('user', axis=1)   # drop target and drop user (user name must not have influence on prediction)
y = df['label'] # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=50)   # test train split

# Gradient Boost Classifier
boost_clf = GradientBoostingClassifier()    # initialize dummy classifier

# Utility function to report best scores
def report(results, n_top=3):
    '''
    copied from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters
param_grid = {
    'learning_rate': [0.05],
    'n_estimators': [200,300,500],
    'max_depth': [3,5],
}

# run grid search
grid_search = GridSearchCV(boost_clf, param_grid=param_grid, n_jobs=8)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)