'''
evaluate recommendations from implicit recommender system
calculate cosine distances
'''
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def rmse(true, pred):
    return mean_squared_error(true, pred, squared=False)

def mae(true, pred):
    return mean_absolute_error(true, pred)

def get_name(x):
    return x.split('_')[-1]

def get_coverage(x):
    cov = ''
    if x[0].isdigit():
        cov = x.split('_')[0]
    else:
        cov = 'full'
    return cov

def normalize(x):
    return x / np.linalg.norm(x)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def cosine_distance(x, y):
    return 1 - cosine_similarity(x,y)

def mag(x): 
    return math.sqrt(sum(i**2 for i in x))

def main():
    df_cold     = pd.read_json('workflow/recommender/test_user_factors.json', orient='index')  # read in json file
    df_cold['name'] = df_cold['user'].apply(lambda x: get_name(x))  # get sample names of subsamples
    df_cold['coverage'] = df_cold['user'].apply(lambda x: get_coverage(x))  # get coverages of subsamples

    # cosine distance
    data = []
    for sample in df_cold['user']:
        name = df_cold.loc[df_cold['user'] == sample, 'name'].iloc[0]   # sample name w/o coverage prefix
        #true_to_hot  = np.array(df_hot_full.loc[df_hot_full['user'] == name, 'factors'].iloc[0])
        true = np.array(df_cold.loc[df_cold['user'] == name, 'factors'].iloc[0])
        pred = np.array(df_cold.loc[df_cold['user'] == sample, 'factors'].iloc[0])
        cov  = df_cold.loc[df_cold['user'] == sample, 'coverage'].iloc[0]
        #sim = cosine_similarity(normalize(true), normalize(pred))
        #dist_to_hot  = cosine_distance(normalize(true_to_hot), normalize(pred))
        dist = cosine_distance(normalize(true), normalize(pred))

        data.append([cov, dist])

    df = pd.DataFrame(data, columns=['coverage', 'cos_distance'])
    df.to_csv('cosine_recommender.csv', index=False, header=True)

if __name__ == "__main__":
    main()