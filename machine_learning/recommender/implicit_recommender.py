'''
This script creates a recommender system by using the implicit module from
benfred (https://github.com/benfred/implicit).
'''

import os
import json
import glob
import umap
import implicit
import argparse
import umap.plot
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import date
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from implicit.evaluation import precision_at_k

def get_arguments():
    '''
    get specified command line arguments and define default parameters
    '''
    parser = argparse.ArgumentParser(description='Train a recommender system based on gene-presence-absence vectors to calculate relations between organisms and annotated domains')
    input_args = parser.add_argument_group('Input')
    input_args.add_argument(
        '--input_train',
        metavar='',
        type=str,
        default='workflow/gpav/gpav_train/*',
        help='path to train gene-presence-absence files'
    )
    input_args.add_argument(
        '--input_test',
        metavar='',
        type=str,
        default='workflow/gpav/gpav_test/*',
        help='path to test gene-presence-absence files'
    )
    input_args.add_argument(
        '--metadata',
        metavar='',
        type=str,
        default='workflow/metadata/metadata_preprocessed.csv',
        help='path to input metadata csv file'
    )
    input_args.add_argument(
        '--model',
        metavar='',
        type=str,
        default='ALS',
        help='recommender model to use {ALS:Alternating Least Squares,\
        BPR: Bayesian Personaliyed Ranking, LMF: Logistic Matrix Factorization, \
        AALS: Approximate Alternating Least Squares}  (default: ALS)'
    )
    input_args.add_argument(
        '--factors',
        metavar='',
        type=int,
        default=50,
        help='number of factors (default=50)'
    )
    input_args.add_argument(
        '--iterations',
        metavar='',
        type=int,
        default=20,
        help='number of fitting iterations (default=20)'
    )
    input_args.add_argument(
        '--regularization',
        metavar='',
        type=float,
        default=0.01,
        help='regularization factor (default=0.01)'
    )
    input_args.add_argument(
        '--confidence',
        metavar='',
        type=float,
        default=1.0,
        help='confidence factor (default=1.0)'
    )
    input_args.add_argument(
        '--threads',
        metavar='',
        type=int,
        default=0,
        help='number of used threads (default=all)'
    )
    input_args.add_argument(
        '--dimension_reduction',
        nargs='?',
        metavar='',
        const='tsne',
        type=str,
        help='optional dimension reduction, choose between umap (global structure) and tsne (local structure) (default = tsne)'
    )
    """
    input_args.add_argument(
        '--accessory_only',
        metavar='',
        nargs='?',
        const='accessory_genome_95.csv',
        type=str,
        help='optionally include only domains listed in the specified csv file (default="accessory_genome_95.csv")'
    )
    """
    input_args.add_argument(
        '--train_test',
        action='store_true',
        help='optional calculation of recommendations for test data'
    )

    # command line parameter settings for recalcultion of test inputs (hot/cold recommendation)
    parser.add_argument(
        '--recalculation',
        dest='recalculation',
        action='store_true',
        help='recalculation of the model for recommendations of test data (hot recommendation)'
    )
    parser.add_argument(
        '--no_recalculation',
        dest='recalculation',
        action='store_false',
        help='no recalculation of the model recommendation of test data (cold recommendation, default)'
    )
    parser.set_defaults(recalculation=False)


    output_args = parser.add_argument_group('Output')
    output_args.add_argument(
        '--output_dir',
        metavar='',
        type=str,
        default='workflow/recommender',
        help='directory for different outputs and results (default=redux/workflow/recommender)'
    )
    output_args.add_argument(
        '--output_filename',
        metavar='',
        type=str,
        default=date.today().strftime('%Y-%m-%d'),
        help='file prefix for different outputs and results (default=today`s date)'
    )

    return parser.parse_args()


def interaction_matrix_from_dataframe(data):
    '''
    this function returns an user-item-interaction matrix (csr format) created from a pandas dataframe
    
    pandas.Series.cat.codes:    return series of codes as well as the index.
    pandas.Categorical.codes:   codes are an array of integers which are the positions of the actual values in the categories array.
    
    compressed sparse row matrix
    instantiation: csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)]),
                    where data, row_ind and col_ind satisfy the relationship
                    a[row_ind[k], col_ind[k]] = data[k] (look at func: csr_matrix_instatiation_test()).
    '''
    data['user']        = data['user'].astype('category')
    data['item']        = data['item'].astype('category')
    data['label']    = data['label'].astype('category')
    data['user_code']  = data['user'].cat.codes.copy()
    data['item_code']  = data['item'].cat.codes.copy()
    data['label_code'] = data['label'].cat.codes.copy()

    interaction_matrix  = csr_matrix(   # instatiation of an interaction matrix in csr format 
        (np.ones(data.shape[0]), (data['item_code'], data['user_code']))
    )

    return data, interaction_matrix


def model_initialization(model_type, num_factors, num_iterations, fac_regularization, num_threads):
    '''
    initializes the specified recommender model with specified parameters
    '''

    if not model_type or model_type == 'ALS' or model_type == 'Alternating Least Squares':
        model = implicit.als.AlternatingLeastSquares(
            iterations     = num_iterations,
            factors        = num_factors,
            regularization = fac_regularization,
            num_threads    = num_threads
        )
        #model_name = 'Alternating Least Squares'

    elif model_type == 'BPR' or model_type == 'Bayesian Personaliyed Ranking':
        model = implicit.bpr.BayesianPersonalizedRanking(
            iterations     = num_iterations,
            factors        = num_factors,
            regularization = fac_regularization,
            num_threads    = num_threads
        )
        #model_name = 'Bayesian Personalized Ranking'

    elif model_type == 'LMF' or model_type == 'Logistic Matrix Factorization':
        model = implicit.lmf.LogisticMatrixFactorization(
            iterations     = num_iterations,
            factors        = num_factors,
            regularization = fac_regularization,
            num_threads    = num_threads
        )
        #model_name = 'Logistic Matrix Factorization'

    elif model_type == 'AALS' or model_type == 'Approximate Alternating Least Squares':
        model = implicit.approximate_als.NMSLibAlternatingLeastSquares(
            iterations     = num_iterations,
            factors        = num_factors,
            regularization = fac_regularization,
            num_threads    = num_threads
        )
        #model_name = 'Approximate Alternating Least Squares'

    else:
        raise Exception('Specified model type is not available. See the help message for further details.\n')

    return model


def print_model_parameters(model):
    '''
    prints selected model parameters to the command line
    (number of factors, number of fitting iterations, regularization factor)
    '''
    res  = ''
    res += 'Model parameters:\n'
    res += 'Number of factors:\t\t{}\n'.format(model.factors)
    res += 'Number of fitting iterations:\t{}\n'.format(model.iterations)
    res += 'Regularization factor:\t\t{}\n'.format(model.regularization)
    print(res)

    return


def get_similar_users(model, userid, N):
    '''
    calculates the n (N) similar users for a specified user (userid)
    '''
    rec = model.similar_users(userid=userid, N=N)

    return rec


def get_similar_items(model, itemid, N):
    '''
    calculates the n (N) similar items for a specified item (itemid)
    '''
    rec = model.similar_items(itemid=itemid, N=N)

    return rec


def recommend_items_for_a_user(model, userid, user_item_matrix, N):
    '''
    calculates the n (N) best recommendations for a user (userid), and returns a list of itemids, score.
    '''
    recs = model.recommend(
        userid=userid,
        user_items=user_item_matrix,
        N=N,
    )

    return recs


def recommend_items_for_all_users(model, user_items_matrix, N):
    '''
    calculates the n (N) best recommendations for all users and returns numpy ndarray of shape (userid, N) with item’s ids in reversed probability order
    '''
    rec = model.recommend_all(
        user_items=user_items_matrix,
        N=N,
        filter_already_liked_items=False,
        show_progress=True
    )

    return rec


def trim_dataframe_to_accessory_genome(data, acc_gen):
    acce = pd.read_csv(acc_gen, names=['item'])['item'].tolist() # read in accessory domains from specified csv file to a list
    data[data['item'].isin(acce)].reset_index(drop=True, inplace=True)  # only keep rows with an entry in the above specified list (--> remove core genome)

    return data


def preprocessing(data, meta, acc_gen):
    '''
    preprocess the input data
    return data and interaction matrix
    '''

    # get and preprocess metadata (label), afterwards map it to the input data
    meta = pd.read_csv(meta, header=0, usecols=[0, 2], names=['sample_name','label']) # get metadata from input metadata file to map labels to users
    data = pd.merge(data, meta, on='sample_name')  # map metadata to gpa data

    if acc_gen: # if accessory genome was specified
        data = trim_dataframe_to_accessory_genome(data, acc_gen)    # trim the data to accessory domains only

    #print(data, '\n')   # print the created dataframe
    #print(data.info(), '\n')    # print information on the created dataframe
    #print(data.isnull().sum(), '\n')    # print the number of missing values

    data, interaction_matrix = interaction_matrix_from_dataframe(data)  # create an interaction matrix and expand information on the dataframe (category codes)
   
    #data.to_csv('{}/{}_data.csv'.format(outdir, outfile), index=False)  # save the expanded dataframe to a csv file

    interaction_matrix.check_format()   # scipy function to check whether the matrix format is valid
    #print(interaction_matrix.get_shape(), '\n') # print the shape of the interaction matrix
    #print(interaction_matrix.count_nonzero(), '\n') # print the number of non zero elements in the interaction matrix
    #print(interaction_matrix.getnnz(), '\n')    # print the total number of elements in the interaction matrix
    #sparsity = 100*(1 - (interaction_matrix.count_nonzero()/(interaction_matrix.shape[0]*interaction_matrix.shape[1]))) # print the sparsity of the created interaction matrix (number of interactions/number of possible interactions)
    #print('Sparsity:\t\t{}'.format(sparsity), '\n')

    return data, interaction_matrix


def model_training(confidence, interaction_matrix, model):
    '''
    train recommender model with input data
    '''
    model.fit(confidence * interaction_matrix, show_progress=False) # fit model to train data

    return model


def make_recommendations(model, output_filename):
    '''
    returns a recommendation made from specified functions from implicit module (benfred);
    see the specified functions for further information
    '''
    recommendation = None
    #recommendation = recommend_items_for_a_user(model, 1, interaction_matrix.T, 3)
    #recommendation = recommend_items_for_all_users(model, interaction_matrix.T,3)
    #recommendation = get_similar_items(model, 1, 3)
    #recommendation = get_similar_users(model, 1, 3)

    #np.savetxt('{}_recommendations.csv'.format(output_filename), recommendation, delimiter=',')    # save the recommendation to an output csv file

    return recommendation


def dimension_reduction(software, factors, outdir, outfile, data, labels):
    
    if software == 'umap':  # if umap was specified in the command line
        reducer = umap.UMAP(    # initialization of umap model (better for global structure)
            n_components=2, # reduction to two dimensions
            n_neighbors=20,
            min_dist=0.1,
            metric='euclidean'
            )

        embedding = reducer.fit_transform(factors)  # fit data to model

    elif software == 'tsne':    # if tsne was specified in the command line
        reducer = TSNE( # initialization of tsne model (better for local structur)
            n_components = 2    # reduction to two dimensions
        )

        embedding = reducer.fit_transform(factors)  # fit data to model

    liste = []
    for label, (c1, c2) in zip(labels, embedding):  # iterate over each label and each embedding (x and y coordinate)
        liste.append([label,c1,c2]) # append list by label, x- and y-coordinate of embedding

    embedding = pd.DataFrame(liste, columns=['label', 'x', 'y'])    # create dataframe from liste with embedding and labels
    embedding.to_csv('{}/{}_embedding.csv'.format(outdir, outfile), index=False)    # save dataframe to csv file

    return embedding


def get_all_input_gpav_files(input_files):
    data = []  # initialize a list, which will be filled with data
    for input_file in glob.glob(input_files): # iterate over every file in the input directory
        name = input_file.split('/')[-1]

        if name[0].isdigit():
            gpav_name  = '_'.join(name.split('_')[:2]) # get the sample name (ENA accession number) from file name
            proportion = name.split('_')[0] # get proportion from file name
            sample_name = name.split('_')[1] # get sample name from file name

        else:
            gpav_name  = name.split('_')[0] # get the sample name (ENA accession number) from file name
            proportion = 'full' # assign 'full' for not subsampled input read file
            sample_name = gpav_name  # assign gpav_name as sampleName

        with open(input_file, 'r') as f:    # open input file
            for line in f:  # iterate over every line in input file
                split_line  = line.strip().split('\t')
                dom_name    = split_line[0] # get the domain name
                ann_binary  = int(split_line[1])    # get binary value for presence (1) or absence (0)

                if ann_binary == 1: # if domain is present
                    new_list = []   # initialize new list for sample data
                    new_list.append(gpav_name)  # add sample name to new list
                    new_list.append(dom_name)   # add domain name to new list
                    new_list.append(sample_name) # add sample name to new list
                    new_list.append(proportion)   # add proportion to new list
                    data.append(new_list)   # add new list to data list

    df = pd.DataFrame(data, columns=['user', 'item', 'sample_name', 'sample_proportion'])    # create a pandas dataframe from list of lists

    return df


def get_label_mapping(data):
    '''
    get a list of labels (phenotypes) in the right order;
    sorted by ascending user category codes, cause user category
    codes are needed for interaction matrix construction and created
    by lexicographical order of user names
    --> index of label maps to index of user in user factors

    conclusion: making weird list stuff to get a mapping from user factor index to label
    '''
    labels = []
    foo = 0 # helper variable, to intermediatly save an user code
    for i,user in enumerate(data['user_code']): # iterate over every user code
        if user != foo:
            labels.append([user, data['label'][i]])
            foo = user
    labels = [el[1] for el in sorted(labels, key=lambda x: x[0])]   # sort the labels by user code but keep only labels

    return labels


def plot_embedding(embedding, labels, outdir, outfile):
    '''
    plot the calculated embedding in png and svg file
    '''
    scatter_x = np.array(embedding['x'])
    scatter_y = np.array(embedding['y'])
    group     = np.array(embedding['label'])


    fig,ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], label = g, s = 5)

    plt.savefig('{}/{}_dimension_reduction.png'.format(outdir, outfile), format='png', dpi=300)
    plt.savefig('{}/{}_dimension_reduction.svg'.format(outdir, outfile), format='svg', dpi=1200)

    return fig


def save_train_user_factors_and_labels_to_json(factors, labels, data, outdir, outfile):
    '''
    create a dictionary from an user factor label mapping;
    save the dictionary in json format to a file
    '''
    liste = []
    data = data.sort_values('user_code')
    sample_names = data['user'].unique().tolist()
    
    for i,sample in enumerate(sample_names):
        liste.append([sample, labels[i], factors[i]])
    
    df = pd.DataFrame(liste, columns = ['user', 'label', 'factors'])
    df.to_json('{}/{}_train_user_factors_labels.json'.format(outdir, outfile), orient='index')

    return None


def preprocessing_test_data(df, meta, acc_gen):
    '''
    preprocess input test data
    '''

    # get and preprocess metadata (label), afterwards map it to the input data
    meta = pd.read_csv(meta, header=0, usecols=[0, 2], names=['sample_name','label']) # get metadata from input metadata file to map labels to users
    df = pd.merge(df, meta, on='sample_name')  # map metadata to gpa data

    if acc_gen: # if accessory genome was specified
        df = trim_dataframe_to_accessory_genome(df, acc_gen)    # trim the data to accessory domains only

    return df


def calculate_and_save_test_user_factors_and_labels_to_json(model, d_train_raw, d_train_pro, d_test, meta, acc_gen, outdir, outfile, recalculate, model_type, conf):
    '''
    model.recalculate_user(userid, user_items)
    https://github.com/benfred/implicit/issues/343

    user_items (csr_matrix) – A sparse matrix of shape (number_users, number_items).
    This lets us look up the liked items and their weights for the user.
    This is used to filter out items that have already been liked from the
    output, and to also potentially calculate the best items for this user.

    command: model.recalculate_user(userid, user_items) #return user factors
    '''
    d_test = get_all_input_gpav_files(d_test)
    d_test = preprocessing_test_data(d_test, meta, acc_gen)

    liste_cold = []
    for user in d_test['user'].unique():
        label = d_test.loc[d_test['user'] == user, 'label'].iloc[0]
        items = d_test.loc[d_test['user'] == user, 'item'].tolist()
        name  = d_test.loc[d_test['user'] == user, 'sample_name'].iloc[0]

        factors_cold = recommend_cold(items, d_train_pro, model)
        liste_cold.append([user, label, factors_cold])
    
    df_cold = pd.DataFrame(liste_cold, columns = ['user', 'label', 'factors'])
    df_cold.to_json('{}/{}_test_user_factors_labels_cold.json'.format(outdir, outfile), orient='index')

    liste_hot  = []
    for user in d_test['user'].unique():
        if not user[0].isdigit():
            label = d_test.loc[d_test['user'] == user, 'label'].iloc[0]
            items = d_test.loc[d_test['user'] == user, 'item'].tolist()
            name  = d_test.loc[d_test['user'] == user, 'sample_name'].iloc[0]

            factors_hot = recommend_hot(user, items, name, d_train_raw, model, model_type, meta, acc_gen, conf)
            liste_hot.append([user, label, factors_hot])
    
    df_hot = pd.DataFrame(liste_hot, columns = ['user', 'label', 'factors'])
    df_hot.to_json('{}/{}_test_user_factors_labels_hot.json'.format(outdir, outfile), orient='index')

    """
        if recalculate is True:
            factors = recommend_hot(user, items, name, d_train_raw, model, model_type, meta, acc_gen, conf)

        else:
            factors = recommend_cold(items, d_train_pro, model)

        liste.append([user, label, factors])

    df = pd.DataFrame(liste, columns = ['user', 'label', 'factors'])

    if recalculate is True:
        df.to_json('{}/{}_test_user_factors_labels_hot.json'.format(outdir, outfile), orient='index')

    else:
        df.to_json('{}/{}_test_user_factors_labels_cold.json'.format(outdir, outfile), orient='index')
    """

    return


def _items2matrix(items, data, model):
    '''
    Turn list of items into matrix representation
    '''
    x = data[['item', f'item_code']].drop_duplicates()
    map_name_id = {k: v for _, (k, v) in x.iterrows()}

    star_ids = [map_name_id.get(i, None) for i in items]
    star_ids = [i for i in star_ids if i]
    data = [1 for _ in star_ids]
    rows = [0 for _ in star_ids]
    shape = (1, model.item_factors.shape[0])
    M = sparse.coo_matrix((data, (rows, star_ids)), shape=shape).tocsr()

    return M


def recommend_cold(items, data, model):
    '''
    recommend a user vector by cold recommendation
    (w/o recalculating the whole model)
    '''
    item_matrix = _items2matrix(items, data, model)
    rec_vector = model.recalculate_user(
        userid     = 0,
        user_items = item_matrix
    )

    return rec_vector


def recommend_hot(user, items, name, data, model, model_type, meta, acc_gen, conf):
    '''
    recommend a user vector by hot recommendation
    (w/ recalculating the whole model)
    '''
    new_rows = []   # initialize empty list, to store data
    for item in items:  # iterate over every item in items list
        new_rows.append([user, item, name])   # append list by several variables
    
    new_df   = pd.DataFrame(new_rows, columns=['user', 'item', 'sample_name']) # create dataframe from list
    new_data = data.append(new_df, ignore_index=True)   # append train dataframe by created dataframe from single sample

    new_data_pro, im = preprocessing(new_data, meta, acc_gen)   # preprocess appended dataframe and return new interaction matrix

    model = model_initialization(   # initialize new model based on parameters from old model
        model_type         = model_type,
        num_factors        = model.factors,
        num_iterations     = model.iterations,
        fac_regularization = model.regularization,
        num_threads        = model.num_threads
    )

    model        = model_training(conf, im, model)  # train model with new interaction matrix
    new_data_pro = new_data_pro.sort_values('user_code')    # sort appended dataframe by user_code
    users_sorted = new_data_pro['user'].unique().tolist()   # extract unique users for a mapping to factors (sorted lexicographically by user_code)

    rec_vector = [] # initialize empty list, to store the wanted user factors
    for i,u in enumerate(users_sorted): # iterate over every user in sorted users list
        if u == user:   # if entry matches the wanted user
            rec_vector = model.user_factors.tolist()[i] # define user factors at i as wanted user factors

    return rec_vector


def main():

    # get command line arguments
    args = get_arguments()

    # create a dataframe from train gpav files
    data_raw = get_all_input_gpav_files(args.input_train)

    # print the created dataframes
    #print('Train:', '\n', data, '\n')

    # preprocess train data
    data_pro, interaction_matrix = preprocessing(
        data = data_raw,
        meta    = args.metadata,
        acc_gen = args.accessory_only
    )

    # print the created dataframes
    #print('Train:', '\n', data, '\n')

    # initializing new recommender model
    model = model_initialization(
        model_type         = args.model,
        num_factors        = args.factors,
        num_iterations     = args.iterations,
        fac_regularization = args.regularization,
        num_threads        = args.threads
    )

    # print model parameters
    #print_model_parameters(model)

    # train recommender model on created sparse interaction matrix
    model = model_training(
        confidence         = args.confidence,
        interaction_matrix = interaction_matrix,
        model              = model
    )

    # get a list of labels (phenotypes)
    labels = get_label_mapping(data_pro)
    
    if args.dimension_reduction:    # if dimension reduction was enabled
        # make dimension reduction of user factors
        embedding = dimension_reduction(
            software    = args.dimension_reduction,
            factors     = model.user_factors,
            outdir      = args.output_dir,
            outfile     = args.output_filename,
            data        = data_pro,
            labels      = labels
        )

        # plot the resulting embedding from the dimension reduction
        plot_embedding(
            embedding = embedding,
            labels    = labels,
            outdir    = args.output_dir,
            outfile   = args.output_filename
        )

    # save train user factors and corresponding labels in json file
    save_train_user_factors_and_labels_to_json(
        factors = model.user_factors.tolist(),  # parsing ndarray to list
        labels  = labels,
        data    = data_pro,
        outdir  = args.output_dir,
        outfile = args.output_filename
        )


    if args.train_test:
        # predict test user vectors based on trained model
        calculate_and_save_test_user_factors_and_labels_to_json(
            model       = model,
            d_train_raw = data_raw,
            d_train_pro = data_pro,
            d_test      = args.input_test,
            meta        = args.metadata,
            acc_gen     = args.accessory_only,
            outdir      = args.output_dir,
            outfile     = args.output_filename,
            recalculate = args.recalculation,
            model_type  = args.model,
            conf        = args.confidence
        )


if __name__ == '__main__':

    main()