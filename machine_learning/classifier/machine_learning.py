import sys
import glob
import torch
import argparse
import numpy as np
import pandas as pd
from uuid import uuid4
from datetime import date
from torch import nn, optim
import matplotlib.pyplot as plt
from joblib import parallel_backend
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from torch.utils.data import random_split, DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,recall_score, f1_score, classification_report, cohen_kappa_score

# CPU or GPU?
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")    # uncomment this to run on CPU
# device = torch.device("cuda:0")   # uncomment this to run on GPU

def get_arguments():
    '''
    get command line arguments
    '''
    parser = argparse.ArgumentParser(description='Machine learning script which implements a dummy and a gradien boosting classifier from scikit-learn and a neural network classifier from pytorch')

    input_args = parser.add_argument_group('Input')
    input_args.add_argument(
        '--train',
        metavar='',
        type=str,
        default='workflow/recommender/train_user_factors_labels.json',
        help='path to train input'
    )
    input_args.add_argument(
        '--test',
        metavar='',
        type=str,
        default='workflow/recommender/test_user_factors_labels_*.json',
        help='path to test input'
    )
    input_args.add_argument(
        '--num_epochs',
        metavar='',
        type=int,
        default=10,
        help='number of training iterations (default=10)'
    )
    input_args.add_argument(
        '--min_appearance',
        metavar='',
        type=int,
        default=0,
        help='number of minimal appearances of labels in train set (default=0)'
    )
    input_args.add_argument(
        '--coverage_split',
        dest='coverage_split',
        action='store_true',
        help='split test set by coverages (default=no_coverage_split)'
    )
    input_args.add_argument(
        '--no_coverage_split',
        dest='coverage_split',
        action='store_false',
        help='split not test set by coverages (default)'
    )
    input_args.set_defaults(coverage_split=False)

    input_args.add_argument(
        '--learning_curve',
        dest='learning_curve',
        action='store_true',
        help='produces a learning curve (only for gb yet) and exits (default=False)'
    )
    input_args.set_defaults(learning_curve=False)

    input_args.add_argument(
        '--confusion_matrix',
        action='store_true',
        help='create confusion matrix/matrices'
    )
    input_args.add_argument(
        '--batch_size',
        metavar='',
        type=int,
        default=100,
        help='size of batch of data seen at each epoch at once (default=100)'
    )
    input_args.add_argument(
        '--num_jobs',
        metavar='',
        type=int,
        default=10,
        help='not implemented yet, number of jobs for model calculations (default=10)'
    )
    input_args.add_argument(
        '--dummy',
        action='store_true',
        dest='dummy',
        help='train and test dummy classifier'
    )
    #input_args.add_argument(
    #    '--no_dummy',
    #    action='store_true',
    #    dest='dummy',
    #    help='foo bar'
    #)
    input_args.set_defaults(dummy=False)

    input_args.add_argument(
        '--gradient_boosting',
        action='store_true',
        dest='gradient_boosting',
        help='train and test gradient boosting classifier'
    )
    #input_args.add_argument(
    #    '--no_gradient_boosting',
    #    action='store_true',
    #    dest='gradient_boosting',
    #    help=''
    #)
    input_args.set_defaults(gradient_boosting=False)

    input_args.add_argument(
        '--neural_network',
        action='store_true',
        dest='neural_network',
        help='train and test neural network classifier'
    )
    #input_args.add_argument(
    #    '--no_neural_network',
    #    action='store_false',
    #    dest='neural_network',
    #    help=''
    #)
    input_args.set_defaults(neural_network=False)
    
    input_args.add_argument(
        '--classification_report',
        action='store_true',
        help='calculate classification report'
    )

    output_args = parser.add_argument_group('Output')
    output_args.add_argument(
        '--output_dir',
        metavar='',
        type=str,
        default='.',
        help='directory for different outputs and results (default=".")'
    )
    output_args.add_argument(
        '--output_filename',
        metavar='',
        type=str,
        default=date.today().strftime('%Y-%m-%d'),
        help='file prefix for different outputs and results (default=today`s date)'
    )

    return parser.parse_args()


def preprocessing(path_train, path_test, min_appearance):
    '''
    load train and test json, return dataframes and common label index
    '''
    df_train = pd.read_json(path_train, orient='index')
    df_test  = pd.read_json(path_test, orient='index')

    # drop all samples of labels with less then n appaerances in train sample set
    labels = list(zip(*df_train.values))[1]   # labels into list
    label_appeaerance_map  = [[x,labels.count(x)] for x in set(labels)] # create a list of [label, appearance]
    bottom_labels = [x[0] for x in label_appeaerance_map if x[1] < min_appearance] # get labels with less appearances than min_app
    df_train = df_train[~df_train['label'].isin(bottom_labels)].reset_index(drop=True) # drop all samples with labels in bottom_labels, reset index afterwards
    df_test  = df_test[~df_test['label'].isin(bottom_labels)].reset_index(drop=True) # drop all samples with labels in bottom_labels, reset index afterwards

    # map labels to ints, cause pytorch tensors do not accept strings
    labels_train = list(zip(*df_train.values))[1]
    labels_test  = list(zip(*df_test.values))[1]
    common_labels = labels_train + labels_test
    index = {}  
    for i, j in enumerate(set(common_labels)):   # iterate over every unique label
        index[j] = i
    rev_common_index = {v: k for k, v in index.items()}    # reverse index, cause we want to look up results later

    return df_train, df_test, rev_common_index


class Dataset(object):
    '''
    An abstract class representing a Dataset.
    
    https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

    All other datasets should subclass it. All subclasses should override
    __len__, that provides the size of the dataset, and __getitem__,
     exclusive.
    '''
    def __init__(self, df, index):
        self.index     = index # specify passed index as index
        self.names, self.labels, self.coverages, self.X = self._load_data(df, index)    # load names, labels, coverages and features

    def __len__(self):
        '''
        return the size of the dataset
        '''
        return len(self.names)
    
    def __getitem__(self, index):
        '''
        supporting integer indexing in range from 0 to len(self)
        return a generated sample of data
        '''
        x = self.X[index]
        y = self.labels[index]

        return x, y

    def _load_data(self, df, index):
        '''
        load data from dataframe
        return list of names, label array, list coverages and two dimensional value array
        '''
        names, labels, values = list(zip(*df.values))

        rev_index = {v: k for k, v in index.items()}    # reverse index, cause we want to look up results later
        y = np.array([rev_index[i] for i in labels])  # get list of labels and map labels to integers

        coverages = [name.split('_')[0] if name[0].isdigit() else 'full' for name in names] # get coverages from names
        X = np.array([np.array(x) for x in values]) # get two dimensional array of features

        return list(names), y, coverages, X


def training_loop(num_epochs, optimizer, model, loss_fn, trainloader):
    '''
    train loop for pytorch neural network
    '''

    print('\n')
    print('Neural network learning:')
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(trainloader):
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs.float())
            labels_ = labels.type_as(outputs).unsqueeze(1)  # add extra dimension w/ unsqueeze
            loss = loss_fn(outputs, labels_)
            optimizer.zero_grad()   # zero the parameter gradients, so backprop won't accumulate them
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch {epoch}')
                print(f'Train Loss {loss.item()}')

    print('Finished training')
    return


def create_and_plot_confusion_matrix(true, pred, label_map, coverage, outdir, outfile):
    '''
    create and plot a confusion matrix of true and predicted labels
    '''
    cm = confusion_matrix(  # create the confusion matrix
        y_true = [label_map[x] for x in true],
        y_pred = [label_map[x] for x in pred],
        labels = list(label_map.values())
    )

    # Per-class accuracy
    #class_accuracy = conf_mat.diagonal()/conf_mat.sum(1)
    #print(class_accuracy)

    pd.DataFrame(cm).to_csv('{}/{}_{}_confusion_matrix.csv'.format(outdir, outfile, coverage), index=False, header=False)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))

    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp = disp.plot()  #include_values=include_values, cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)
    plt.savefig('{}/{}_{}_confusion_matrix.png'.format(outdir, outfile, coverage), dpi=300)

    return


def init_and_train_dummy_classifier(params, X_train, y_train):
    '''
    initialize and train dummy classifier, then predict targets
    '''
    clf = DummyClassifier(**params) # initialize dummy classifier
    clf.fit(X_train, y_train) # train dummy classifier

    return clf


def init_and_train_gradient_boosting_classifier(params, X_train, y_train, n_jobs):
    '''
    initialize and train gradient boosting classifier, then predict targets
    '''
    with parallel_backend('threading', n_jobs=n_jobs):
        clf = GradientBoostingClassifier(**params)  # initialize gradient boosting classifier with 'best' results from grid search
        clf.fit(X_train, y_train)   # train gb classifier

    return clf


def get_metrices(true, pred):
    '''
    return metrices 
    '''
    accuracy    = accuracy_score(true, pred)
    f1          = f1_score(true, pred, average='macro')
    recall      = recall_score(true, pred, average='macro')
    kappa       = cohen_kappa_score(true, pred)

    return accuracy, f1, recall, kappa


def get_test_data_by_coverage(data, index):
    '''
    split test data by coverage into new data sets
    return list of new data sets
    '''
    list_test_sets = []

    for cov in set(data.coverages):
        d = []
        for i,c in enumerate(data.coverages):
            if c == cov:
                d.append([data.names[i], index[data.labels[i]], data.X[i]])

        list_test_sets.append(pd.DataFrame(d, columns=['user', 'label', 'factors']))

    list_test_sets = [Dataset(x, index) for x in list_test_sets]

    return list_test_sets


def normalize(x):
    '''
    normalize np array to length 1
    '''
    return x / np.linalg.norm(x)


def plot_learning_curve(X, y, params, num_jobs, outdir, outfile, train_sizes=np.linspace(.1, 1.0, 5)):
    '''
    derived from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    '''

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = GradientBoostingClassifier(**params)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=num_jobs,
        train_sizes=train_sizes,
        return_times=True
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.legend(loc="best")

    df = pd.DataFrame(
        {
        'train_scores_mean':train_scores_mean,
        'train_scores_std':train_scores_std,
        'test_scores_mean':test_scores_mean,
        'test_scores_std':test_scores_std,
        'fit_times_mean':fit_times_mean,
        'fit_times_std':fit_times_std,
        }
    )
    df.to_csv('{}/{}_learning_metrices.csv'.format(outdir, outfile), index=False, header=True)
    del df

    plt.savefig('{}/{}_learning_curve.png'.format(outdir, outfile), dpi = 300)

    return


def main():
# CONFIGURATIONS ###############################################################
    args = get_arguments() # get command line arguments

    dummy_params = {    # parameters for dummy classifier
        'strategy': 'most_frequent'
    }

    gb_params = {   # parameters for gradient boosting classifier
        'learning_rate': 0.05,
        'max_depth' : 3,
        'n_estimators' : 300
    }

    nn_params = {   # parameters for neural network classifier
        'batch_size': args.batch_size,
        'shuffle': True
    }

    if not args.dummy and not args.gradient_boosting and not args.neural_network: # if no model was specified, use all models
        args.dummy             = True
        args.gradient_boosting = True
        args.neural_network    = True
################################################################################


# PREPROCESSING ################################################################

    dict_df_test = {}
    for e in glob.glob(args.test):
        temp = e.split('/')[-1].split('.')[0].split('_')[-1]
        dict_df_test[temp] = e

    df_train, df_test, common_index = preprocessing(    # load train and test dataframes and shared label index
        path_train = args.train,
        path_test  = dict_df_test['cold'],
        min_appearance = args.min_appearance
    )

    df_train['factors'] = df_train['factors'].apply(lambda x: normalize(x))
    df_test['factors']  = df_test['factors'].apply(lambda x: normalize(x))

    data_train = Dataset(df_train, common_index) # load train data with shared label index
    data_test  = Dataset(df_test, common_index)  # load test data with shared label index

    if args.learning_curve: # plot score of training and cross-validation over train samples for gradient boosting classifier
        plot_learning_curve(
            X = data_train.X,
            y = data_train.labels,
            num_jobs = args.num_jobs,
            params = gb_params,
            outdir = args.output_dir,
            outfile = args.output_filename
        )
        sys.exit(0)

    list_test_sets = []
    if args.coverage_split:
        list_test_sets = get_test_data_by_coverage(data_test, common_index)
        del data_test
    else:   # if no split wanted
        list_test_sets.append(data_test)    # append list of test sets only by data_test (to be able to iterate over this list no matter if coverage split is wanted or not)
################################################################################


# SCIKIT #######################################################################
    clf_models = []
    clf_labels = []

    if args.dummy:
        dummy_clf = init_and_train_dummy_classifier(    # initialize and train dummy classifier
            params  = dummy_params,
            X_train = data_train.X,
            y_train = data_train.labels
        )
        clf_models.append(dummy_clf)
        clf_labels.append('most_frequent')

    if args.gradient_boosting:
        gb_clf    = init_and_train_gradient_boosting_classifier(    # initialize and train gb classifier
            params  = gb_params,
            X_train = data_train.X,
            y_train = data_train.labels,
            n_jobs  = args.num_jobs
        )
        clf_models.append(gb_clf)
        clf_labels.append('gradient_boosting')
################################################################################


# PYTORCH ######################################################################
    if args.neural_network:
        partition = [0.6, 0.4]  # set size of train and dev set (test set size is already given)
        N = [int(round(i*len(data_train))) for i in partition]
        assert sum(N) == len(data_train)    # check

        # Create partition of dataset
        seed = torch.Generator().manual_seed(42)    # generate random numbers for weight initialization
        train, dev = random_split(data_train, N, generator=seed)
        # "Batchify" the data -- remember: Learning happens in between batches, ie
        # during backprop, so we should fit as many samples in a batch as our GPU RAM lets us.

        train_ = DataLoader(train, **nn_params)    # provide an iterable over the train set
        dev_ = DataLoader(dev, **nn_params)    # provide an iterable over the dev set
        list_tests_ = [DataLoader(x, **nn_params) for x in list_test_sets]   # provide an iterable over each test set

        # model initialization
        dim = data_train.X.shape[1] # number of features from input data, defines the size of the input layer
        nn_clf = nn.Sequential(
            nn.Linear(dim, 10), # define size of input and last hidden layer (number of in_features, number of out_features)
            nn.Tanh(),    #nn.PReLU(),    #nn.Sigmoid(),    #nn.ReLU(),  # activation function to add nonlinearity
            nn.Dropout(0.1),    # randomly drop x % of weights to avoid overfitting
            nn.Linear(10, 1))   # define output layer, project output into single number (number of out_features, number of outputs)

        optimizer = optim.Adam(nn_clf.parameters(), lr=1e-6)
        training_loop(
            num_epochs  = args.num_epochs,
            optimizer   = optimizer,
            model       = nn_clf,
            loss_fn     = nn.BCEWithLogitsLoss(),
            trainloader = train_
        )

        nn_clf.eval()    # turn on evaluation mode
################################################################################


# EVALUATION ###################################################################
    data_results = []   # universal output list, to save to csv [model, cov, acc, f1, rec, true label, pred label]

    if args.dummy or args.gradient_boosting:
        for test_set in list_test_sets:
            for model, name in zip(clf_models, clf_labels): # iterate over model and corresponding name
                true = test_set.labels
                pred = model.predict(test_set.X)
                acc, f1, rec, kappa = get_metrices(true, pred)
                
                if args.coverage_split:
                    cov = test_set.coverages[1]
                else:
                    cov = 'all'
                data_results.append([name, cov, acc, f1, rec, kappa, [common_index[x] for x in true], [common_index[x] for x in pred]])

                if args.confusion_matrix:   # if confusion_matrix argument was passed
                    create_and_plot_confusion_matrix(
                        true      = true,
                        pred      = pred,
                        label_map = common_index,
                        coverage  = cov,
                        outdir    = args.output_dir,
                        outfile   = args.output_filename
                    )

    if args.neural_network:
        with torch.no_grad():   # turn off gradients computation
            for test_, test_set in zip(list_tests_, list_test_sets):
                pred = []
                true = []
                total, correct = 0, 0
                for data in test_:
                    inputs, labels = data
                    outputs = nn_clf(inputs.float())
                    labels_ = labels.type_as(outputs).unsqueeze(1)
                    prediction = torch.round(torch.sigmoid(outputs))

                    pred.extend([int(i) for e in prediction.numpy().tolist() for i in e])
                    true.extend(labels.numpy().tolist())

                    total += len(labels)
                    correct += (labels_ == prediction).sum().item()
                    #print(f'Accuracy: {round(correct / total, 4)}')

                acc, f1, rec, kappa = get_metrices(true, pred)

                if args.coverage_split:
                    cov = test_set.coverages[1]
                else:
                    cov = 'all'
                data_results.append(['neural_network', cov, acc, f1, rec, kappa, [common_index[x] for x in true], [common_index[x] for x in pred]])

                if args.confusion_matrix:   # if confusion_matrix argument was passed
                    create_and_plot_confusion_matrix(
                        true      = true,
                        pred      = pred,
                        label_map = common_index,
                        coverage  = cov,
                        outdir    = args.output_dir,
                        outfile   = args.output_filename
                    )

                if args.classification_report:
                    clf_report = classification_report(
                        y_true = true, 
                        y_pred = pred,
                        labels = list(common_index.values()),
                        target_names = list(common_index.values())
                    )

    df_results = pd.DataFrame(data_results, columns=['model', 'coverage', 'accuracy', 'f1', 'recall', 'kappa', 'true', 'pred'])
    df_results.to_csv('{}/{}_classifier_results.csv'.format(args.output_dir, args.output_filename), index=False)
################################################################################


if __name__ == "__main__":

    main()