'''
This script splits the whole sample set in train and test set.
Ranom choice of samples, x % test set
'''

import os
import sys
import glob
import random
import argparse
import pandas as pd


def get_arguments():
    '''
    get specified command line arguments and define default parameters
    '''
    parser = argparse.ArgumentParser(description='This script splits the whole sample set in train and test set.')
    input_args = parser.add_argument_group('Input')
    input_args.add_argument(
        '--input',
        nargs='+', # at least one argument
        type=str,
    )

    input_args.add_argument(
        '--test_proportion',
        type=float,
    )

    return parser.parse_args()


def main():
    args = get_arguments()  # get command line arguments

    all_samples = set([x.split('/')[-1].split('.')[0].split('_')[0] for x in args.input])   # get unique sample names from paths
    number_of_test_samples = int(len(all_samples)*args.test_proportion) # get the number of wanted test files from all_samples and args.test_proportion
    test_samples  = random.sample(all_samples, number_of_test_samples)    # get n random samples from all_samples
    train_samples = [x for x in all_samples if x not in test_samples] # get remaining samples from all_samples

    pd.DataFrame(train_samples).to_csv('train.csv',index=False, header=False)
    pd.DataFrame(test_samples).to_csv('test.csv', index=False, header=False)

if __name__ == '__main__':

    main()
