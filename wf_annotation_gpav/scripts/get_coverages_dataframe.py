import sys
import pandas as pd

def main():
    input_files = sys.argv[1:]  # get all inputs from the command line into a list
    data = []   # initialize a data list
    for input_file in input_files:  # iterate over every input file
        count  = open(input_file, 'r').readline().rstrip() # get the first line of the file, wich is the count
        sample = input_file.split('/')[-1].split('_')[0]

        data.append([sample, count])  # append data list by extracted informations

    df = pd.DataFrame(data, columns=['sample', 'coverage']) # create dataframe from data list with specified column names
    df.to_csv('coverages.csv', index=False)   # save dataframe to csv file


if __name__ == '__main__':

    main()