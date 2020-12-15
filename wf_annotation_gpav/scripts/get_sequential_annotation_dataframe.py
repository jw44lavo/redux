import sys
import pandas as pd

def main():
    input_files = sys.argv[1:]  # get all inputs from the command line into a list
    data = []   # initialize a data list
    for input_file in input_files:  # iterate over every input file
        count = open(input_file, 'r').readline().rstrip() # get the first line of the file, wich is the count
        file_name = input_file.split('/')[-1]

        if file_name[0].isdigit():
            cover = file_name.split('_')[0] # get the coverage from file name
            annot = file_name.split('_')[2] # get the used annotation workflow from file name
        else:
            cover = 'full'
            annot = file_name.split('_')[1] # get the used annotation workflow from file name

        data.append([annot, cover, count])  # append data list by extracted informations

    df = pd.DataFrame(data, columns=['annotation', 'coverage', 'count'])    # create dataframe from data list with specified column names
    df = df[df['count'] != 0]   # remove sample, if count is 0
    df.to_csv('annotation_count.csv', index=False)   # save dataframe to csv file


if __name__ == '__main__':

    main()