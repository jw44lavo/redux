'''
This script returns a tsv file with pairs of assemblies, 
which have a jaccard index over a specific threshold (>= 0.95).
assembly 1  \t  assembly 2
...

input:
    csv file (matrix from sourmash compare all vs all)

output:
    tsv file (pairs of assemblies)
'''

import sys
import numpy as np

input_file  = sys.argv[1]
threshold   = float(sys.argv[2])
output_file = 'similar_pairs.tsv'


'''
extract the first line of the input file to get the labels/header
'''
labels = [x.split('.')[0] for x in open(input_file, 'r').readline().strip().split(',')]


'''
import the input csv file into a numpy array and return all pairs of indices with values >= threshold
'''
pairs = np.argwhere(np.loadtxt(open(input_file, "rb"), delimiter=",", skiprows=1) >= threshold)


'''
write the resulting label pairs to an output file but exclude the diagonal values of 1.0
(cuz these are comparisons of the same data)
label pairs appear twice in the output file (cuz the numpy array is an symetrical matrix)
'''
result = 'threshold jaccard index: {}\n'.format(threshold)
for pair in pairs:
    if pair[0] == pair[1]:
        continue
    else:
        result += '{}\t{}\n'.format(labels[pair[0]], labels[pair[1]])

open(output_file, 'w').write(result)
