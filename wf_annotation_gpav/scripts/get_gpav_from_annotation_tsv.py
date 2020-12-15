'''
This script builds a gene presence absence vector.

input:
    Pfam-A.hmm.dat
    tsv file from annotation
'''

import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='')

    input_args = parser.add_argument_group('Input')
    input_args.add_argument(
                            '--input',
                            required=True,
                            type=str,
                            help='path to input file'
                            )
    input_args.add_argument(
                            '--pfam',
                            required=True,
                            type=str,
                            help='path to pfam dat file'
                            )

    output_args = parser.add_argument_group('Output')
    output_args.add_argument(
                            '--output',
                            metavar='',
                            type=str,
                            help='path and name of output file'
                            )


    return parser.parse_args()


def load_vocabulary(dat):
    gpav = {}
    with open(dat, 'r') as f:
        for line in f:
            if line.startswith('#=GF AC'):
                accession_number = line.split()[2]
                gpav[accession_number] = 0

    return gpav


def fill_vector(inp, gpav):
    annotated_proteins = []
    with open(inp, 'r') as f:
        for line in f:
            prot = line.split()[0]
            annotated_proteins.append(prot)

    annotated_proteins = set(annotated_proteins)
    for prot in annotated_proteins:
        gpav[prot] = 1

    return gpav


def write_gpav_to_output(out, gpav):
    res = ''
    for elem in gpav:
        res += '{}\t{}\n'.format(elem, gpav[elem])
    
    with open(out, 'w') as o:
        o.write(res)

    return


def main():
    #GET COMMAND LINE ARGUMENTS
    args = get_arguments()

    #PREPARE GPAV, LOAD EVERY PROTEIN IN PFAM AND SET EVERY VALUE TO 0
    gene_presence_absence_vector = load_vocabulary(
        args.pfam
    )

    #FILL GPAV WITH 1 FOR FOUND PROTEINS FROM INPUT TSV FILE
    gene_presence_absence_vector = fill_vector(
        args.input,
        gene_presence_absence_vector
    )

    #WRITE GPAV OUTPUT FILE
    write_gpav_to_output(
        args.output,
        gene_presence_absence_vector
    )

    print('done')



if __name__ == '__main__':

    main()
