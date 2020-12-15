'''
This script creates symlinks of fastq and assembly files to different directories
depending on the input classification file from the redundant_and_species_check-wf

input:
    classification file (sourmash lca classify)

output:
    symlinks 

ToDo:
    symlinks of fasta files (assemblies)
'''

import os
import sys


reads_dir                                           = 'reads/original'
reads_classified_streptococcus_pneumoniae           = 'reads/original/classified_streptococcus_pneumoniae'
reads_classified_streptococcus_but_no_species       = 'reads/original/classified_streptococcus_but_no_species'
reads_classified_not_streptococcus_at_all           = 'reads/original/classified_not_streptococcus_at_all'
reads_classified_streptococcus_but_not_pneumoniae   = 'reads/original/classified_streptococcus_but_not_pneumoniae'

assemblies_dir                                          = 'assemblies'
assemblies_classified_streptococcus_pneumoniae          = 'assemblies/classified_streptococcus_pneumoniae'
assemblies_classified_streptococcus_but_no_species      = 'assemblies/classified_streptococcus_but_no_species'
assemblies_classified_not_streptococcus_at_all          = 'assemblies/classified_not_streptococcus_at_all'
assemblies_classified_streptococcus_but_not_pneumoniae  = 'assemblies/classified_streptococcus_but_not_pneumoniae'


input_file  = sys.argv[1]
name        = input_file.split('_')[0]

species     = ''
with open(input_file, 'r') as f:
    classification = f.readlines()[1].strip().split(',')
    genus   = classification[7]
    species = classification[8]



'''
different try blocks for different functions needed, because a second function would not be executed in case of exception.
FastQ1 and FastQ2 should be okay, because in most cases their symlinks got created at the same time for the first time.
But keep attention to this anyways and consider to open a third try block. 
'''
try: 
    if species == 'Streptococcus pneumoniae':
        os.symlink('{}/{}.fasta'.format(assemblies_dir, name), '{}/{}.fasta'.format(assemblies_classified_streptococcus_pneumoniae, name))

    elif genus == 'Streptococcus' and species == '':
        os.symlink('{}/{}.fasta'.format(assemblies_dir, name), '{}/{}.fasta'.format(assemblies_classified_streptococcus_but_no_species, name))


    elif genus == 'Streptococcus' and species != 'Streptococcus pneumoniae':
        os.symlink('{}/{}.fasta'.format(assemblies_dir, name), '{}/{}.fasta'.format(assemblies_classified_streptococcus_but_not_pneumoniae, name))

    else:
        os.symlink('{}/{}.fasta'.format(assemblies_dir, name), '{}/{}.fasta'.format(assemblies_classified_not_streptococcus_at_all, name))

except FileExistsError:
    pass


try:
    if species == 'Streptococcus pneumoniae':
        os.symlink('{}/{}_1.fastq.gz'.format(reads_dir, name), '{}/{}_1.fastq.gz'.format(reads_classified_streptococcus_pneumoniae, name))
        os.symlink('{}/{}_2.fastq.gz'.format(reads_dir, name), '{}/{}_2.fastq.gz'.format(reads_classified_streptococcus_pneumoniae, name))

    elif genus == 'Streptococcus' and species == '':
        os.symlink('{}/{}_1.fastq.gz'.format(reads_dir, name), '{}/{}_1.fastq.gz'.format(reads_classified_streptococcus_but_no_species, name))
        os.symlink('{}/{}_2.fastq.gz'.format(reads_dir, name), '{}/{}_2.fastq.gz'.format(reads_classified_streptococcus_but_no_species, name))


    elif genus == 'Streptococcus' and species != 'Streptococcus pneumoniae':
        os.symlink('{}/{}_1.fastq.gz'.format(reads_dir, name), '{}/{}_1.fastq.gz'.format(reads_classified_streptococcus_but_not_pneumoniae, name))
        os.symlink('{}/{}_2.fastq.gz'.format(reads_dir, name), '{}/{}_2.fastq.gz'.format(reads_classified_streptococcus_but_not_pneumoniae, name))


    else:
        os.symlink('{}/{}_1.fastq.gz'.format(reads_dir, name), '{}/{}_1.fastq.gz'.format(reads_classified_not_streptococcus_at_all, name))
        os.symlink('{}/{}_2.fastq.gz'.format(reads_dir, name), '{}/{}_2.fastq.gz'.format(reads_classified_not_streptococcus_at_all, name))


except FileExistsError:
    pass



