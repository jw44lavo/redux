R E D U X
==========
Bacterial phenotype prediction from low pass sequencing using recommender systems and machine learning classifiers.

See [redux](https://osf.io/uxpv4/) at OSF to get the data.


# Abstract
*Streptococcus pneumoniae* is known for causing pneumonia, meningitis and sepsis. This bacterium develops different capsular phenotypes, so-called serotypes. Modern vaccines tackle the most common serotypes and therefore need regular adjustment to currently dominant capsule types. Hence, it is important for the public health sector, to monitor serotype frequencies by the determination of phenotypes in clinical isolates. Standard methods use antibody specificity to distinguish capsule types, which is arduous and expensive. Whole genome sequencing is an alternative approach, wich could automate capsular typing. But it currently requires high sequencing depths, which makes a routine application in diagnostics too expensive. If the sequencing depth could be reduced, time and money would be saved. Thus, the challenge of this work was to predict pneumococcal serotypes from low-pass sequencing data. We aimed for implementing a method based on genomic data, which is capable of replacing the Quellung reaction. Therefore, low-pass sequencing was simulated by subsampling from published sequencing data. The data was applied to train a recommender system and a machine learning classifier. Test data was applied to complete incomplete data and to predict pneumococcal serotypes from genomic data. This process resulted in a prediction accuracy of up to 78 %, what is still to low to replace established methods in diagnostics. But this work demonstrates further steps in the direction of data-driven analyses and predictive science.  


<p align="center">
  <img src="https://github.com/jw44lavo/redux/blob/main/workflow_git.png" alt="Workflow visualization" width="350" />
</p>



# Getting started

## Scripts and Dependencies
```
git clone https://github.com/jw44lavo/redux.git
cd redux
conda env create -f redux.yml
conda activate redux
```

## Data
```
mkdir -p data && cd data
wget https://osf.io/da95s/download -O metadata.csv
sed -E 's/("([^"]*)")?,/\2\t/g' metadata.csv | cut -f4 | tail -n +2 > read_files \
&& sed -E 's/("([^"]*)")?,/\2\t/g' metadata.csv | cut -f5 | tail -n +2 >> read_files
sort read_files > read_files.txt && rm read_files
wget -i read_files.txt -nc
```
