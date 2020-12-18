R E D U X
==========
Bacterial phenotype prediction from low pass sequencing using recommender systems and machine learning classifiers.

See [redux](https://osf.io/uxpv4/) at OSF to get the data.


# Abstract
[...]



<p align="center">
  <img src="https://github.com/jw44lavo/redux/blob/main/workflow_git.png" alt="Workflow visualization" width="250" />
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
