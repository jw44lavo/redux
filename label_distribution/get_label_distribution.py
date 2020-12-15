'''
This script takes metadata csv files before and after preprocessing to return csv files where each label gets a count.
Output csv file for plotting with ggplot2
'''

import pandas as pd

df_before = pd.read_csv('workflow/metadata/metadata.csv')
df_after  = pd.read_csv('workflow/metadata/metadata_preprocessed.csv')

df_before['Serotype'] = df_before['Serotype'].apply(lambda x: x.split(',')[0])

df_before['Serotype'].value_counts().reset_index().to_csv('label_distribution_before_preprocessing.csv', index=False, header=['Serotype', 'Count'])
df_after['Serotype'].value_counts().reset_index().to_csv('label_distribution_after_preprocessing.csv', index=False, header=['Serotype', 'Count'])

