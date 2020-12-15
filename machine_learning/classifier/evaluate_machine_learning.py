import ast
import glob
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

def get_kappa(x, y):
    x = ast.literal_eval(x)
    x = [n.strip() for n in x]
    y = ast.literal_eval(y)
    y = [n.strip() for n in y]

    return cohen_kappa_score(x,y)

def get_f1(x,y):
    x = ast.literal_eval(x)
    x = [n.strip() for n in x]
    y = ast.literal_eval(y)
    y = [n.strip() for n in y]

    return f1_score(x,y, average='macro')

inputs_ = glob.glob('workflow/mlearning/*_classifier_results.csv')

#for input_ in inputs_:
#    print(input_)

list_of_df = []
for input_ in inputs_:
    min_appearance = input_.split('/')[-1].split('_')[1]
    df_ = pd.read_csv(input_)
    df_['min_appearance'] = min_appearance

    list_of_df.append(df_)

df = pd.concat(list_of_df)

df = df.drop(df[df['model'] == 'neural_network'].index) # drop neural network results

df['kappa'] = df.apply(lambda x: get_kappa(x['true'], x['pred']), axis=1)
df['f1 (macro)'] = df.apply(lambda x: get_f1(x['true'], x['pred']), axis=1)

df_2 = pd.DataFrame(
    {
    'min_appearance':[0, 50, 100, 200],
    'classes':[82,31,17,8]
    }
)

df['min_appearance'] = df['min_appearance'].astype(int) 
df = pd.merge(df, df_2, on='min_appearance')

df.loc[df['coverage'] == 'full', 'coverage'] = 1000.0
df['accuracy'] = df['accuracy'].apply(lambda x: round(x, 3))
df['f1 (macro)'] = df['f1 (macro)'].apply(lambda x: round(x, 3))
df['kappa'] = df['kappa'].apply(lambda x: round(x, 3))

df['coverage'] = df['coverage'].astype(float) 

df = df.sort_values(['min_appearance', 'coverage'])
df.loc[df['coverage'] == 1000.0, 'coverage'] = 'full'

df.to_csv('evaluation_for_osf.csv', index = False, header = True, columns = ['model', 'min_appearance', 'classes', 'coverage', 'accuracy', 'f1 (macro)', 'kappa', 'true', 'pred'], sep = ',')