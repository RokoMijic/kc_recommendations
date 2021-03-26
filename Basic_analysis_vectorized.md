# Basic analysis of KlasCement Data - vectorized target variables & negative sampling

```python
!pip install --upgrade scikit-learn
!pip install --upgrade "dask[dataframe]"
!pip install --upgrade dask
!pip install --upgrade s3fs
!pip install --upgrade pandas
!pip install --upgrade swifter
!pip install --upgrade seaborn
```

```python
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import seaborn as sns

from dask.diagnostics import ProgressBar
from IPython.display import display, HTML

import math
import s3fs
import swifter

from scipy import sparse
```

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 150)
```

```python
print('The pandas version is {}.'.format(pd.__version__))
print('The dask version is {}.'.format(dask.__version__))
print('The s3fs version is {}.'.format(s3fs.__version__))
```

##### Load Data

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
df_nodate = pd.read_csv(          filepath_or_buffer = 's3://{}/{}'.format(bucket, 'klascement_no_dates_sorted.csv'), 
                                              dtype  = {
                                                        'res_cid': 'int32', 
                                                        'user_cid': 'int32', 
                                                        'event|favourited': 'int8',
                                                        'score':  'int8',
                                                        'event|clicked_through': 'int8',
                                                        'event|previewed': 'int8',
                                                        'event|downloaded': 'int8', 
                                                        'event|used': 'int8',
                                                        'event|visited_detail_pg': 'int8'  
                                                       }
                           )
```

```python
display(df_nodate.head(n=8))
print(df_nodate.shape)
```

```python
display(df_nodate.sum())
```

```python
def show_interaction_count(colname, color):
    interact_counts_df = pd.DataFrame(  {colname[:-4] + '_interac_ct': df_nodate.groupby(colname)[colname].count()}  )\
                           .reset_index(drop=False)\
                           .sort_values(by=colname[:-4] + '_interac_ct')\
                           .astype('int32')
            
    interact_counts_df[colname[:-4] + '_interac_ct_log'] = interact_counts_df[colname[:-4] + '_interac_ct'].swifter.apply(math.log10)
    display(interact_counts_df.head(n=1))
    interact_counts_df.hist(column = [colname[:-4] + '_interac_ct_log'], color=color, bins=100 )
    return interact_counts_df
```

```python
res_interact_counts_df = show_interaction_count('res_cid', 'darkred')
```

```python

```

```python
user_interact_counts_df = show_interaction_count('user_cid', 'blue')
```

```python
minimum_interactions_threshold = 10
```

```python
users_gr_thr = user_interact_counts_df[user_interact_counts_df['user_interac_ct'] >= minimum_interactions_threshold].reset_index(drop=True)
users_less_thr = user_interact_counts_df[user_interact_counts_df['user_interac_ct'] < minimum_interactions_threshold].reset_index(drop=True)
display(users_less_thr.head(n=2))
print(users_less_thr.shape[0])
```

```python
res_gr_thr = res_interact_counts_df[res_interact_counts_df['res_interac_ct'] >= minimum_interactions_threshold].reset_index(drop=True)
res_less_thr = res_interact_counts_df[res_interact_counts_df['res_interac_ct'] < minimum_interactions_threshold].reset_index(drop=True)
display(res_less_thr.head(n=2))
print(res_less_thr.shape[0])

```

###### Exclude all users or resources with less than 10 interactions 

```python
print(df_nodate.shape)
df_nodate = df_nodate.merge(users_gr_thr[['user_cid']], on='user_cid', how='inner')
print(df_nodate.shape)
df_nodate = df_nodate.merge(res_gr_thr[['res_cid']], on='res_cid', how='inner')
print(df_nodate.shape)
display(df_nodate.head(n=3))
```

```python
_ = show_interaction_count('user_cid', 'blue')
```

```python
_ = show_interaction_count('res_cid', 'red')
```

----------------------------------------------------------


##### Rename columns

```python
for col in df_nodate.columns.tolist():
    if col.startswith('event|'):
        df_nodate = df_nodate.rename(columns={col:'t_b_' + col[6:]})

df_nodate = df_nodate.rename(columns={'score':'t_i_score'})

df_nodate = (lambda d, c2s : d.reindex(columns = c2s +[c for c in d.columns.tolist() if c not in c2s]))\
             (df_nodate, ['res_cid', 'user_cid'])
```

```python
display(df_nodate.head(n=2))
print(df_nodate.shape[0])
```

```python
df_nodate.filter(regex='^(?!t_).+', axis=1).head(n=0)
```

```python
df_nodate.filter(regex='t_', axis=1).head(n=0)
```

---------------------------------------


##### Find out how many interactions there are for each column

```python
col_sums_df = pd.DataFrame(df_nodate.filter(regex='t_.*', axis=1).sum(axis=0)).rename(columns={0:'sum'})
col_sums_df['sum'] = col_sums_df['sum'].apply(  lambda n: '{:,}'.format( int( float ('%.3g' % n ) ) )      )
col_sums_df
```

```python

```

-------------------------------------------


##### Find out how many users, rescources and total interactions there are 

```python
n_resources =  np.max(df_nodate['res_cid']).astype('int64') 
n_users     =  np.max(df_nodate['user_cid']).astype('int64')

print('n_resources          ≈  {:,}'.format( int( float ('%.3g' % n_resources     ) ) )   )
print('n_users              ≈  {:,}'.format( int( float ('%.3g' % n_users     ) ) )   )

poss_interactions = n_resources * n_users
print('poss_interactions    ≈  {:,}'.format( int( float ('%.3g' % poss_interactions     ) ) )   )

actual_interactions = df_nodate.shape[0]
print('actual_interactions  ≈  {:,}'.format( int( float ('%.3g' % actual_interactions     ) ) )   )
```

--------------------------------------------------------------------------


#### Add negative data ("Negative Sampling")


Add datapoints with (user, item) pairs where there is no interaction. This is achieved by simply permuting the 'user_cid' 
column - this means we have the same distributions

```python
df_negative = df_nodate.copy()
cols_targets = df_negative.filter(regex='t_', axis=1).head(n=0).columns.tolist()

for col in cols_targets:
    df_negative[col] = 0
    df_negative[col] = df_negative[col].astype('int8')
    
df_negative['user_cid_shuff'] = np.random.RandomState(seed=0).permutation(df_negative['user_cid'].values) 
df_negative = df_negative.drop(columns='user_cid').rename(columns={'user_cid_shuff':'user_cid'})
df_negative = (lambda d, c2s : d.reindex(columns = c2s +[c for c in d.columns.tolist() if c not in c2s]))\
              (df_negative, ['res_cid', 'user_cid'])
```

```python
df_negative.head(n=2)
```

```python
df_pos_neg_nodate = pd.concat([df_nodate, df_negative], axis=0, ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
```

```python
df_pos_neg_nodate
```

```python
corr_fr = df_pos_neg_nodate[['t_b_favourited','t_i_score','t_b_clicked_through','t_b_previewed','t_b_downloaded','t_b_used','t_b_visited_detail_pg']].corr()
```

```python
def corrplot_of_corrdfr(corrdfr, cmap = sns.diverging_palette(0, 250, n=256)):
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap( corrdfr,  center=0, cmap=cmap, square=True, annot=True  )
    ax.set_xticklabels( ax.get_xticklabels(), rotation=45, horizontalalignment='right');
```

```python
corrplot_of_corrdfr(corr_fr)
```

```python

```

### Save dataframe

```python
df_to_save = df_pos_neg_nodate
```

```python
save_rating_data = False
```

```python
if save_rating_data:
    filename = 'klascement_vectorized_filtered10_negsampling'
    data_location = 's3://{}/{}'.format(bucket, filename)

    df_to_save.to_csv(data_location + '.csv', encoding='utf-8', index=False) 
else:
    print("nothing saved, since save_rating_data = False")
```

```python

```

```python

```
