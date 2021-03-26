# Basic analysis of KlasCement Data

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
display(df_nodate.head(n=3))
print(df_nodate.shape)
```

```python
def show_interaction_count(colname, color):
    interact_counts_df = pd.DataFrame(  {'interaction_count': df_nodate.groupby(colname)[colname].count()}  ).reset_index(drop=False).sort_values(by='interaction_count')
    interact_counts_df['log_interaction_count'] = interact_counts_df['interaction_count'].swifter.apply(math.log10)
    display(interact_counts_df.head(n=1))
    interact_counts_df.hist(column = ['log_interaction_count'], color=color, bins=100 )
    return interact_counts_df
```

```python
res_interact_counts_df = show_interaction_count('res_cid', 'darkred')
```

```python
user_interact_counts_df = show_interaction_count('user_cid', 'blue')
```

```python
minimum_interactions_threshold = 10
```

```python
users_gr_thr = user_interact_counts_df[user_interact_counts_df['interaction_count'] >= minimum_interactions_threshold].reset_index(drop=True)[['user_cid']]
users_less_thr = user_interact_counts_df[user_interact_counts_df['interaction_count'] < minimum_interactions_threshold].reset_index(drop=True)[['user_cid']]
display(users_less_thr.head(n=2))
print(users_less_thr.shape[0])
```

```python
res_gr_thr = res_interact_counts_df[res_interact_counts_df['interaction_count'] >= minimum_interactions_threshold].reset_index(drop=True)[['res_cid']]
res_less_thr = res_interact_counts_df[res_interact_counts_df['interaction_count'] < minimum_interactions_threshold].reset_index(drop=True)[['res_cid']]
display(res_less_thr.head(n=2))
print(res_less_thr.shape[0])

```

###### Exclude all users or resources with less than 10 interactions 

```python
print(df_nodate.shape)
df_nodate = df_nodate.merge(res_gr_thr, on='res_cid', how='inner')
print(df_nodate.shape)
df_nodate = df_nodate.merge(users_gr_thr, on='user_cid', how='inner')
print(df_nodate.shape)
display(df_nodate.head(n=3))
```

```python
_ = show_interaction_count('user_cid', 'blue')
```

```python

```

----------------------------------------------------------


##### Begin the analysis

```python
df_nodate['event|goodscore'] =  df_nodate['score'].swifter.apply(lambda x: 1 if x >= 4 else 0).astype('int8')
```

```python
df_nodate['event|badscore'] =  df_nodate['score'].swifter.apply(lambda x: 1 if  x == 1 or x == 2 else 0).astype('int8')
```

```python
# del df_nodate['event|used'] 
```

```python
df_nodate.head()
```

```python
df_nodate
```

```python
corr_fr = df_nodate[['event|favourited','score','event|clicked_through','event|previewed','event|downloaded','event|used','event|visited_detail_pg']].corr()
```

```python
def corrplot_of_corrdfr(corrdfr, cmap = sns.diverging_palette(20, 220, n=256)):
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap( corrdfr,  center=0, cmap=cmap, square=True, annot=True  )
    ax.set_xticklabels( ax.get_xticklabels(), rotation=45, horizontalalignment='right');
```

```python
corrplot_of_corrdfr(corr_fr)
```

```python
sums  = df_nodate.loc[:,df_nodate.columns.str.startswith("event|")].astype(bool).sum(axis=0).astype("int64")
infos = sums/(np.max(df_nodate['res_cid']).astype("int64")*np.max(df_nodate['user_cid']).astype("int64"))

#TODO: there are some magic numbers in here 
infos = infos.apply(lambda x: np.round(-7-math.log2(x), 1) )

infos['event|badscore'] = -1*infos['event|badscore'] 

info_weight_dict = infos.to_dict()
info_weight_dict.items()

```

```python
df_nodate['eng_score'] =  df_nodate.swifter.apply(lambda x:   sum (  [  w*x[c] for c , w in info_weight_dict.items() ] )    , axis=1 ).astype('float16')
```

```python
df_nodate['eng_score'] = df_nodate['eng_score'].swifter.apply(lambda x: max(x, 0) ).astype('float16')
```

```python
scaler = StandardScaler(with_mean=False)
df_nodate['eng_score']  = scaler.fit_transform( df_nodate[['eng_score']]   ).astype('float16')
df_nodate['eng_score'] = df_nodate['eng_score'].swifter.apply(lambda x: round(x, 0) ).astype('int8')

# Here we limit the score to 5 in oder to avoid a long tail of scores between 5 and 10
df_nodate['eng_score'] = df_nodate['eng_score'].swifter.apply(lambda x: min(x, 5) ).astype('int8')

```

```python
df_nodate.hist(column = ['eng_score'], bins=20)
```

```python

```

```python
df_nodate.sample(n=50)
```

```python
n_resources =  np.max(df_nodate['res_cid']).astype('int64') 
n_users     =  np.max(df_nodate['user_cid']).astype('int64')

print('n_resources          ≈ {:,}'.format( int( float ('%.3g' % n_resources     ) ) )   )
print('n_users              ≈ {:,}'.format( int( float ('%.3g' % n_users     ) ) )   )

poss_interactions = n_resources * n_users
print('poss_interactions    ≈ {:,}'.format( int( float ('%.3g' % poss_interactions     ) ) )   )

actual_interactions = df_nodate.shape[0]
print('actual_interactions  ≈ {:,}'.format( int( float ('%.3g' % actual_interactions     ) ) )   )
```

```python

```

```python
df_rating = df_nodate[['res_cid' , 'user_cid' , 'eng_score' ]]
```

```python
df_rating
```

```python
save_rating_data = True
```

```python
if save_rating_data:
    filename = 'klascement_ratings_05_filtered10'
    data_location = 's3://{}/{}'.format(bucket, filename)

    df_rating.to_csv(data_location + '.csv', encoding='utf-8', index=False,  float_format='%.1f') 
else:
    print("nothing saved, since save_rating_data = False")
```

```python

```

```python
# df_nodate['event|usednd'] =  df_nodate.swifter.apply(lambda x:   1 if  x['event|used'] == 1 and x['event|downloaded'] == 0 else 0    , axis=1 ).astype('int8')
```

```python
# def memuse_per_row(frame):
    
#     if('dask' in str(type(frame)) ): 
#         with ProgressBar():
#             df = frame.compute()
#     else:
#         df = frame
    
#     nrows = df.shape[0]
#     memuse = sum(df.memory_usage() )
#     bytes_per_row = memuse/nrows
#     return bytes_per_row
```

```python

```

```python

```

```python

```

```python

```
