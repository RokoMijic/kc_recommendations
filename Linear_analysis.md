# Linear Analysis

```python
install_packages=True
```

```python
if install_packages:
    !pip install --upgrade numpy
    !pip install --upgrade numba  
    !pip install --upgrade swifter
    !pip install --upgrade scikit-learn
    !pip install --upgrade scipy
    !pip install --upgrade pandas
    !pip install --upgrade uncertainties
    !pip install --upgrade s3fs
    !pip install --upgrade joblib
    !pip install matplotlib
    !pip install  'more_itertools'
    !pip install git+https://github.com/gbolmier/funk-svd
    !pip install tqdm
    !pip install matplotlib
    !pip install seaborn
```

-------------------------------

```python
import numpy as np
import pandas as pd

import seaborn as sns


from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, coo_matrix

from funk_svd import SVD

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD

from matplotlib import pyplot as plt

import swifter
import re

```

```python
import experiment_runner
import z_utilities
import importlib
importlib.reload(experiment_runner)
importlib.reload(z_utilities)
from experiment_runner import run_experiments
from z_utilities import stack_dfs_common_column, df_to_matix
```

------------------------------------------------

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 150)
```

--------------------------------------------------


### Load data 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
df_ratings = pd.read_csv(          filepath_or_buffer = 's3://{}/{}'.format(bucket, 'klascement_ratings_05_filtered10.csv'), 
                                              dtype  = {
                                                        'res_cid': 'int32', 
                                                        'user_cid': 'int32', 
                                                        'eng_score': 'int8', 
                                                       }
                        )
```

```python
filenames =  [ 'users_courses' ,  'users_edutypes' ,   'resources_courses'  ,   'resources_edutypes'  ,   'resources_keywords'   ,
               'course_desc' ,  'educ_type_desc' , 'keywords_desc',  'user_cids'     ,  'res_cids'       ,   'resources_title'    ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    
    if 'desc' or 'title' in df_name  :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8') 
    else                             :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 

    
users_courses_df        =   dfs['users_courses']  
users_edutypes_df       =   dfs['users_edutypes']   
resources_courses_df    =   dfs['resources_courses']   
resources_edutypes_df   =   dfs['resources_edutypes'] 
resources_keywords_df   =   dfs['resources_keywords'] 
course_desc_df          =   dfs['course_desc'] 
edutype_desc_df         =   dfs['educ_type_desc'] 
keywords_desc_df        =   dfs['keywords_desc']
user_cids_df            =   dfs['user_cids']  
res_cids_df             =   dfs['res_cids']   
res_title_df            =   dfs['resources_title'] 

course_desc_df.set_index('course_id', inplace=True, drop=False)
edutype_desc_df.set_index('edutype_id', inplace=True, drop=False)


```

```python
resources_keywords_df
```

-----------------------------

```python
df_ratings_small = df_ratings.sample(n=500000).reset_index(inplace=False, drop=True)
df_ratings_med   = df_ratings.sample(n=1500000).reset_index(inplace=False, drop=True)
df_ratings_large = df_ratings.sample(n=7000000).reset_index(inplace=False, drop=True)
df_ratings_full  = df_ratings.sample(frac = 1.0, random_state=4).reset_index(inplace=False, drop=True)

print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
print('df_ratings_med     = {:,}'.format( int( float ('%.3g' % df_ratings_med.shape[0]    ) ) )   )
print('df_ratings_large   = {:,}'.format( int( float ('%.3g' % df_ratings_large.shape[0]  ) ) )   )
print('df_ratings_full    = {:,}'.format( int( float ('%.3g' % df_ratings.shape[0]        ) ) )   )
```

```python
create_fake_df = False
```

```python
if create_fake_df:
    poss_vals_probs = pd.DataFrame( {'prob': df_ratings.groupby('eng_score')['eng_score'].count()} ).reset_index(drop=False)
    poss_vals_probs['prob'] /= df_ratings.shape[0]
    eng_vals  = poss_vals_probs['eng_score'].values.tolist()
    eng_probs = poss_vals_probs['prob'].values.tolist()
    df_ratings_fake = df_ratings.copy()
    df_ratings_fake['eng_score'] = np.random.choice(a=eng_vals, size=df_ratings_fake.shape[0], replace=True, p=eng_probs)
    display(df_ratings_fake.head(n=2))
```

------------------------------------------------------


###### Define train test split split function

```python
def train_test_split_aug(df, test_size):
    train, test = train_test_split(df, test_size=test_size)
    
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    assert(test.shape[0] >= 1 )
    assert(train.shape[0] >= 1 )
    return train, test 
```

------------------------------------------------


### Funk SVD Algorithm

```python
def funk_algorithm(dataset, metrics_dict, learning_rate, regularization, n_epochs, n_factors, u_id='user_cid', i_id='res_cid', rating='eng_score', return_model=False, test_size=0.1 ):
    
    rn_dataset_df = dataset.rename(columns = {u_id:'u_id', i_id:'i_id', rating:'rating' } )
    if not return_model: 
        train_df, test_df =  train_test_split_aug(rn_dataset_df, test_size=test_size)
    else : 
        _, test_df        =  train_test_split_aug(rn_dataset_df, test_size=test_size)
        train_df          =  rn_dataset_df
    
    svd = SVD(learning_rate=learning_rate, regularization=regularization, n_epochs=n_epochs, n_factors=n_factors, min_rating=0, max_rating=5)
    svd.fit(X=train_df,  early_stopping=False,  shuffle=False)
    test_pred = svd.predict(test_df )
    
    metrics_res = {  metric_name : metric_fn(test_pred, test_df['rating'])   for   metric_name, metric_fn in metrics_dict.items()  }  
    
    if return_model : return {'svd': svd, 'train_df': train_df.rename(columns = {'u_id':u_id, 'i_id':i_id, 'rating':rating } ),  'metrics_res': metrics_res}
    else            : return metrics_res   
```

---------------------------------------------------------------------------------


### Add embedding dimensions to data

```python
def create_embedded_df(df_in, test_size=0.05 ):
    learning_rate = 0.005   
    regularization = 0.0  
    n_epochs = 60   
    n_factors = 3   
    
    assert 0.0 < test_size < 1.0
    
    df = df_in.copy()
    
    #  ------------------------------------
    
    provisional_test_size = 1.1*test_size
    
    df['initial_testset'] = np.random.choice([0,1], size=df.shape[0], p=[1.0 - provisional_test_size, provisional_test_size ]).astype('bool')
    testtable_users_df = pd.DataFrame(  {'can_test_user'  :  df.groupby('user_cid')['initial_testset'].agg(lambda x: not(all(x)) )}   ).reset_index(drop=False)
    testable_res_df    = pd.DataFrame(  {'can_test_res'   :  df.groupby('res_cid' )['initial_testset'].agg(lambda x: not(all(x)) )}   ).reset_index(drop=False)
    df = df.merge(testtable_users_df, on='user_cid', how='left').merge(testable_res_df, on='res_cid', how='left')
    df['testset'] = df['initial_testset'] & df['can_test_user'] & df['can_test_res']
    del df['initial_testset'] 
    del df['can_test_user'] 
    del df['can_test_res'] 
    
    #  ------------------------------------
    
    mr_dic    =    funk_algorithm(dataset=df[df['testset'] == False]          ,
                                  metrics_dict={   'mae'  :  mean_absolute_error   }  ,
                                  learning_rate=learning_rate                         ,
                                  regularization=regularization                       , 
                                  n_epochs=n_epochs                                   ,
                                  n_factors=n_factors                                 , 
                                  return_model=True,                                  ) 
    

    metrics_res = mr_dic['metrics_res']
    print(f'metric_result = {metrics_res}')
    
    model = mr_dic['svd']
    
    assert n_factors == model.pu.shape[1] == model.qi.shape[1]
    
    users_emb_df = pd.DataFrame({**{'user_cid':list(model.user_dict.keys())} , **{'f_svd_user' + str(i) :  model.pu[:,i].astype('float32') for i in range(n_factors) },   **{'f_bias_user': (model.bu).astype('float32')} })
    res_emb_df = pd.DataFrame({**{'res_cid' :list(model.item_dict.keys())} , **{'f_svd_res' + str(i)  :  model.qi[:,i].astype('float32') for i in range(n_factors) },   **{'f_bias_res': (model.bi).astype('float32')} })
    
    df = df.merge(users_emb_df, on='user_cid', how='left').merge(res_emb_df, on='res_cid', how='left')
    
    for i in range(n_factors): df['f_svd_user'+str(i)+'_x_res'+str(i)] = df['f_svd_user'+str(i)]*df['f_svd_res'+str(i)]
            
    df = (lambda d, cols_2_end : d.reindex(columns=[c for c in d.columns.tolist() if c not in cols_2_end] + cols_2_end)) (df, ['f_bias_user', 'f_bias_res', 'eng_score'])
    
    return df
```

-----------------------------------------------

```python
use_full_dataset_for_aug = True
```

-----------------------------------------------

```python
if use_full_dataset_for_aug: 
    augmented_df = create_embedded_df(df_ratings_full)
else:
    augmented_df = create_embedded_df(df_ratings_small)
```

```python
augmented_df = augmented_df.sort_values(by=['res_cid', 'user_cid'], axis=0, inplace=False).reset_index(inplace=False, drop=True).set_index(['res_cid', 'user_cid'])
```

```python
augmented_df.head(n=3)
```

# Save the Dataframe with useful features

```python
def create_features_df(u_r):
    assert u_r in ['user', 'res']
    r_u = 'user' if u_r=='res' else 'res'
    
    return augmented_df.filter(regex='(f_svd_' + u_r + '[0-9]*$|f_bias_' + u_r + ')',   axis=1).reset_index(drop=False).copy()\
                                                                                               .drop(columns=[r_u + '_cid'])\
                                                                                               .drop_duplicates(subset=[u_r + '_cid'], ignore_index = True)\
                                                                                               .sort_values(  by=[u_r + '_cid'], axis=0, ascending=True )\
                                                                                               .reset_index(drop=True)\
                                                                                               .astype({u_r + '_cid':'int32'})
```

```python
features_dfs = {}
```

```python
features_dfs['user'] = create_features_df('user')
```

```python
features_dfs['res'] = create_features_df('res')
```

```python
display(features_dfs['user'].head(n=2))
```

```python
display(features_dfs['res'].head(n=2))
```

```python
save_features_dfs=False
```

```python
if save_features_dfs:
    for u_r in ['user', 'res']:
        filename = 'klascement_'+ u_r +'_features_df'
        data_location = 's3://{}/{}'.format(bucket, filename)
        features_dfs[u_r].to_csv(data_location + '.csv', encoding='utf-8', index=False, float_format='%.5f') 
else:
    print("nothing saved, since save_features_dfs = False")
```

```python

```

```python
# assert False
```

```python

```

### Train a linear model on the embedding features

```python
def train_linear_model(df_data, feature_regex):
    
    print('predicting with the following features:::   ' + str( df_data.head(n=0).filter(regex=feature_regex, axis=1).columns.tolist() )     )

    def feats(df): return df.filter(regex=feature_regex, axis=1)
    def targ(df): return df['eng_score'].ravel()
    
    train_df_X, train_y, test_df_X, test_y =   (lambda tr, te:  (  feats(tr) , targ(tr), feats(te) , targ(te)  )   )    (    *( lambda d: ( d[d['testset']==False] , d[d['testset']==True] ) )( df_data )   )
    
    ols=linear_model.LinearRegression(n_jobs=-1)
    
    ols.fit(X=train_df_X, y=train_y)
    
    pred_y = ols.predict( test_df_X )
    
    return  {  'mean_absolute_error' : mean_absolute_error(pred_y, test_y) , 'model' : ols    }     
```

```python
%%time
train_linear_model(augmented_df, feature_regex='f_bias_res.*')
```

```python
%%time
train_linear_model(augmented_df, feature_regex='f_bias.*')
```

```python
%%time
res_best = train_linear_model(augmented_df, feature_regex='f_(bias|svd.*_x_.*)')
print(res_best['mean_absolute_error'])
```

```python
best_model = res_best['model']
```

```python
best_model_coeffs_dict = dict(  zip(  augmented_df.head(n=0).filter(regex='f_(bias|svd.*_x_.*)', axis=1).columns.tolist(), best_model.coef_.tolist()    )  ) 
```

```python
best_model_coeffs_dict
```

```python
best_model.intercept_
```
