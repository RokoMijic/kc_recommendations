---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Funk Experiments - vectorized version

```python jupyter={"outputs_hidden": true}
!pip install --upgrade numpy
!pip install --upgrade numba  
!pip install --upgrade swifter
!pip install --upgrade scikit-learn
!pip install --upgrade scipy
!pip install --upgrade pandas
!pip install --upgrade uncertainties
!pip install --upgrade s3fs
!pip install --upgrade joblib
!pip install --upgrade jupytext
!pip install  'more_itertools'
!pip install git+https://github.com/gbolmier/funk-svd
!pip install tqdm
```

```python
from joblib import Parallel, delayed
from joblib import parallel_backend

from itertools import  product
from functools import reduce

import numpy as np
import pandas as pd

import swifter

from scipy.sparse import csr_matrix, coo_matrix

import time

import importlib

from funk_svd import SVD

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import random_projection
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
```

```python
import experiment_runner
import z_utilities

importlib.reload(experiment_runner)
importlib.reload(z_utilities)
from experiment_runner import run_experiments, run_experis_fr_settings
from z_utilities import hash_df
```

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('max_colwidth', 150)
```

### Load data 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
df_ratings = pd.read_csv(          filepath_or_buffer = 's3://{}/{}'.format(bucket, 'klascement_vectorized_filtered10_negsampling.csv'), 
                                              dtype  = {
                                                        'res_cid'                  : 'int32', 
                                                        'user_cid'                 : 'int32', 
                                                        't_b_favourited'           : 'int8', 
                                                        't_i_score'                : 'int8', 
                                                        't_b_clicked_through'      : 'int8', 
                                                        't_b_previewed'            : 'int8', 
                                                        't_b_downloaded'           : 'int8', 
                                                        't_b_used'                 : 'int8', 
                                                        't_b_visited_detail_pg'    : 'int8', 
                                                       }
                        )
```

###### load noninteraction data

```python
filenames =  [ 'users_courses' ,  'users_edutypes' ,    
               'resources_courses'  ,   'resources_edutypes'  ,   'resources_keywords'   ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 

    
users_courses_df        =   dfs['users_courses']  
users_edutypes_df       =   dfs['users_edutypes']   
resources_courses_df    =   dfs['resources_courses']   
resources_edutypes_df   =   dfs['resources_edutypes'] 
resources_keywords_df   =   dfs['resources_keywords'] 
```

###### Define ways to augment dataframes:  


-------------------------------------------------------------------------------------

```python
display(df_ratings.head(n=1))
print(df_ratings.shape[0])
```

```python
res_series = df_ratings['res_cid'].drop_duplicates().reset_index(drop=True)
print(res_series.shape[0])
print( max(res_series.values.tolist())  )
```

```python
users_series = df_ratings['user_cid'].drop_duplicates().reset_index(drop=True)
print(users_series.shape[0])
print(   max(users_series.values.tolist())   )
```

----------------------------------------------------------

```python
%%time
df_ratings_micro = df_ratings.sample(n=5        , replace=True).reset_index(inplace=False, drop=True)
df_ratings_small = df_ratings.sample(n=500000   , replace=True).reset_index(inplace=False, drop=True)
df_ratings_med   = df_ratings.sample(n=10000000 , replace=True).reset_index(inplace=False, drop=True)
df_ratings_full  = df_ratings

print('df_ratings_micro   = {:,}'.format( int( float ('%.3g' % df_ratings_micro.shape[0]  ) ) )   )
print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
print('df_ratings_med     = {:,}'.format( int( float ('%.3g' % df_ratings_med.shape[0]  ) ) )   )
print('df_ratings_full    = {:,}'.format( int( float ('%.3g' % df_ratings.shape[0]        ) ) )   )
```

-----------------------------------------------------------------------


###### Define train test split split function

```python
def train_test_split_aug(df, test_size, random_state=0):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    assert(test.shape[0] >= 1 )
    assert(train.shape[0] >= 1 )
    return train, test 
```

```python

```

-----------------------------------------------------


##### Define Metrics

```python
def rel_mean_absolute_error(y_true, y_pred, mean_to_guess=None):
    if mean_to_guess is None: mean_to_guess =  np.mean(y_true)

    mae_raw   = mean_absolute_error(y_true,    y_pred)
    mae_const = mean_absolute_error(y_true,    np.full(y_pred.shape, mean_to_guess, dtype='float32')   ) 

    return mae_raw/mae_const
```

------------------------------------------------


##### Define DF selection methods

```python
def get_negative_samples(df, neg_val:int, seed=0):
    '''
    gets negative samples to a dataframe
    '''
    
    df_neg = df.copy()
    cols_targets = df_neg.filter(regex='t_', axis=1).head(n=0).columns.tolist()

    for col in cols_targets:
        df_neg[col] = neg_val
        df_neg[col] = df_neg[col].astype('int8')
    
    df_neg['user_cid_shuff'] = np.random.RandomState(seed=seed).permutation(df_neg['user_cid'].values) 
    df_neg = df_neg.drop(columns='user_cid').rename(columns={'user_cid_shuff':'user_cid'})
    df_neg = (lambda d, c2s : d.reindex(columns = c2s +[c for c in d.columns.tolist() if c not in c2s]))\
             (df_neg, ['res_cid', 'user_cid'])   
    
    return df_neg
```

```python
def select_t(df, target, seed=0, min_integer_target_val=1):
    '''
    Selects the target variable of interest
    The assumption here is that 0 means unknown values in the input dataframe, but 0 means negative in the output
    '''
    if   target.startswith('t_i')  :  neg_val=min_integer_target_val
    elif target.startswith('t_b')  :  neg_val=0

    df_pos = df[df[target] > 0 ]  [[  'user_cid' ,  'res_cid'   ,  target   ]]
    df_neg = get_negative_samples(df_pos, neg_val=neg_val, seed=seed)
    df_pos_neg  = pd.concat([df_pos, df_neg], axis=0, ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)   
        
    return df_pos_neg
```

```python

```

### Analyse with Funk

```python
def funk_algorithm(dataset, metrics_dict, learning_rate, regularization, n_epochs, n_factors, u_id='user_cid', i_id='res_cid', rating=None, test_size=0.05, dat_rnd=0):
    n_epochs  = int(n_epochs)
    n_factors = int(n_factors) 
    
    if rating==None: rating = [c for c in dataset.columns.tolist() if c.startswith('t_')][0]
    
    rn_dataset_df = dataset[[u_id, i_id, rating]].rename(columns = {u_id:'u_id', i_id:'i_id', rating:'rating' } )

    train_df, test_df =  train_test_split_aug(rn_dataset_df, test_size=test_size, random_state=dat_rnd)
    
    svd = SVD(learning_rate=learning_rate, regularization=regularization, n_epochs=n_epochs, n_factors=n_factors, min_rating=0, max_rating=5)
    svd.fit(X=train_df,  early_stopping=False,  shuffle=False)
    test_pred = np.array(svd.predict(test_df ))
    
    try:
        metrics_res = {  metric_name : metric_fn(test_df['rating'].values, test_pred)   for   metric_name, metric_fn in metrics_dict.items()  }  
    except: 
        metrics_res = {  metric_name : float("NaN")                                     for   metric_name, metric_fn in metrics_dict.items()  }
    
    
    return {'model': svd, 'metrics_res': metrics_res}          
```

```python
1+1
```

### Get fast baseline Predictions


Mean regressor -- predicts the mean rating for each item, user, or just a global mean prediction

```python
class MeanRegressor(BaseEstimator, RegressorMixin):  
    """Flexible mean regressor for dataframes with  ['u_id', 'i_id', 'rating' ] columns"""

    def __init__(self, regtype):
        if not (regtype in ['g', 'u', 'i'] ): raise ValueError(f'Bad regressor type: {regtype}; should be one of  (g, u, i)')
        self._regtype = regtype

    def fit(self, X: pd.DataFrame):
        X.reset_index(drop=True, inplace=True)
        if not (  set(X.columns.values.tolist()) == {'i_id', 'u_id', 'rating'}): raise ValueError(f'Bad columns: {X.columns.values.tolist()}; should be [i_id, u_id, rating]') 
            
        self._meanval = X['rating'].mean()
        
        if self._regtype == 'i':  self._item_mean_df = pd.DataFrame({'rating' : X.groupby('i_id')['rating'].mean()}).reset_index(drop=False)
        if self._regtype == 'u':  self._user_mean_df = pd.DataFrame({'rating' : X.groupby('u_id')['rating'].mean()}).reset_index(drop=False)    
        return self

    def predict(self, X: pd.DataFrame):
        try:
            getattr(self, "_meanval")
            if self._regtype == 'i': getattr(self, "_item_mean_df")
            if self._regtype == 'u': getattr(self, "_user_mean_df")  
                
        except AttributeError:
            raise RuntimeError("You must train the regressor before predicting data!")

        if not (X.columns.values.tolist() == ['i_id', 'u_id']): raise ValueError(f'Bad columns: {X.columns.values.tolist()}; should be [i_id, u_id]')

        if self._regtype=='g'    :  return(  np.full( shape=(X.shape[0],), fill_value=self._meanval, dtype='float32'  )                  ) 
        if self._regtype=='i'    :  return(  X.merge(self._item_mean_df, on='i_id', how='left').fillna(self._meanval)['rating'].values   )
        if self._regtype=='u'    :  return(  X.merge(self._user_mean_df, on='u_id', how='left').fillna(self._meanval)['rating'].values   )

```

```python
def make_baseline_algo(Regressor, regtype):
    ''' Makes a baseline algorithm (no parameters) out of a regressor
    '''
    
    def regressor_algo(dataset, metrics_dict, u_id='user_cid', i_id='res_cid', rating=None, test_size=0.1, dat_rnd=0):
        if rating==None: rating = [c for c in dataset.columns.tolist() if c.startswith('t_')][0]

        rn_dataset_df = dataset.rename(columns = {u_id:'u_id', i_id:'i_id', rating:'rating' } )
        train_df, test_df =  train_test_split_aug(rn_dataset_df, test_size=test_size, random_state=dat_rnd)
        
        iregressor = Regressor(regtype=regtype)
        iregressor.fit(X=train_df)
        test_pred = iregressor.predict(test_df[['i_id', 'u_id']] )
        
        try:
            metrics_res = {  metric_name : metric_fn(test_df['rating'].values, test_pred)   for   metric_name, metric_fn in metrics_dict.items()  }  
        except: 
            metrics_res = {  metric_name : float("NaN")                                     for   metric_name, metric_fn in metrics_dict.items()  }
        
        return {'model': iregressor,  'metrics_res': metrics_res}
    
    return regressor_algo
```

#### Run with care, this could take a long time

```python
top_hyperparameters_found = {}
top_mae_rel_found         = {}
```

###### Funk SVD on scores

```python
search_paramspace_scores=False
```

```python
importlib.reload(experiment_runner)
from experiment_runner import run_experiments, run_experis_fr_settings
```

```python
if search_paramspace_scores:
    res    =  run_experiments(   algo_dict   ={   'funk'             :  funk_algorithm                                    ,
                                                  'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )           ,
                                                  'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )           ,
                                                  'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )        }  ,
                                 dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_i_score' )        }  , 
                                 metrics_dict={   'mae'              :  mean_absolute_error                               ,
                                                  'mae_rel'          :  rel_mean_absolute_error                        }  , 
                                 hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03]                 ,      
                                                  'regularization'   :  [0.003, 0.01, 0.06, 0.1, 0.3, 0.6]                ,
                                                  'n_epochs'         :  reversed([25, 50, 75, 125, 200 ])                 ,
                                                  'n_factors'        :  reversed([0, 1, 5, 8])                         }  , 
                                 rchoice_tot=25                                                                           , 
                                 rchoice_hparam=21                                                                        ,
                                 n_jobs=7                                                                                 , 
                                 backend_name='loky' 
                              )

    display(res['df'])
```

```python
top_hyperparameters_found['t_i_score'] = {'learning_rate' : 0.01, 'regularization' : 0.06, 'n_epochs' : 75.0, 'n_factors' : 5.0}  
top_mae_rel_found['t_i_score']         = 0.84495
```

```python

```

###### Funk SVD on favorites

```python
search_paramspace_favorites=False
```

```python
if search_paramspace_favorites:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                    ,
                                              'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )           ,
                                              'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )           ,
                                              'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )        }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_favourited' )   }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                               ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }  , 
                             hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03, 0.1, 0.3]       ,      
                                              'regularization'   :  [0.003, 0.01, 0.06, 0.1, 0.3, 0.6]                ,
                                              'n_epochs'         :  [25, 50, 75, 125, 200 ]                           ,
                                              'n_factors'        :  [0, 2, 4, 10]                                  }  , 
                             rchoice_tot=100                                                                          , 
                             rchoice_hparam=97                                                                        ,
                             n_jobs=11                                                                                , 
                             backend_name='loky' 
                          )

    display(res['df'])
```

```python
top_hyperparameters_found['t_b_favourited'] = {'learning_rate' : 0.03, 'regularization' : 0.003, 'n_epochs' : 125, 'n_factors' : 4}  
top_mae_rel_found['t_b_favourited']         = 0.85399
```

```python

```

```python

```

###### Funk SVD on clickthroughs

```python
search_paramspace_clickthroughs=False
```

```python
top_hyperparameters_found['t_b_clicked_through'] = {'learning_rate' : 0.03, 'regularization' : 0.003, 'n_epochs' : 125, 'n_factors' : 5}  
top_mae_rel_found['t_b_clicked_through']         = 0.7800
```

```python
if search_paramspace_clickthroughs:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                      ,
                                              'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )             ,
                                              'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )             ,
                                              'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )          }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_clicked_through') }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }    , 
                             hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03, 0.1]              ,      
                                              'regularization'   :  [0.003, 0.006, 0.01, 0.06, 0.1]                     ,
                                              'n_epochs'         :  [50, 75, 125, 200 ]                                 ,
                                              'n_factors'        :  [0, 1, 2, 3, 4, 5, 10]                           }  , 
                             rchoice_tot=75                                                                             , 
                             rchoice_hparam=72                                                                          ,
                             n_jobs=11                                                                                  , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

```python

```

###### Funk SVD on previews

```python
search_paramspace_previews=False
```

```python
top_hyperparameters_found['t_b_previewed'] = {'learning_rate' : 0.02, 'regularization' : 0.01, 'n_epochs' : 50, 'n_factors' : 50}  
top_mae_rel_found['t_b_previewed']         = 0.6539
```

```python
if search_paramspace_previews:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                      ,
                                              'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )             ,
                                              'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )             ,
                                              'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )          }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_previewed')       }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }    , 
                             hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03, 0.1]              ,      
                                              'regularization'   :  [0.001, 0.003, 0.006, 0.01]                         ,
                                              'n_epochs'         :  reversed([25, 50, 75, 125, 150, 250])               ,
                                              'n_factors'        :  reversed([0, 10, 20, 35, 50])                    }  , 
                             rchoice_tot=17                                                                             , 
                             rchoice_hparam=14                                                                          ,
                             n_jobs=7                                                                                   , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

```python
if search_paramspace_previews:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                   }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_previewed')       }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }    , 
                             hyperp_dict ={   'learning_rate'    :  [0.01, 0.02, 0.03, 0.04, 0.05]                      ,      
                                              'regularization'   :  [0.01, 0.03]                                        ,
                                              'n_epochs'         :  reversed([75])                                      ,
                                              'n_factors'        :  reversed([50])                                   }  , 
                             rchoice_tot=10                                                                             , 
                             rchoice_hparam=10                                                                          ,
                             n_jobs=5                                                                                   , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

###### Funk SVD on downloads

```python
search_paramspace_previews=False
```

```python
top_hyperparameters_found['t_b_downloaded'] = {'learning_rate' : 0.02, 'regularization' : 0.003, 'n_epochs' : 50, 'n_factors' : 25}  
top_mae_rel_found['t_b_downloaded']        = 0.6448
```

```python
if search_paramspace_previews:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                      ,
                                              'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )             ,
                                              'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )             ,
                                              'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )          }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_downloaded')      }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }    , 
                             hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03, 0.1]              ,      
                                              'regularization'   :  [0.001, 0.003, 0.006, 0.01]                         ,
                                              'n_epochs'         :  reversed([50, 75, 125, 225])                        ,
                                              'n_factors'        :  reversed([0, 1, 10, 25, 32, 50])                 }  , 
                             rchoice_tot=17                                                                             , 
                             rchoice_hparam=7                                                                           ,
                             n_jobs=7                                                                                   , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

```python

```

###### Funk SVD on used

```python
search_paramspace_used=False
```

```python
top_hyperparameters_found['t_b_used'] = {'learning_rate' : 0.03, 'regularization' : 0.003, 'n_epochs' : 35, 'n_factors' : 35}  
top_mae_rel_found['t_b_used']         = 0.6289
```

```python
if search_paramspace_used:
    res = run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                      ,
                                              'ItemMean'         :  make_baseline_algo(MeanRegressor, 'i' )             ,
                                              'UserMean'         :  make_baseline_algo(MeanRegressor, 'u' )             ,
                                              'GlobalMean'       :  make_baseline_algo(MeanRegressor, 'g' )          }  ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_used')            }  , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                        }    , 
                             hyperp_dict ={   'learning_rate'    :  [0.001, 0.003, 0.006, 0.01, 0.03, 0.1]              ,      
                                              'regularization'   :  [0.001, 0.003, 0.006, 0.01]                         ,
                                              'n_epochs'         :  reversed([25, 50, 75])                              ,
                                              'n_factors'        :  reversed([0, 1, 15, 32, 50])                     }  , 
                             rchoice_tot=25                                                                             , 
                             rchoice_hparam=27                                                                          ,
                             n_jobs=11                                                                                  , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

```python

```

###### Funk SVD on visited_detail

```python
search_paramspace_visited_detail=False
```

```python
top_hyperparameters_found['t_b_visited_detail_pg'] = {'learning_rate' : 0.02, 'regularization' : 0.003, 'n_epochs' : 30, 'n_factors' : 30}  
top_mae_rel_found['t_b_visited_detail_pg']         = 0.6288
```

```python
if search_paramspace_visited_detail:
    res = run_experiments(   algo_dict   ={   'funk'             :  funk_algorithm                                    } ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_visited_detail_pg')} , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                           } , 
                             hyperp_dict ={   'learning_rate'    :  [ 0.01  ]                                           ,      
                                              'regularization'   :  [ 0.006 ]                                           ,
                                              'n_epochs'         :  reversed([50])                                      ,
                                              'n_factors'        :  reversed([1, 20, 25, 30, 40, 45 ])                } , 
                             rchoice_tot=6                                                                              , 
                             rchoice_hparam=6                                                                           ,
                             n_jobs=5                                                                                   , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

```python
if search_paramspace_visited_detail:
    res = run_experiments(   algo_dict   ={   'funk'             :  funk_algorithm                                    } ,
                             dataset_dict={   'df_ratings_full'  :  select_t(df_ratings_full, 't_b_visited_detail_pg')} , 
                             metrics_dict={   'mae'              :  mean_absolute_error                                 ,
                                              'mae_rel'          :  rel_mean_absolute_error                           } , 
                             hyperp_dict ={   'learning_rate'    :  [ 0.01  ]                                           ,      
                                              'regularization'   :  [ 0.006 ]                                           ,
                                              'n_epochs'         :  reversed([10,20,30,40,50])                          ,
                                              'n_factors'        :  reversed([40])                                    } , 
                             rchoice_tot=6                                                                              , 
                             rchoice_hparam=6                                                                           ,
                             n_jobs=5                                                                                   , 
                             backend_name='loky' 
                          )
    display(res['df'])
```

----------------------------------------------------------------------------------------


#### Create Models for the best hyperparameters for each dataset

```python
pd.DataFrame({'mae_rel':top_mae_rel_found}).sort_values(by='mae_rel', ascending=False)
```

```python
pd.DataFrame(top_hyperparameters_found).transpose() 
```

```python
set(top_hyperparameters_found.keys())
```

```python

```

```python
def make_models_all_targets(df, df_name, hyperparameters_list):
    targets=df.filter(regex='t_').columns.tolist()
    assert set(hyperparameters_list.keys()) == set(targets)
    
    
    experi_settings = [ {'dataset'      :   select_t(df, t)                                                      , 
                         'algorithm'    :   funk_algorithm                                                       , 
                         'hparams'      :   hyperparameters_list[t]                                              , 
                         'metrics_dict' : {'mae': mean_absolute_error, 'mae_rel': rel_mean_absolute_error }
                        } 
                       for t in targets
                      ]

    experi_names    = [ {'dataset'      :   'df_' + df_name + '_|_' + str(t)                                     , 
                         'algorithm'    :   'Funk'                                                               , 
                         'hparams'      :   hyperparameters_list[t]                                              , 
                         'metrics_dict' :  {'mae', 'mae_rel' }
                        } 
                       for t in targets
                      ]
    
    res = run_experis_fr_settings(experi_settings, experi_names, n_jobs=7)
    
    display(res['df'])
    
    models_list = [{'target':d['setting']['dataset'].split('_|_')[1] , 'model':d['model']} for d in res['models']]
    
    return models_list
```

```python

```

###### Expect 7 minutes to train the models

```python
# here we exclude a 10% training set, which can be used later to analyse the quqlity of the embeddings 
df_train_full, _ = train_test_split_aug(df_ratings_full, test_size=0.1, random_state=0)
```

hashes: reXXomF$, cmViUA0z

```python
print(hash_df(df_train_full))
print(hash_df(_))
```

```python

```

```python
trained_models_list = make_models_all_targets(df_train_full, 'full', top_hyperparameters_found)
```

```python

```

t_b_favourited         - 7 sec

t_i_score              - 8.7 sec

t_b_clicked_through	   - 27 sec

t_b_previewed          - 164 sec

t_b_downloaded         - 211 sec

t_b_used               - about 400 sec

t_b_visited_detail_pg  - about 400 sec

```python
trained_models_list
```

# Save Dataframes with user/item features

```python
users_df  =  df_train_full[['user_cid']].drop_duplicates().sort_values(by=['user_cid']).reset_index(drop=True)
res_df    =  df_train_full[['res_cid' ]].drop_duplicates().sort_values(by=['res_cid' ]).reset_index(drop=True) 
```

```python
targets=df_ratings_full.filter(regex='t_').columns.tolist()
targets

users_all_emb_df = users_df.copy()
res_all_emb_df   = res_df.copy()

for t in targets: 

    model = [d for d in trained_models_list if d['target']==t][0]['model']

    assert model.pu.shape[1] == model.qi.shape[1]
    n_factors = model.pu.shape[1]

    users_emb_df = pd.DataFrame({**{'user_cid':list(model.user_dict.keys())} , **{'f__' + t[4:] + '__svd_' + str(i) :  model.pu[:,i].astype('float16') for i in range(n_factors) },   **{'f__' + t[4:] +'__bias' : (model.bu).astype('float16')} })
    res_emb_df   = pd.DataFrame({**{'res_cid' :list(model.item_dict.keys())} , **{'f__' + t[4:] + '__svd_' + str(i) :  model.qi[:,i].astype('float16') for i in range(n_factors) },   **{'f__' + t[4:] +'__bias' : (model.bi).astype('float16')} })
    
    users_all_emb_df = pd.merge(users_all_emb_df, users_emb_df, on='user_cid', how='left')
    res_all_emb_df   = pd.merge(res_all_emb_df  , res_emb_df  , on='res_cid' , how='left')

    print(t)
```

```python
display(users_all_emb_df.head(n=5))
```

```python
features_dfs = {'user': users_all_emb_df,  'res': res_all_emb_df}
```

```python
save_features_dfs=True
```

```python
if save_features_dfs:
    for u_r in ['user', 'res']:
        filename = 'klascement_'+ u_r +'_vec_feats_df'
        data_location = 's3://{}/{}'.format(bucket, filename)
        features_dfs[u_r].to_csv(data_location + '.csv', encoding='utf-8', index=False, float_format='%.5f') 
else:
    print("nothing saved, since save_features_dfs = False")
```

```python

```
