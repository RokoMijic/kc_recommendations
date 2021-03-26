---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Funk Experiments

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

import numpy as np
import pandas as pd

import swifter

from scipy.sparse import csr_matrix, coo_matrix

import time

from funk_svd import SVD

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import random_projection
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD

from experiment_runner import run_experiments
```

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 150)
```

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

###### load noninteraction data

```python
filenames =  [ 'users_courses' ,  'users_edutypes' ,    
               'resources_courses'  ,   'resources_edutypes'  ,   'resources_keywords'   ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 
    
    print(dfs[df_name].dtypes)
    

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
df_ratings_small = df_ratings.sample(n=500000).reset_index(inplace=False, drop=True)
df_ratings_med   = df_ratings.sample(n=2000000).reset_index(inplace=False, drop=True)
df_ratings_large = df_ratings.sample(n=7000000).reset_index(inplace=False, drop=True)
df_ratings_full  = df_ratings.sample(frac = 1.0, random_state=4).reset_index(inplace=False, drop=True)

print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
print('df_ratings_med     = {:,}'.format( int( float ('%.3g' % df_ratings_med.shape[0]    ) ) )   )
print('df_ratings_large   = {:,}'.format( int( float ('%.3g' % df_ratings_large.shape[0]  ) ) )   )
print('df_ratings_full    = {:,}'.format( int( float ('%.3g' % df_ratings.shape[0]        ) ) )   )
```

```python
poss_vals_probs = pd.DataFrame( {'prob': df_ratings.groupby('eng_score')['eng_score'].count()} ).reset_index(drop=False)
poss_vals_probs['prob'] /= df_ratings.shape[0]
eng_vals  = poss_vals_probs['eng_score'].values.tolist()
eng_probs = poss_vals_probs['prob'].values.tolist()
print(eng_vals)
print(eng_probs)

df_ratings_fake = df_ratings.copy()
df_ratings_fake['eng_score'] = np.random.choice(a=eng_vals, size=df_ratings_fake.shape[0], replace=True, p=eng_probs)
```

-----------------------------------------------------------------------


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

```python

```

-----------------------------------------------------

```python

```

```python
import experiment_runner
import importlib
importlib.reload(experiment_runner)
from experiment_runner import run_experiments
```

```python

```

### Analyse with Funk

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

###### Run with "df_ratings_large" for results in 100 seconds on a 16 core machine, "df_ratings_full" takes about 6-10 minutes and "df_ratings_med" takes 15 seconds

```python

```

```python
# run_experiments( algo_dict   ={   'funk'             :  funk_algorithm                                 }  ,
#                  dataset_dict={   'df_ratings_small' :  df_ratings_small                               }  , 
#                  metrics_dict={   'mae'              :  mean_absolute_error                            }  , 
#                  hyperp_dict ={   'learning_rate'    :  [0.003, 0.006, 0.01]                              ,      
#                                   'regularization'   :  [0.0, 0.01, 0.06]                                 ,
#                                   'n_epochs'         :  [50, 100, 150, 200]                               ,
#                                   'n_factors'        :  [0, 1, 2, 3, 5, 7, 10 ]                        }  , 
#                  rchoice_tot=8                                                                           , 
#                  rchoice_hparam=6                                                                        ,
#                  n_jobs=4                                                                                , 
#                  backend_name='loky' 
#               )
```

```python

```

### Get fast baseline Predictions


Mean regressor -- predicts the mean rating for each item, user, or a combination of the two

```python
class MeanRegressor(BaseEstimator, RegressorMixin):  
    """Flexible mean regressor for dataframes with  ['u_id', 'i_id', 'rating' ] columns"""

    def __init__(self, regtype):
        if not (regtype in ['g', 'u', 'i', 'ui'] ): raise ValueError(f'Bad regressor type: {regtype}; should be one of  (g, u, i, ui)')
        self._regtype = regtype

    def fit(self, X: pd.DataFrame):
        X.reset_index(drop=True, inplace=True)
        if not (X.columns.values.tolist() == ['i_id', 'u_id', 'rating']): raise ValueError(f'Bad columns: {X.columns.values.tolist()}; should be [i_id, u_id, rating]') 
            
        self._meanval = X['rating'].mean()
        
        if self._regtype in ['i', 'ui']:  self._item_mean_df = pd.DataFrame({'rating' : X.groupby('i_id')['rating'].mean()}).reset_index(drop=False)
        if self._regtype in ['u', 'ui']:  self._user_mean_df = pd.DataFrame({'rating' : X.groupby('u_id')['rating'].mean()}).reset_index(drop=False)    
        if self._regtype == 'ui'       :  
            #TODO: this code is wrong and produces a bad answer, but I dont know why:
            bu = X[['u_id']].merge(self._user_mean_df, on='u_id', how='left')
            bi = X[['i_id']].merge(self._item_mean_df, on='i_id', how='left')
            rui_minus_bi = X['rating'] - bi['rating']
            bu_minus_bi  = bu['rating'] - bi['rating']
            self._alpha  = (rui_minus_bi*bu_minus_bi).sum() / (bu_minus_bi*bu_minus_bi).sum()
        return self

    def predict(self, X: pd.DataFrame):
        try:
            getattr(self, "_meanval")
            if self._regtype in ['i', 'ui']: getattr(self, "_item_mean_df")
            if self._regtype in ['u', 'ui']: getattr(self, "_user_mean_df")
            if self._regtype == 'ui'       : getattr(self, "_alpha")  
                
        except AttributeError:
            raise RuntimeError("You must train the regressor before predicting data!")

        if not (X.columns.values.tolist() == ['i_id', 'u_id']): raise ValueError(f'Bad columns: {X.columns.values.tolist()}; should be [i_id, u_id]')
            
        if self._regtype in ['i', 'ui']: item_mean_preds =  X.merge(self._item_mean_df, on='i_id', how='left').fillna(self._meanval)['rating']
        if self._regtype in ['u', 'ui']: user_mean_preds =  X.merge(self._user_mean_df, on='u_id', how='left').fillna(self._meanval)['rating']
            
        if self._regtype=='g'  :  return( [self._meanval]*X.shape[0]       ) 
        if self._regtype=='i'  :  return( item_mean_preds.values.tolist()  )
        if self._regtype=='u'  :  return( user_mean_preds.values.tolist()  )
        #TODO: remove magic number 0.2 here and replace with a correct calculation of alpha
        if self._regtype=='ui' :  return(    ( 0.2*user_mean_preds + (1-0.2)*item_mean_preds).values.tolist()     )
```

```python

```

```python
def make_baseline_algo(Regressor, regtype):
    ''' Makes a baseline algorithm (no parameters) out of a regressor
    '''
    
    def regressor_algo(dataset, metrics_dict):
        rn_dataset_df = dataset.rename(columns = {'user_cid':'u_id', 'res_cid':'i_id', 'eng_score':'rating' } )
        train_df, test_df =  train_test_split_aug(rn_dataset_df, test_size=0.1)
        
        iregressor = Regressor(regtype=regtype)
        iregressor.fit(X=train_df)
        test_pred = iregressor.predict(test_df[['i_id', 'u_id']] )

        return {  metric_name : metric_fn(test_pred, test_df['rating'])   for   metric_name, metric_fn in metrics_dict.items()  } 
    
    return regressor_algo
```

```python

```

```python

```

```python
search_paramspace=False
```

```python
if search_paramspace:
    res =     run_experiments( algo_dict   ={     'funk'             :  funk_algorithm                                    ,  
                                                  'useritemmean'     :  make_baseline_algo(MeanRegressor, regtype='ui')   ,                               
                                                  'itemmean'         :  make_baseline_algo(MeanRegressor, regtype='i')    ,
                                                  'usermean'         :  make_baseline_algo(MeanRegressor, regtype='u')    ,
                                                  'globalmean'       :  make_baseline_algo(MeanRegressor, regtype='g') }  ,
                                 dataset_dict={   'df_ratings_full'  :  df_ratings_full                                }  , 
                                 metrics_dict={   'mae'              :  mean_absolute_error                            }  , 
                                 hyperp_dict ={   'learning_rate'    :  [0.003, 0.006, 0.01]                              ,      
                                                  'regularization'   :  [0.0, 0.01, 0.06]                                 ,
                                                  'n_epochs'         :  reversed([75, 125, 225])                          ,
                                                  'n_factors'        :  reversed([0, 1, 2, 3, 4, 5])                   }  , 
                                 rchoice_tot=-1                                                                           , 
                                 rchoice_hparam=30                                                                        ,
                                 n_jobs=11                                                                                , 
                                 backend_name='loky' 
                              )
    
    display(res)

else: 
    print("Search not executed since search_paramspace=False")
```

-------------------------------------------------------------------------


### Add embedding dimensions to data


###### This code will augment the data with embedding dimensions from Funk svd, which can then be used in linear models. 

```python
def create_embedded_df(df_in, test_size=0.05 ):
    learning_rate = 0.003   
    regularization = 0.0  
    n_epochs = 125   
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
    
    users_emb_df = pd.DataFrame({**{'user_cid':list(model.user_dict.keys())} , **{'f_svd_user' + str(i) :  model.pu[:,i].astype('float16') for i in range(n_factors) },   **{'f_user_bias': (model.bu).astype('float16')} })
    items_emb_df = pd.DataFrame({**{'res_cid' :list(model.item_dict.keys())} , **{'f_svd_res' + str(i)  :  model.qi[:,i].astype('float16') for i in range(n_factors) },   **{'f_item_bias': (model.bi).astype('float16')} })
    
    df = df.merge(users_emb_df, on='user_cid', how='left').merge(items_emb_df, on='res_cid', how='left')
    
    for i in range(n_factors): df['f_svd_user'+str(i)+'_x_res'+str(i)] = df['f_svd_user'+str(i)]*df['f_svd_res'+str(i)]
        
    df.set_index(['res_cid', 'user_cid'], drop=True, inplace=True)
    
    df = (lambda d, cols_2_end : d.reindex(columns=[c for c in d.columns.tolist() if c not in cols_2_end] + cols_2_end)) (df, ['f_user_bias', 'f_item_bias', 'eng_score'])
    
    
    return df
```

```python
use_full_dataset_for_aug = False
```

```python jupyter={"outputs_hidden": true}
if use_full_dataset_for_aug: 
    augmented_full_df = create_embedded_df(df_ratings_full)
else:
    augmented_full_df = create_embedded_df(df_ratings_small)
```

```python
augmented_full_df
```

##### Train a linear model on the embedding features

```python
augmented_small_df = augmented_full_df.sample(frac=0.01)
augmented_med_df   = augmented_full_df.sample(frac=0.1)
```

```python
augmented_df = augmented_full_df
```

```python
def train_linear_model(df):
    train_df, test_df = ( lambda d: ( d[d['testset']==False] , d[d['testset']==True] ) )( augmented_df )
    def feats(df): return df.filter(like='f_', axis=1)
    def targ(df): return df['eng_score'].ravel()
    
    train_df_X, train_y, test_df_X, test_y =   (lambda tr, te:  (  feats(tr) , targ(tr), feats(te) , targ(te)  )   )    (    *( lambda d: ( d[d['testset']==False] , d[d['testset']==True] ) )( augmented_df )   )
    
    ols=linear_model.LinearRegression(n_jobs=-1)
    
    ols.fit(X=train_df_X, y=train_y)
    
    pred_y = ols.predict( test_df_X )
    
    return  {  'mean_absolute_error' : mean_absolute_error(pred_y, test_y)   }     
```

```python
train_linear_model(augmented_df)
```

----------------------------------------------------------------------

```python

```

```python


```

```python


```

```python

```

--------------------------------------------------------

```python

```

### Analyse with augmented data


##### Create features based on number of shared courses

```python
def make_union_intersect_df(user_res_df, user_extra_d_df, res_extra_d_df, verbose=False):
    
    assert 'user_cid' in user_extra_d_df.columns.tolist()
    assert 'res_cid'  in res_extra_d_df.columns.tolist()
    xd_name_1 = [c for c in user_extra_d_df.columns.tolist() if 'user_cid' not in c][0]
    xd_name_2 = [c for c in res_extra_d_df.columns.tolist() if 'res_cid' not in c][0]
    assert xd_name_1==xd_name_2
    xd_name = xd_name_1
    
    df1 = pd.merge(user_res_df, user_extra_d_df, on='user_cid')
    df2 = pd.merge(user_res_df, res_extra_d_df, on='res_cid')
    df3 = pd.concat([df1, df2], ignore_index=True)
    del df1
    del df2

    df_final = df3.groupby(['user_cid', 'res_cid'])[xd_name].agg(intersec='count', union='nunique').reset_index().astype({'user_cid': 'int32', 'res_cid': 'int32', 'intersec' : 'int16', 'union': 'int16' })
    del df3
    df_final['intersec'] -= df_final['union']
    
    df_final.set_index(['res_cid', 'user_cid'], drop=True, inplace=True)
    
    return df_final
```

```python
# Expect Wall time: 2min 48s
%%time
ui_courses_df = make_union_intersect_df(user_res_df = df_ratings_full[['user_cid', 'res_cid']], user_extra_d_df  = users_courses_df, res_extra_d_df  = resources_courses_df , verbose=True)
```

```python
ui_courses_df.head(n=2)
```

```python
# Expect Wall time:  5min 1s
%%time
ui_edutypes_df = make_union_intersect_df(user_res_df = df_ratings_full[['user_cid', 'res_cid']], user_extra_d_df  = users_edutypes_df, res_extra_d_df  = resources_edutypes_df , verbose=True)
```

```python
ui_edutypes_df['f_IoU_edutype'] = (ui_edutypes_df['intersec']/ui_edutypes_df['union']).astype('float16')
ui_edutypes_df.head(n=2)
```

```python

```

```python
ui_edutypes_df.dtypes
```

```python

```

```python

```

```python

```

###### PCA breakdowns of the users/resources course spaces

```python
def df_to_matix(df, col1, col2, value=1.0): 
    return coo_matrix( ([value]*df.shape[0], (df[col1], df[col2]) )  )
```

```python
def make_svd_df(data_df, col1, col2, n_components):
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=1)
    data_matrix = df_to_matix(data_df, col1, col2)
    svd.fit(data_matrix)
    
```

```python
users_courses_matrix = df_to_matix(users_courses_df, 'user_cid', 'course_id')
users_courses_matrix.shape
```

```python
res_courses_matrix = df_to_matix(resources_courses_df, 'res_cid', 'course_id')
res_courses_matrix.shape
```

```python
resources_keywords_matrix = df_to_matix( (lambda d, c, n: d[d[c] >= n]) (resources_keywords_df, 'keyword_count' , 300 ) , 'res_cid', 'keyword_id')
resources_keywords_matrix.shape
```

```python
res_edutypes_matrix = df_to_matix(resources_edutypes_df, 'res_cid', 'edutype_id')
res_edutypes_matrix.shape
```

```python
users_edutypes_matrix = df_to_matix(users_edutypes_df, 'edutype_id', 'user_cid')
users_edutypes_matrix.shape
```

```python
svd = TruncatedSVD(n_components=4, n_iter=5, random_state=1)
```

```python
svd.fit(users_edutypes_matrix)
```

```python
print(svd.explained_variance_ratio_)
```

```python
print(svd.explained_variance_ratio_.sum())
```

```python
print(svd.explained_variance_ratio_.sum())
```

```python
print((svd.components_).shape[0] )
```

```python
1+1
```

```python
n_components = (svd.components_).shape[0]

pd.DataFrame(  {'f_user_edu'+str(i) :  svd.components_[i,:].astype('float16') for i in range( n_components ) }    )
```

```python
users_edutypes_df[['user_cid']].drop_duplicates().reset_index(drop=True)
```

```python

```

```python

```

```python

```

```python

```

Augment the dataframe to contain information about the tags and subjects that learning resources are associated with

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

### Analyse with scipy KNN and random projections


#### Use random projections to reduce the dimensionality of the space 

```python
def df_to_sparse(df): return coo_matrix((df.rating, (df.u_id, df.i_id)))  
```

```python
def reduce_matrix_dimension_rproj(matrix_hidim, eps):
    
    transformer = random_projection.SparseRandomProjection(n_components='auto', density='auto', eps=eps, dense_output=True, random_state=314)
    matrix_lowdim = (transformer.fit_transform(matrix_hidim)).astype('float16')
    
    return matrix_lowdim
```

```python
sm_ratings = df_to_sparse(chosen_df)
print(sm_ratings.shape)
print(sm_ratings.getnnz())
( lambda m : m.getnnz() / np.prod(m.shape)  ) ( sm_ratings )
```

```python
red_m_ratings = reduce_matrix_dimension_rproj(sm_ratings, eps=0.2)
print(red_m_ratings.shape)
```

```python
kd_red_model_knn = NearestNeighbors(metric='l1', algorithm='kd_tree', n_neighbors=1, n_jobs=-1)
kd_red_model_knn.fit(red_m_ratings)
```

```python
red_model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=1, n_jobs=-1)
red_model_knn.fit(red_m_ratings)
```

```python
# full_model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=1, n_jobs=-1)
# full_model_knn.fit(sm_ratings)
```

```python
def get_neighbors(row, model_knn, matrix, n_neighbors):
    
    print("1")
    
    rowvector = matrix.getrow(row)  if 'sparse' in str(type(matrix)) else matrix[[row]]
    
    print("2")
    
    neighbors_data = model_knn.kneighbors( rowvector   , n_neighbors=n_neighbors  )
    
    print("3")
    
    neighbors = neighbors_data[1][0][1:]
    
    print("4")
    
    return neighbors
```

```python
def nn_preds(row, model_knn, matrix, full_matrix, n_neighbors, n_preds=None): 
    
    print("start")
    
    neighbors = get_neighbors(row, model_knn, matrix, n_neighbors=n_neighbors)
    
    print("A")
    
    full_matrix_csr = full_matrix.tocsr()

    print("B")

    neighbors_choices =  [ full_matrix_csr.getrow(user)  for user in neighbors]   
    
    print("C")
    
    sum_choices = sum(neighbors_choices).tocoo()
    
    print("D")
    
    mean_choices = (sum_choices / n_neighbors).astype("float16")
    
    print("E")
    
    scored_choices = sorted(zip(mean_choices.nonzero()[1] , mean_choices.data), key = lambda x: x[1], reverse=True)  
    
    print("F")
    
    if n_preds is None : return scored_choices
    else               : return scored_choices[:n_preds]
```

```python
nn_preds(70000, red_model_knn, red_m_ratings, sm_ratings, n_neighbors=20, n_preds=10)
```

```python

```

```python

```

```python
print(sm_ratings.getrow(70000))
```

```python

```

```python

```

```python
# import sklearn
# sorted(sklearn.neighbors.VALID_METRICS['kd_tree'])
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
assert False
```

```python

```

```python
neighbors = neighbors_data[1].tolist()[0][1:]
neighbors
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
algo_dict = {    'plus':  plusalgo ,
                 'times': timesalgo,
                 'minus': minusalgo     }
```

```python
dataset_dict =  {    'smallnums' : [1, 2, 3, 4, 5]  *300000, 
                     'mednums' :   [11, 12, 13, 14] *300000, 
                     'bignums' :   [51, 62, 73, 85] *300000    }
```

```python
hyperp_dict =  {  'h' : [0.5, 0.1],      'g' : [3,6,9]     }
```

```python

metrics_dict = {    'p-met'   :   pmetr    ,
                    'l-met'   :   lmetr         }
```

```python

```

```python
def experiment_fn(  setting  ):
    dataset, algorithm, hparams, metrics_dict   =   setting['dataset'], setting['algorithm'], setting['hparams'], setting['metrics_dict']       
     
    print(".", end="")
    result = algorithm(dataset=dataset, **hparams)
    
    return {n: m( result ) for n, m in metrics_dict.items() }
```

```python

```

```python
def run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn , n_jobs=16, rchoice_hparam = -1, rchoice_tot = -1, verbose=True):

    hyperp_settings_list = [   dict(  zip(  hyperp_dict.keys() ,  hparam_tuple  ) )  for    hparam_tuple  in  product(*hyperp_dict.values() )     ]
    
    if  0 < rchoice_hparam < len(hyperp_settings_list) :    hyperp_settings_list = random.sample(hyperp_settings_list, rchoice_hparam)
  
    experi_names_list =      [   dict(  zip(  ['dataset', 'algorithm', 'hparams'] ,  exp_tuple  ) )  
                                 for   exp_tuple  in  product( dataset_dict.keys(), algo_dict.keys(), hyperp_settings_list  )    
                             ]
    
    if  0 < rchoice_tot < len(experi_names_list) :    experi_names_list = random.sample(experi_names_list, rchoice_tot)
    
    if verbose: print(    f"Running {len(experi_names_list)} experiments"    )
    
    experi_settings_list = [   { 'dataset'      :  dataset_dict[setting_n['dataset']]      ,   
                                 'algorithm'    :  algo_dict[setting_n['algorithm']]       , 
                                 'hparams'      :  setting_n['hparams']                    ,
                                 'metrics_dict' :  metrics_dict                                } 
                            
                               for setting_n in experi_names_list
                           ]
    
    start_t = time.time()
    
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        results = Parallel()(delayed(experiment_fn)(setting) for setting in experi_settings_list)

    end_t = time.time()
    
    if verbose: print("\n%.2f seconds elapsed \n" % (end_t - start_t) )
        
    results_w_settings_list = [  {'setting': s, 'result' : r} for s, r in zip(experi_names_list, results) ]
        
    return results_w_settings_list
```

```python
run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn, rchoice_tot = 16  )
```

```python

```

```python

```

```python
# def make_union_intersect_df(user_res_df, user_extra_d_df, res_extra_d_df, verbose=False):
    
#     assert 'user_cid' in user_extra_d_df.columns.tolist()
#     assert 'res_cid'  in res_extra_d_df.columns.tolist()
#     xd_name_1 = [c for c in user_extra_d_df.columns.tolist() if 'user_cid' not in c][0]
#     xd_name_2 = [c for c in res_extra_d_df.columns.tolist() if 'res_cid' not in c][0]
#     assert xd_name_1==xd_name_2
#     xd_name = xd_name_1
    

#     df1 = pd.merge(user_res_df, user_extra_d_df, on='user_cid')
#     df2 = pd.merge(user_res_df, res_extra_d_df, on='res_cid')
#     if verbose: print('finished item merges')
        
#     df3 = pd.merge(df1, df2, on=['user_cid', 'res_cid'], how='outer')
#     df3['m'] = df3[xd_name+'_x'].eq(df3[xd_name+'_y']).astype('int16')    
#     df_intersect = df3.groupby(['user_cid', 'res_cid'])['m'].agg(  intersec='sum' ).reset_index().astype({'user_cid': 'int32', 'res_cid': 'int32'})
#     del df3
#     if verbose: print('finished intersection')
        
#     df_s1 = df1.groupby(['user_cid', 'res_cid'])[xd_name].count().reset_index().astype({'user_cid': 'int32', 'res_cid': 'int32', xd_name : 'int16' })
#     df_s2 = df2.groupby(['user_cid', 'res_cid'])[xd_name].count().reset_index().astype({'user_cid': 'int32', 'res_cid': 'int32', xd_name : 'int16' })
#     del df1
#     del df2
#     df_sum = df_s1.merge(df_s2, on=['user_cid', 'res_cid'], how='inner')
#     del df_s1
#     del df_s2
#     df_sum['sum'] = df_sum[xd_name+'_x'] + df_sum[xd_name+'_y']
#     del df_sum[xd_name+'_x']
#     del df_sum[xd_name+'_y']
#     if verbose: print('finished sum')
    
#     df_final = df_sum.merge(df_intersect,  on=['user_cid', 'res_cid'])
#     df_final['union'] = df_final['sum'] - df_final['intersec']
#     del df_final['sum']
#     if verbose: print('finished union')
        
#     return df_final
```
