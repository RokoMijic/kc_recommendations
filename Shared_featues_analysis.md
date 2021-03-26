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

# Analyse with augmented data


## Create features based on number of shared courses

```python
def make_union_intersect_df(user_res_df, user_extra_d_df, res_extra_d_df, verbose=False):
    
    user_res_df.reset_index(inplace=True)
    
    #---------------------------------------------------------
    assert 'user_cid' in user_extra_d_df.columns.tolist()
    assert 'res_cid'  in res_extra_d_df.columns.tolist()
    xd_name_1 = [c for c in user_extra_d_df.columns.tolist() if 'user_cid' not in c][0]
    xd_name_2 = [c for c in res_extra_d_df.columns.tolist() if 'res_cid' not in c][0]
    assert xd_name_1==xd_name_2
    xd_name = xd_name_1
    #---------------------------------------------------------
    
    df1 = pd.merge(user_res_df, user_extra_d_df, on='user_cid')
    df2 = pd.merge(user_res_df, res_extra_d_df, on='res_cid')
    df3 = pd.concat([df1, df2], ignore_index=True)
    del df1
    del df2

    df_final = df3.groupby(['user_cid', 'res_cid'])[xd_name].agg(intersec='count', union='nunique').reset_index().astype({'user_cid': 'int32', 'res_cid': 'int32', 'intersec' : 'int16', 'union': 'int16' })
    del df3
    
    df_final['intersec'] -= df_final['union']
    
    return df_final
```

```python
def make_ui_df(xtra_d_type):
    if xtra_d_type=='courses'    :  user_extra_d_df, res_extra_d_df  = users_courses_df, resources_courses_df
    elif xtra_d_type=='edutypes' :  user_extra_d_df, res_extra_d_df  = users_edutypes_df,  resources_edutypes_df 
        
    uidf=make_union_intersect_df(user_res_df = pd.DataFrame(index=augmented_df.index), user_extra_d_df=user_extra_d_df, res_extra_d_df=res_extra_d_df , verbose=True)   
    
    return uidf.rename(columns = {'intersec':'intersec_' + xtra_d_type, 'union':'union_' + xtra_d_type } )    
```

###### Note: 64GB RAM is needed to run these two in parallel, 32GB for series. Expect Wall time: about 5-7 minutes :

```python
make_iou_features = True
```

```python
run_parallel = False
```

```python
%%time
if run_parallel and make_iou_features:  ui_courses_df, ui_edutypes_df  = Parallel(n_jobs=2)(delayed(make_ui_df)( xtra_d_type ) for xtra_d_type in ['courses', 'edutypes' ] )  
```

```python
%%time
if (not run_parallel) and make_iou_features: ui_courses_df = make_ui_df( 'courses' )
```

```python
%%time
if (not run_parallel) and make_iou_features:  ui_edutypes_df = make_ui_df( 'edutypes' )
```

```python
%%time
if make_iou_features:  df_ui_edutype_courses = pd.merge(ui_courses_df, ui_edutypes_df, on=['res_cid', 'user_cid'], how='inner')
```

```python
if make_iou_features: 

    df_ui_edutype_courses['f_IoU_courses'] = (df_ui_edutype_courses['intersec_courses'] / df_ui_edutype_courses['union_courses'] ).astype('float32')
    df_ui_edutype_courses['f_IoU_edutypes'] = (df_ui_edutype_courses['intersec_edutypes'] / df_ui_edutype_courses['union_edutypes'] ).astype('float32')
    df_ui_edutype_courses['f_I_courses'] = (df_ui_edutype_courses['intersec_courses'] / df_ui_edutype_courses['intersec_courses'].mean() ).astype('float32')
    df_ui_edutype_courses['f_I_edutypes'] = (df_ui_edutype_courses['intersec_edutypes'] / df_ui_edutype_courses['intersec_edutypes'].mean() ).astype('float32')

    del df_ui_edutype_courses['intersec_courses']
    del df_ui_edutype_courses['union_courses']
    del df_ui_edutype_courses['intersec_edutypes']
    del df_ui_edutype_courses['union_edutypes']

    df_ui_edutype_courses = df_ui_edutype_courses.sort_values(by=['res_cid', 'user_cid'], axis=0, inplace=False).reset_index(inplace=False, drop=True).set_index(['res_cid', 'user_cid'])

    display(df_ui_edutype_courses.head(n=2))

    aug_w_iou_df = pd.merge(augmented_df, df_ui_edutype_courses, left_index=True, right_index=True)

    aug_w_iou_df['f_const'] = 1.0
    aug_w_iou_df['f_const'] = aug_w_iou_df['f_const'].astype('float32')
    aug_w_iou_df['f_noise'] =  np.random.uniform(low=0.0, high=1.0, size=aug_w_iou_df.shape[0]).astype('float32')
    aug_w_iou_df = (lambda d, cols_2_end : d.reindex(columns=[c for c in d.columns.tolist() if c not in cols_2_end] + cols_2_end)) (aug_w_iou_df, ['eng_score'])

    display(aug_w_iou_df.head(n=2))


```

----------------------------------------

```python
%%time
if make_iou_features: train_linear_model(aug_w_iou_df, feature_regex='f_bias')
```

```python
%%time
if make_iou_features: train_linear_model(aug_w_iou_df, feature_regex='f_(bias|I)')
```

```python
%%time
if make_iou_features: train_linear_model(aug_w_iou_df, feature_regex='f_(bias|svd.*_x_.*)')
```

```python
%%time
if make_iou_features: train_linear_model(aug_w_iou_df, feature_regex='f_(bias|svd.*_x_.*|I)')
```
