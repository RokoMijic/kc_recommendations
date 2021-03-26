# Linear Analysis - vectorized

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
from z_utilities import stack_dfs_common_column, df_to_matix, hash_df
```

------------------------------------------------

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('max_colwidth', 175)
```

--------------------------------------------------


### Load data 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
cid_dtypes    =   {
                    'res_cid'                  : 'int32', 
                    'user_cid'                 : 'int32'
                  }

target_dtypes =   {
                    't_b_favourited'           : 'int8', 
                    't_i_score'                : 'int8', 
                    't_b_clicked_through'      : 'int8', 
                    't_b_previewed'            : 'int8', 
                    't_b_downloaded'           : 'int8', 
                    't_b_used'                 : 'int8', 
                    't_b_visited_detail_pg'    : 'int8' 
                   }

TARGETS = list(target_dtypes.keys())
TARGETS_N = [t[4:] for t in TARGETS]

cid_target_dtypes = {**cid_dtypes, **target_dtypes}

df_ratings = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, 'klascement_vectorized_filtered10_negsampling.csv'),   dtype  = target_dtypes )
```

```python

```

#### Load features dataframe

```python
feats_dfs = {}

for u_r in ['user', 'res']:
    filename = 'klascement_'+ u_r +'_vec_feats_df'
    data_location = 's3://{}/{}'.format(bucket, filename)

    feats_dfs[u_r]  = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, filename + '.csv')  ,
                                  dtype  = {   u_r+'_cid'      : 'int32'      }                              )
    
    for c in [c for c in feats_dfs[u_r].columns.tolist() if 'f_' in c]:     feats_dfs[u_r][c] = feats_dfs[u_r][c].astype('float32')
        
user_feats_df = feats_dfs['user']
res_feats_df  = feats_dfs['res']    
```

```python
display(user_feats_df.head(n=2))
print(user_feats_df.shape)
```

```python
display(res_feats_df.head(n=2))
print(res_feats_df.shape)
```

```python
# feats_df_res.dtypes
```

-----------------------------------------------

```python
def train_test_split_aug(df, test_size, random_state=0):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    assert(test.shape[0] >= 1 )
    assert(train.shape[0] >= 1 )
    return train, test 
```

-----------------------------

```python
df_ratings_small = df_ratings.sample(n=500000).reset_index(inplace=False, drop=True)
df_ratings_full  = df_ratings

print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
print('df_ratings_full    = {:,}'.format( int( float ('%.3g' % df_ratings.shape[0]        ) ) )   )
```

------------------------------------------------------

```python
df_train_full, df_test_full = train_test_split_aug(df_ratings_full, test_size=0.1, random_state=0)
```

expect hashes: reXXomF$, cmViUA0z

```python
print(hash_df(df_train_full))
print(hash_df(df_test_full))
```

```python
df_test_full
```

------------------------------------------------

```python

```

```python

```

```python

```

### Make a product dataframe given a list of user/resource pairings and uers and resoruce feature dataframes

```python
def make_feats_df(user_res_set, user_feats_df, res_feats_df):
    
    assert user_feats_df.columns.tolist()[1:] == res_feats_df.columns.tolist()[1:] 
    
    user_feats_ss_df = pd.merge(user_res_set[['user_cid']], user_feats_df, on='user_cid', how='inner')
    res_feats_ss_df  = pd.merge(user_res_set[['res_cid']], res_feats_df, on='res_cid', how='inner')    
    
    assert user_feats_ss_df.columns.tolist()[1:] == res_feats_ss_df.columns.tolist()[1:]
    
    user_bias_ss_df  = user_feats_ss_df.filter(regex='.*_bias.*')
    res_bias_ss_df   = res_feats_ss_df.filter(regex='.*_bias.*')
    
    user_svd_ss_df   = user_feats_ss_df.filter(regex='^f__.*_svd_.*')
    res_svd_ss_df    = res_feats_ss_df.filter(regex='^f__.*_svd_.*')

    user_svd_matrix  = user_svd_ss_df.to_numpy()
    res_svd_matrix   = res_svd_ss_df.to_numpy()
    combined_svd_matrix = user_svd_matrix*res_svd_matrix
    
    combined_svd_df = pd.DataFrame(combined_svd_matrix, columns=user_svd_ss_df.columns)    
    
    assert combined_svd_df.columns.tolist()[1:] == user_svd_ss_df.columns.tolist()[1:] 

    combined_df = pd.concat([user_res_set, user_bias_ss_df, res_bias_ss_df, combined_svd_df], axis=1) 
    
    return combined_df
```

```python

```

```python
%%time
med_feats_df = make_feats_df(df_test_full.head(n=500000), user_feats_df, res_feats_df)
display(med_feats_df.head(n=10))
```

```python

```

```python
def make_total_score_df(feats_df):
    
    totals_df = feats_df[['res_cid', 'user_cid']].copy()
    
    for t in TARGETS_N: 
        totals_df['f__' + t + '_tot' ] = feats_df.filter(regex='^f__' + t + '.*').sum(axis=1) 

    return totals_df
```

```python

```

```python
%%time
med_totals_df = make_total_score_df(make_feats_df(df_ratings_full.head(n=200000), user_feats_df, res_feats_df))
display(med_totals_df.head(n=2))
print(med_totals_df.shape)
```

```python

```

```python
def corrplot_of_corrdfr(corrdfr, cmap = sns.diverging_palette(20, 220, n=256)):
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap( corrdfr,  center=0, cmap=cmap, square=True, annot=True )
    ax.set_xticklabels( ax.get_xticklabels(), rotation=45, horizontalalignment='right');
```

```python
corrplot_of_corrdfr(med_totals_df.filter(regex='^f__.*').corr())
```

```python
corrplot_of_corrdfr(df_test_full.filter(regex='^t_.*').corr())
```

-------------------------------------------------------------------------------------


## Generate principal components of the different total scores

```python
def do_pca_cols_df(data_df, n_components=None, expl_var_goal=None, verbose=True):
    '''
    Do PCA on a dataframe which has each variable in its own column
    Returns the trained PCA model and explained variance. 
    '''
    
    assert (n_components is not None) or (expl_var_goal is not None)
        
    data_matrix = data_df.to_numpy()
    
    def get_explained_variance_and_model(n_components):
        model = TruncatedSVD(n_components=n_components, n_iter=5, random_state=1)
        model.fit(data_matrix)
        return(  sum(model.explained_variance_ratio_), model  )
    
    def get_explained_variance(n_components): 
        explained_variance, _ = get_explained_variance_and_model(n_components)
        return explained_variance
    
    if n_components is not None:  
        explained_variance, model = get_explained_variance_and_model(n_components)
    else:
        max_tries, curr_low_bnd, curr_high_bnd     =   10, 1, data_matrix.shape[1]-1
        
        for try_num in range(max_tries):
            new_guess = curr_low_bnd   +   (curr_high_bnd - curr_low_bnd)\
                       // 2**( max(1, int(2/3*min(10, max_tries) ) - int(1.5*try_num) ) )      
            print(f'{new_guess} comps', end=', ')    
            new_val   =    get_explained_variance(new_guess)    
            print(f'{new_val:.2f} ', end='; ') 
            if   new_val >= expl_var_goal: curr_high_bnd = new_guess
            elif new_val <  expl_var_goal: curr_low_bnd  = new_guess
            if(abs(curr_high_bnd-curr_low_bnd)) <=1: break
                
        explained_variance, model = get_explained_variance_and_model(curr_high_bnd)
    
    return explained_variance, model
```

```python
def comp_fr_mdl(model, desc_df, thr_to_show=None , num_to_show=None, verbose=True):
    '''
    print meaningful descriptions of principal components, also return a dataframe of component loadings 
    with meaningful names
    '''    
    desc_name_col = [x for x in desc_df.columns.tolist() if '_id' not in x ][0] 
    desc_id_col   = [x for x in desc_df.columns.tolist() if '_id'     in x ][0] 
    desc_word     = desc_id_col[:-3].title()
    
    comps_df      = pd.merge(  desc_df.set_index(desc_id_col, drop=False ),
                          pd.DataFrame(  {'comp_'+str(i) :  model.components_[i,:].astype('float32') 
                                          for i in range( (model.components_).shape[0]  ) }  
                                      ), 
                          left_index=True, right_index=True
                       )
    
    for i in range( (model.components_).shape[0]  ):
        comp_i = 'comp_' + str(i)
        df_this_comp = comps_df[[desc_name_col , comp_i]].sort_values(  by=[comp_i], axis=0, ascending=False )\
                                                         .reset_index(drop=True)
        
        if thr_to_show is not None     :  df_this_c_f = (lambda d, c, v: d[d[c] >= v] )  (df_this_comp, comp_i, thr_to_show )
        if num_to_show is not None     :  df_this_c_f = df_this_c_f.head(n=num_to_show) 
        if (thr_to_show is None and num_to_show is None) or df_this_c_f.shape[0]==0   :   df_this_c_f = df_this_comp.head(n=2) 
            
        def make_comp_name(list_of_var_descs):
            cleaned_list = [s.title() for s in [re.sub(r'[^a-zA-Z0-9\s]',r'', s)  for s in list_of_var_descs  ] ]
            split_lists = [s.split(' ') for s in cleaned_list]
            limited_lists = [l[:2] for l in split_lists[:3] ]
            flat_short_list = [w[:10] for l in limited_lists for w in l ]
            return ''.join(flat_short_list)[:16]
        
        perc_var = int(100.0*model.explained_variance_ratio_[i])
            
        comp_i_meaningful_name = 'COMP' + '__' + make_comp_name(df_this_c_f[desc_name_col].values.tolist()) + '__' + str(perc_var)+'%__'
    
        df_this_c_f  = df_this_c_f.rename(columns = {comp_i: comp_i_meaningful_name} )
        comps_df     =    comps_df.rename(columns = {comp_i: comp_i_meaningful_name} )
        
        if verbose: display( df_this_c_f  )
    
    return comps_df
```

```python
def plot_pca_heatmap(comps_df, cmap='RdBu_r', sort_by_c=0, top_n=500, pop_comp=0):
    '''
    Use Seaborn to plot a heatmap of components and their loadings, 
    with some sorting and trimming of irrelevant variables
    '''
    
    comps_cp_df = comps_df.copy()
    
    desc_name_col = [x for x in comps_cp_df.columns.tolist() if ('name' in x) and ('%' not in x) ][0]
    compnames     = [x for x in comps_cp_df.columns.tolist() if '%' in x ]
    
    comps_cp_df               =  comps_cp_df.set_index(desc_name_col, inplace=False).filter(regex='.*%.*', axis=1)
    comps_cp_df.index         =  comps_cp_df.index.str[:15]
    comps_cp_df               =  comps_cp_df.sort_values(  by=[compnames[pop_comp]], axis=0, ascending=False ).head(n=top_n)
    comps_cp_df_sorted        =  comps_cp_df.sort_values(  by=[compnames[sort_by_c]], axis=0, ascending=False )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(comps_cp_df_sorted, annot=True, cmap=cmap, center=0.00)
```

```python
targets_n_df = pd.DataFrame( {'targed_id': range(len(TARGETS_N)), 'target_name': TARGETS_N }   )
```

```python
for nc in range(1, 7):
    print(nc, end=', ')
    print(do_pca_cols_df(med_totals_df.filter(regex='^f__.*'), n_components=nc))
```

```python
for nc in range(1, 7):
    print(nc, end=', ')
    print(do_pca_cols_df(df_test_full.filter(regex='^t_.*').head(n=500000), n_components=nc))
```

```python

```

```python

```

```python
pred_var_expl_6, pred_f_6_model = do_pca_cols_df(med_totals_df.filter(regex='^f__.*'), n_components=6)
```

```python
var_expl_6, f_6_model = do_pca_cols_df(df_test_full.filter(regex='^t_.*').head(n=500000), n_components=6)
```

```python
var_expl_6
```

```python

```

```python
preds_comps_6_df = comp_fr_mdl(model=pred_f_6_model, desc_df=targets_n_df, verbose=False)
```

```python
comps_6_df = comp_fr_mdl(model=f_6_model, desc_df=targets_n_df, verbose=False)
```

```python
comps_6_df
```

```python

```

```python

```

```python
plot_pca_heatmap(comps_6_df, sort_by_c=0)
```

```python
# preds_comps_6_df['COMP__ClickedthrFavour__49%__'] *= -1
# preds_comps_6_df['COMP__UsedVisiteddet__9%__'] *= -1
# preds_comps_6_df['COMP__UsedClickedthr__5%__'] *= -1
# preds_comps_6_df['COMP__DownloadedFavour__4%__'] *= -1
# preds_comps_6_df['COMP__UsedPreviewed__3%__'] *= -1
```

```python

```

```python
plot_pca_heatmap(preds_comps_6_df, sort_by_c=0)
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

```

```python

```

```python

```

```python

```

### Add embedding dimensions to data

```python
def create_embedded_df(df_in, test_size=0.05 ):
    learning_rate = 0.005   
    regularization = 0.0  
    n_epochs = 60   
    n_factors = 3   
    
    assert 0.0 < test_size < 1.0
    
    df = df_in.copy()
    
    # In this section we make sure that there are no users or resources that only occur in the test set:
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
res_best['mean_absolute_error']
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
