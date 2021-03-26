# PCA Analysis

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
from z_utilities import stack_dfs_common_column, df_to_matix, assert_column_subset, assert_nonan
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
df_ratings_full  = df_ratings.sample(frac = 1.0, random_state=4).reset_index(inplace=False, drop=True)

print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
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


### Metrics

```python
def rel_mean_absolute_error(y_true, y_pred, mean_to_guess=None):
    if mean_to_guess is None: mean_to_guess =  np.mean(y_true)
    mae_raw   = mean_absolute_error(y_true,    y_pred)
    mae_const = mean_absolute_error(y_true,    np.full(y_pred.shape, mean_to_guess, dtype='float32')   ) 
    return mae_raw/mae_const
```

```python
def rel_mean_absolute_error_top(y_true, y_pred, threshold=4.0):
    mean_to_guess = np.mean(y_true)
    mask = (y_true >= threshold)
    return ( rel_mean_absolute_error(y_true[mask], y_pred[mask] , mean_to_guess=mean_to_guess)   )
```

```python
def rel_mean_absolute_error_val(y_true, y_pred, val):
    mean_to_guess = np.mean(y_true)
    mask = (y_true == val)
    return ( rel_mean_absolute_error(y_true[mask], y_pred[mask], mean_to_guess=mean_to_guess) )
```

```python
def rel_mean_absolute_error_1(y_true, y_pred): return rel_mean_absolute_error_val(y_true, y_pred, 1)
```

```python
def rel_mean_absolute_error_2(y_true, y_pred): return rel_mean_absolute_error_val(y_true, y_pred, 2)
```

```python
def rel_mean_absolute_error_3(y_true, y_pred): return rel_mean_absolute_error_val(y_true, y_pred, 3)
```

```python
def rel_mean_absolute_error_4(y_true, y_pred): return rel_mean_absolute_error_val(y_true, y_pred, 4)
```

```python
def rel_mean_absolute_error_5(y_true, y_pred): return rel_mean_absolute_error_val(y_true, y_pred, 5)
```

----------------------------------------------


### Load funk SVD features for users and  resources

```python
feats_dfs = {}

for u_r in ['user', 'res']:
    filename = 'klascement_'+ u_r +'_features_df'
    data_location = 's3://{}/{}'.format(bucket, filename)

    feats_dfs[u_r]  = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, filename + '.csv')  ,       dtype  = {   u_r+'_cid'      : 'int32'      }      )
    
    for c in [c for c in feats_dfs[u_r].columns.tolist() if 'f_' in c]:     feats_dfs[u_r][c] = feats_dfs[u_r][c].astype('float32')
                
feats_df_user = feats_dfs['user']
feats_df_res  = feats_dfs['res']   
```

```python
filename = 'klascement_funk_train_test_split'
data_location = 's3://{}/{}'.format(bucket, filename)        
        
train_test_split_df  = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, filename + '.csv')  ,  dtype  = {   'user_cid' : 'int32'  ,   'res_cid' : 'int32'   }  )
```

```python
display(train_test_split_df.head(n=1))
```

```python
display(feats_df_user.head(n=1))
```

```python
display(feats_df_res.head(n=1))
```

###### Augment the ratings data with the Funk features

```python
def make_aug_feats_df(ratings_df, user_feats_df, res_feats_df, traintest_df):

    assert_column_subset(ratings_df, 'user_cid', user_feats_df, 'user_cid' )
    assert_column_subset(ratings_df, 'res_cid', res_feats_df, 'res_cid' )
    assert(ratings_df.shape[0] == traintest_df.shape[0])

    ratings_with_traintest = pd.merge(ratings_df, traintest_df, on=['user_cid', 'res_cid'], how='inner')
    assert_nonan(ratings_with_traintest)
    
    aug_feats_df = ratings_with_traintest.merge(user_feats_df, on='user_cid', how='inner').merge(res_feats_df, on='res_cid', how='inner')
    assert_nonan(aug_feats_df)

    return aug_feats_df
```

```python
%%time
augmented_df = make_aug_feats_df(df_ratings_full, feats_df_user, feats_df_res,  train_test_split_df)
```

```python
display(augmented_df.head(n=1))
print(augmented_df.shape[0])
```

```python
def add_product_features(df, product_list):
    '''
    product_list is a list of pairs of columns of df, must start with f_
    this function works in place for memory efficiency reasons
    '''
    
    for (col1, col2) in product_list:
        assert col1 in df.columns.tolist()
        assert col2 in df.columns.tolist()
        assert col1.startswith('f_')
        assert col2.startswith('f_')
        product_name = 'f_' + col1[2:] + '_x_' + col2[2:]
        df[product_name] = df[col1] * df[col2]

    return None
```

```python
def add_noise_feat(df):
    '''
    adds a featue that is pure noise in place in a dataframe 
    '''
    
    df['f_noise'] = np.random.uniform(0, 1, size=len(df)).astype('float32')
    
    return None
```

```python
add_noise_feat(augmented_df)
```

```python
add_product_features(augmented_df, [  ('f_svd_user0', 'f_svd_res0'),  ('f_svd_user1','f_svd_res1'),  ('f_svd_user2','f_svd_res2')  ]   )   
```

```python
augmented_df
```

## Train a linear model

```python
def train_linear_model(df_data, feature_regex):
    
    print('predicting with the following features:::   ' + str( df_data.head(n=0).filter(regex=feature_regex, axis=1).columns.tolist() )     )

    def feats(df): return df.filter(regex=feature_regex, axis=1)
    def targ(df): return df['eng_score'].ravel()
    
    train_df_X, train_y, test_df_X, test_y =   (lambda tr, te:  (  feats(tr) , targ(tr), feats(te) , targ(te)  )   )    (    *( lambda d: ( d[d['testset']==False] , d[d['testset']==True] ) )( df_data )   )
    
    ols=linear_model.LinearRegression(n_jobs=-1)
    
    ols.fit(X=train_df_X, y=train_y)
    
    pred_y = ols.predict( test_df_X )
    
    return  {  'rel_mean_absolute_error'   : rel_mean_absolute_error(test_y, pred_y)    ,
               'rel_mean_absolute_error_1' : rel_mean_absolute_error_1(test_y, pred_y)  ,
               'rel_mean_absolute_error_2' : rel_mean_absolute_error_2(test_y, pred_y)  ,
               'rel_mean_absolute_error_3' : rel_mean_absolute_error_3(test_y, pred_y)  ,
               'rel_mean_absolute_error_4' : rel_mean_absolute_error_4(test_y, pred_y)  ,
               'rel_mean_absolute_error_5' : rel_mean_absolute_error_5(test_y, pred_y)  ,
               'model' : ols    }  
```

```python
%%time
train_linear_model(augmented_df, feature_regex='f_noise')
```

```python
%%time
train_linear_model(augmented_df, feature_regex='f_(bias)')
```

```python
%%time
train_linear_model(augmented_df, feature_regex='f_(bias|.*_x_.*)')
```

## PCA breakdowns of the users/resources course spaces

```python
augmented_df.head(n=2)
```

-------------------------------------

```python

```

```python
def do_pca_on_df(data_df, col1=None, col2=None, n_components=None, expl_var_goal=None, verbose=True):
    '''
    Do PCA on a dataframe to get components
    uses intelligent binary search to look for a optimal number of components for a cetain variance explained
    actually uses TruncatedSVD, but this can be varied. 
    Returns the trained PCA model and explained variance. 
    '''
    
    assert (n_components is not None) or (expl_var_goal is not None)
    assert   ( (col1 is None) and (col2 is None) ) or ( (col1 is not None) and (col2 is not None) )
    
    if (col1 is None) and (col2 is None): col1, col2 = data_df.columns.tolist()[0] , data_df.columns.tolist()[1] 
        
    data_matrix = df_to_matix(data_df, col1, col2, numeric_type='float32')

    if not data_matrix.shape[0] > data_matrix.shape[1]: 
        raise ValueError(f'{col2} has max value {data_matrix.shape[1]} which is greater'+\
                         ' than max for {col1}, {data_matrix.shape[0]}. Please switch the columns') 
    
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
        max_tries, curr_low_bnd, curr_high_bnd     =   10, 1, data_matrix.shape[1]
        
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

```

---------------------------------

```python
def comp_fr_mdl(model, data_df, desc_df, thr_to_show=None , num_to_show=None, verbose=True):
    '''
    print meaningful descriptions of principal components, also return a dataframe of component loadings 
    with meaningful names
    '''    
    desc_name_col = [x for x in desc_df.columns.tolist() if '_id' not in x ][0] 
    desc_id_col   = [x for x in desc_df.columns.tolist() if '_id'     in x ][0] 
    data_cid_col = [x for x in data_df.columns.tolist() if '_cid' in x ][0]
    desc_word = desc_id_col[:-3].title()
    data_word = data_cid_col[:-4]
    
    comps_df = pd.merge(  desc_df.set_index(desc_id_col, drop=False ),
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
            
        comp_i_meaningful_name = 'f_' + data_word + desc_word + '_%' + make_comp_name(df_this_c_f[desc_name_col].values.tolist()) + str(i)+'%'
    
        df_this_c_f  = df_this_c_f.rename(columns = {comp_i: comp_i_meaningful_name} )
        comps_df     =    comps_df.rename(columns = {comp_i: comp_i_meaningful_name} )
        
        if verbose: display( df_this_c_f  )
    
    return comps_df
```

```python

```

```python
def comp_fr_data( data_df, desc_df, n_components=None, expl_var_goal=None, 
                                 thr_to_show=None, num_to_show=None, verbose=False):
    '''
    Make a model, then use it to make a dataframe of component loadings with meaningful names
    '''
    
    explained_variance, model = do_pca_on_df(data_df=data_df, n_components=n_components, 
                                             expl_var_goal=expl_var_goal, verbose=verbose)
    
    print(f'Percentage of variance explained = { int(100*explained_variance) }%')
    
    comps_df = comp_fr_mdl(model, data_df=data_df, desc_df=desc_df, thr_to_show=thr_to_show , num_to_show=num_to_show, verbose=verbose)
    
    return comps_df
```

```python

```

```python
comp_fr_data( users_courses_df, course_desc_df, n_components=2, verbose=False).head(n=2)
```

----------------------------------------------------


## Plot some heatmaps of component loadings

```python
plot_heatmaps = True
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
    
    comps_cp_df               =  comps_cp_df.set_index(desc_name_col, inplace=False).filter(regex='%.*%', axis=1)
    comps_cp_df.index         =  comps_cp_df.index.str[:15]
    comps_cp_df               =  comps_cp_df.sort_values(  by=[compnames[pop_comp]], axis=0, ascending=False ).head(n=top_n)
    comps_cp_df_sorted        =  comps_cp_df.sort_values(  by=[compnames[sort_by_c]], axis=0, ascending=False )
    
    plt.figure(figsize=(19, 20))
    sns.heatmap(comps_cp_df_sorted, annot=False, cmap=cmap, center=0.00)
    
```

```python



```

```python
users_courses_comps_df = comp_fr_data(users_courses_df, course_desc_df, n_components=15, thr_to_show=0.15 , num_to_show=3)
```

```python
if plot_heatmaps: plot_pca_heatmap(users_courses_comps_df, cmap='PiYG', sort_by_c=5 , top_n=75, pop_comp=0)
```

```python
res_courses_comps_df = comp_fr_data(resources_courses_df, course_desc_df, n_components=18, thr_to_show=0.15 , num_to_show=3)
```

```python
if plot_heatmaps: plot_pca_heatmap(res_courses_comps_df, cmap='PiYG', sort_by_c=5 , top_n=63, pop_comp=1 )
```

```python
users_edutypes_comps_df = comp_fr_data(users_edutypes_df, edutype_desc_df, n_components=11, thr_to_show=0.15 , num_to_show=3)
```

```python
if plot_heatmaps: plot_pca_heatmap(users_edutypes_comps_df, cmap='RdBu_r', sort_by_c=1 , top_n=75, pop_comp=0 )
```

```python
res_edutypes_comps_df = comp_fr_data(resources_edutypes_df, edutype_desc_df, n_components=11, thr_to_show=0.15 , num_to_show=3)
```

```python
if plot_heatmaps: plot_pca_heatmap(res_edutypes_comps_df, cmap='RdBu_r' , sort_by_c=1 , top_n=25, pop_comp=0)
```

-------------------------------------------------------------------------


## Look at correlations of components from user and resource space 

```python
edutypes_comps_df = pd.merge(res_edutypes_comps_df, users_edutypes_comps_df, on=['edutype_id', 'education_type_name'], how='inner')
```

```python
edutypes_comps_df.head(n=2)
```

```python

```

```python
plt.figure(figsize=(13, 10))
sns.heatmap(edutypes_comps_df.filter(regex='f_', axis=1).corr().applymap(lambda x: x if abs(x) > 0.35 else 0)  , annot=False, cmap='RdBu_r', center=0.00)
```

```python
courses_comps_df = pd.merge(res_courses_comps_df, users_courses_comps_df, on=['course_id', 'course_name'], how='inner')
```

```python
plt.figure(figsize=(13, 10))
sns.heatmap(courses_comps_df.filter(regex='f_', axis=1).corr().applymap(lambda x: x if abs(x) > 0.35 else 0)  , annot=False, cmap='PiYG', center=0.00)
```

--------------------------------------------------------------

```python
importlib.reload(z_utilities)
from experiment_runner import run_experiments
from z_utilities import stack_dfs_common_column, df_to_matix, assert_nonan, assert_column_subset, valset
```

```python

```

```python
def make_pcaFeats_df(data_df, desc_df, n_components=None, expl_var_goal=None, verbose=True):
    '''
    Make a dataframe of PCA features for each user/res, based on a dataframe of original variables for each user/res
    '''
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    
    assert_nonan(data_df)
    assert_nonan(desc_df)
    assert (n_components is not None) or (expl_var_goal is not None)
    desc_id_col  = [x for x in desc_df.columns.tolist() if '_id'  in x ][0]
    data_cid_col = [x for x in data_df.columns.tolist() if '_cid' in x ][0]
    if not desc_id_col in data_df.columns.tolist(): raise ValueError(f'{desc_id_col} must be in {data_df.columns.tolist()}')
    assert_column_subset(data_df, desc_id_col, desc_df, desc_id_col)  

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    
    comps_df = comp_fr_data(data_df, desc_df, expl_var_goal=expl_var_goal, n_components=n_components, thr_to_show=0.15 , num_to_show=3, verbose=verbose)\
                          .filter(regex='%', axis=1)
    assert_nonan(comps_df)
    
    # We need to replace the values in data_df with contiguous values 0, .... n. For the data id column we have to get the set of data ids from the description df
    
    desc_ix_df = desc_df.reset_index(drop=True)[[desc_id_col]].drop_duplicates().sort_values(by=desc_id_col).reset_index(drop=False)
    assert_nonan(desc_ix_df)
    
    data_df_c = pd.merge(data_df, desc_ix_df, on=desc_id_col, how='left')[[data_cid_col, 'index']]
    assert_nonan(data_df_c)
    
    data_matrix  = df_to_matix(data_df_c, col1=data_cid_col, col2='index', numeric_type='int8', matrix_type='dense')

    comps_matrix = np.array(comps_df.values, dtype='float32')

    # ----------------------------------------------
    datcomps_mat = data_matrix @ comps_matrix
    # ----------------------------------------------
    
    datcomps_df = pd.DataFrame({ n:d for (n, d) in zip( comps_df.columns.tolist() , [datcomps_mat[:,i] for i in range(datcomps_mat.shape[1]) ]  )} )\
                              .reset_index(drop=False)\
                              .rename(columns={'index':data_cid_col})\
                              .set_index(data_cid_col)
    assert_nonan(datcomps_df)

    return datcomps_df 
```

-------------------------------------------------------

```python
users_courses_pca_df = make_pcaFeats_df(users_courses_df, course_desc_df, n_components=8, verbose=False)
display(users_courses_pca_df.head(n=1))
```

```python
res_courses_pca_df = make_pcaFeats_df(resources_courses_df, course_desc_df, n_components=7, verbose=False)
display(res_courses_pca_df.head(n=1))
```

```python
users_edutypes_pca_df = make_pcaFeats_df(users_edutypes_df, edutype_desc_df, n_components=7, verbose=False)    
display(users_edutypes_pca_df.head(n=1))
```

```python
res_edutypes_pca_df = make_pcaFeats_df(resources_edutypes_df, edutype_desc_df, n_components=7, verbose=False)  
display(users_edutypes_pca_df.head(n=1))
```

```python
res_keywords_pca_df = make_pcaFeats_df(resources_keywords_df[resources_keywords_df['keyword_count'] >= 9][['res_cid', 'keyword_id']], 
                                       keywords_desc_df, n_components=20, verbose=False)  
display(res_keywords_pca_df.head(n=1))
```

#### Use PCA features to try to predict the engagement score

```python
augmented_df.head(n=3)
```

```python
def augment_with_pca_datas(df, pca_data_df_s):
    
    if 'user_cid' or 'res_cid' in df.columns.tolist():
        df_w_pca = df.set_index(['user_cid', 'res_cid']).copy()
    else:    
        df_w_pca = df.copy()
    
    for pca_data_df in pca_data_df_s:
        print('.', end='')
        df_w_pca = pd.merge(df_w_pca, pca_data_df, left_index=True, right_index=True)

    df_w_pca = (lambda d, c2s : d.reindex(columns = c2s +[c for c in d.columns.tolist() if c not in c2s])) (df_w_pca, ['testset', 'f_bias_user', 'f_bias_res', 'eng_score'])
    
    return df_w_pca
```

```python
aug_wpca_df = augment_with_pca_datas(augmented_df, [users_courses_pca_df , res_courses_pca_df, users_edutypes_pca_df, res_edutypes_pca_df] )
```

```python
aug_wpca_df
```

```python
aug_wpca_df.columns
```

----------------------------------------------------------------------------------------------------

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(bias|.*_x_.*)')
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_noise')
```

```python
assert False
```

----------------------------------------------------------

```python
def add_product_features(df, product_list):
    '''
    product_list is a list of pairs of columns of df, must start with f_
    this function works in place for memory efficiency reasons
    '''
    
    for (col1, col2) in product_list:
        assert col1 in df.columns.tolist()
        assert col2 in df.columns.tolist()
        assert col1.startswith('f_')
        assert col2.startswith('f_')
        product_name = 'f_' + col1[2:] + '_x_' + col2[2:]
        df[product_name] = df[col1] * df[col2]

    return None
```

```python
product_list_svd = [  ('f_svd_user0', 'f_svd_res0'),  ('f_svd_user1','f_svd_res1'),  ('f_svd_user2','f_svd_res2')  ]
```

```python
add_product_features(aug_wpca_df, product_list_svd)
```

```python
product_list_basic_pca = [  ('f_userCourse_%NotRelated1%', 'f_resCourse_%NotRelated0%'),  ('f_userCourse_%BiologyPhysicsCh5%','f_resCourse_%NaturalScienceBi5%'),  ('f_userEdutype_%Primary3RdPrimar1%','f_resEdutype_%Primary4ThPrimar1%')  ]
```

```python
add_product_features(aug_wpca_df, product_list_basic_pca)
```

```python
aug_wpca_df.head(n=2).filter(regex='(f_.*_x_.*|^(?!f_).+)', axis=1)
```

```python

```

```python
# aug_wpca_df.columns
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_bias')
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(bias|noise)')
```

```python
'%.7f' % ( 0.7895145 - 0.78951466)
```

```python
'%.7f' % (  0.76482093 - 0.7648204)
```

```python

```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(.*%.*_x_.*|bias)')
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(svd.*_x_.*|bias)')
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(.*_x_.*|bias)')
```

```python
'%.7f' % (  0.70567006 -  0.7053416)
```

```python
%%time
train_linear_model(aug_wpca_df, feature_regex='f_(bias)')
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
# focus_col='NaturalScienceBi%4%'

# make_pcaFeats_df(resources_courses_df, course_desc_df, n_components=10, verbose=False).sort_values(by=focus_col, ascending=False)\
#                                                                                      .merge(res_cids_df, on='res_cid')\
#                                                                                      .merge(res_title_df, on='res_cid')\
#                                                                                      .set_index('res_id')\
#                                                                                      .drop(columns=['res_cid'])\
#                                                                                      .head(n=5)\
#                                                                                      [[focus_col, 'title']]
```

```python
# course_count_df = pd.merge(   course_desc_df  ,   pd.DataFrame(  {'course_n_users': users_courses_df.groupby('course_id')['user_cid'].count() }  )    ,   left_index=True, right_index=True    )\
#                     .sort_values(  by=['course_n_users'], axis=0, ascending=False )\
#                     .reset_index(drop=False)\
#                     .reset_index(drop=False)\
#                     .set_index('course_id')\
#                     .rename(columns={'index':'course_rank'})

# course_count_df.head(n=5)
```

```python
# display(( lambda df: df[df.isnull().any(axis=1)]  )  ( res_features_df  ))
# print(res_features_df.shape)
```

```python
# users_courses_edutypes_df = stack_dfs_common_column(users_courses_df, users_edutypes_df, common_name='user_cid', diff_name_1='course_id', diff_name_2='edutype_id')
# do_pca_on_df(users_courses_edutypes_df,  expl_var_goal=0.70)
# courses_edutypes_desc_df = stack_dfs_common_column(course_desc_df.rename(columns = {'course_name':'edu_or_crs_name' } ) , 
#                                                    edutype_desc_df.rename(columns = {'education_type_name':'edu_or_crs_name' } )  , 
#                                                    common_name='edu_or_crs_name',  diff_name_1='edutype_id',  diff_name_2='course_id')
```
