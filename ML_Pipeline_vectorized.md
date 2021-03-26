# ML pipeline - vectorized version

```python
!pip install --upgrade scikit-learn
!pip install --upgrade s3fs
!pip install --upgrade pandas
!pip install --upgrade numpy
!pip install --upgrade openpyxl
```

```python
import pandas as pd
import s3fs
import numpy as np

from openpyxl.styles import Font
from openpyxl.styles.colors import Color

import io
import boto3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from z_utilities import hash_df
```

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
```

## Load cid/id tables 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
filenames =  [ 'user_cids' ,  'res_cids' , 'resources_status' ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    
    if 'title' or 'status' in df_name  :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8') 
    else                               :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 

user_cids_df          =   dfs['user_cids']  
res_cids_df           =   dfs['res_cids']   
resources_status_df   =   dfs['resources_status'] 

```

```python
TARGETS =         [
                    't_b_favourited'           , 
                    't_i_score'                , 
                    't_b_clicked_through'      , 
                    't_b_previewed'            , 
                    't_b_downloaded'           , 
                    't_b_used'                 , 
                    't_b_visited_detail_pg' 
                  ]

TARGETS_N = [t[4:] for t in TARGETS]
```

#### Load features dataframes

```python
feats_dfs = {}

for u_r in ['user', 'res']:
    filename = 'klascement_'+ u_r +'_vec_feats_df'
    data_location = 's3://{}/{}'.format(bucket, filename)

    feats_dfs[u_r]  = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, filename + '.csv')  ,
                                  dtype  = {   u_r+'_cid'      : 'int32'      }                              )
    
    for c in [c for c in feats_dfs[u_r].columns.tolist() if 'f_' in c]:     feats_dfs[u_r][c] = feats_dfs[u_r][c].astype('float32')
        
user_feats_kc_df = feats_dfs['user']
res_feats_kc_df  = feats_dfs['res']    
```

```python

```

-------------------------------------------------------------------------------

```python
def make_feats_df(user_res_set, user_feats_df, res_feats_df):
    
    ''' 
    Given a dataframe with users/resources and dataframes with user and resource features, 
    output a dataframe with users, resources and features.
    Bias features are currently thrown away since they cannot help with recommendations
    '''
    
    assert user_feats_df.columns.tolist()[1:] == res_feats_df.columns.tolist()[1:] 
    
    user_feats_ss_df = pd.merge(user_res_set[['user_cid']], user_feats_df, on='user_cid', how='left')
    res_feats_ss_df  = pd.merge(user_res_set[['res_cid']] , res_feats_df , on='res_cid' , how='left')  
    
    assert user_feats_ss_df.columns.tolist()[1:] == res_feats_ss_df.columns.tolist()[1:]
    
    user_svd_ss_df   = user_feats_ss_df.filter(regex='^f__.*_svd_.*')
    res_svd_ss_df    = res_feats_ss_df.filter(regex='^f__.*_svd_.*' )

    user_svd_matrix  = user_svd_ss_df.to_numpy()
    res_svd_matrix   = res_svd_ss_df.to_numpy()
    
    # Multiply the user features by the resource features
    #======================================================
    combined_svd_matrix = user_svd_matrix*res_svd_matrix
    #======================================================
    
    combined_svd_df = pd.DataFrame(combined_svd_matrix, columns=user_svd_ss_df.columns)    
    
    assert combined_svd_df.columns.tolist()[1:] == user_svd_ss_df.columns.tolist()[1:] 
    
    # Here we throw away the bias terms by not including them:
    combined_df = pd.concat([user_res_set[['res_cid', 'user_cid']].sort_values( by=['user_cid', 'res_cid'] , ascending=[False, False] ), combined_svd_df], axis=1) 
    
    return combined_df
```

```python
def aggregate_feats_df(ua_feats_df, feature_categories):
    '''
    Given an unaggretated features dataframe, aggregate it by a set of feature categories
    '''
    
    aggr_feats_df = ua_feats_df[['res_cid', 'user_cid']].copy()
    
    for fc in feature_categories:
        aggr_feats_df[fc] =     ua_feats_df.filter(regex='f__' + fc + '.*', axis=1).sum(axis=1)
        
        # divide here since scores are on a 1-5 scale
        # TODO: 3 is a magic number here 
        if fc == 'score': aggr_feats_df[fc] /= 3
            
    return aggr_feats_df
```

```python
def get_weighted_from_agg_feats_df(agg_feats_df, weights_dict):
    '''
    given a dataframe of multiplied features for users and resources, and a weights dictionary, generate weighted engagement 
    '''
    total_weight = sum([abs(w) for w in weights_dict.values() ])
    norm_weights_dict = {k:w/total_weight for (k, w) in weights_dict.items()} if total_weight > 0 else {k:0 for (k, w) in weights_dict.items()}
    
    weighed_feat_sum_df = agg_feats_df[['res_cid', 'user_cid']].copy()
    weighed_feat_sum_df['pred_eng_score'] = 0
    
    for fc in list(weights_dict.keys()):
        
        # if all the weights are 0, we will rank recommendations randomly 
        rand_ser    = pd.Series( np.random.uniform(0,1, agg_feats_df.shape[0])  , index=agg_feats_df[fc].index  )
        weighted_rs = (0 if total_weight > 0 else 1/(10*len(norm_weights_dict))  ) * rand_ser   
        
        # this is the main path if the total weight is greater than 0
        weighed_feat_sum_df['pred_eng_score'] += norm_weights_dict[fc]*agg_feats_df[fc] + weighted_rs
        
    return weighed_feat_sum_df
```

```python
def make_recs_from_users_feats(user_cid_subset, user_feats_df, res_feats_df,  weights_dict, p_resources_status_df=resources_status_df, top_n=5 ): 
    
    '''
    Given a list of user cids, and user and res features and a weighting dictionary, produce a list of recommendations.
    Unpublished resources are excluded here
    '''
    
    def cartesian_product_basic(left, right): 
        return left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1)
    
    
    res_feats_df.reset_index(drop=True, inplace=True)
    
    # Excude unpublished resouces as they are not useful recommendations
    res_feats_df = pd.merge(res_feats_df, p_resources_status_df, on='res_cid' )
    res_feats_df = res_feats_df[res_feats_df['status_approved'] != False].drop(columns='status_approved')
    
    all_res_df          = res_feats_df[['res_cid']]

    user_cid_subset_df  = pd.DataFrame(   {'user_cid': pd.Series(list(set(user_cid_subset)), dtype='int32')  }    )
    
    users_ss_res_df     = cartesian_product_basic(user_cid_subset_df , all_res_df)
    
    feats_ss_df         = make_feats_df(users_ss_res_df, user_feats_df, res_feats_df)
    
    agg_feats_df_ss     = aggregate_feats_df(feats_ss_df, list( weights_dict.keys() ) )
    
    weighted_final_df   = get_weighted_from_agg_feats_df(agg_feats_df_ss, weights_dict )
    
    sorted_df           = weighted_final_df.sort_values( by=[ 'user_cid', 'pred_eng_score'], axis=0, ascending=[True, False] )\
                                           .reset_index(drop=True)\
                                           .groupby('user_cid').head(top_n)
    
    return sorted_df
```

```python
def replace_cid_with_id(df, p_user_cids_df, p_res_cids_df):
    '''
    replace cids with ids in a datafarame
    '''
    
    df_rep = df.merge(p_user_cids_df, how='left', on='user_cid')\
               .merge(p_res_cids_df, how='left', on='res_cid')\
               .drop(['user_cid', 'res_cid'], axis=1)
    
    df_rep = (lambda d, c2e : d.reindex(columns=[c for c in d.columns.tolist() if c not in c2e] + c2e))\
             (df_rep, ['pred_eng_score'])
    
    return df_rep
```

```python
def get_top_n_recs_list_from_ids(user_id_list, weights_dict, 
                                 user_feats_df, res_feats_df, 
                                 p_user_cids_df=user_cids_df, p_res_cids_df=res_cids_df, 
                                 top_n=5                                                  ):
    
    '''
    Get a list of recommendations from user ids and weightings and features dataframes for users and resources
    
    '''
    
    user_cid_subset_df = (  lambda d, c, l : d[d[c].isin(l)]  )  (  p_user_cids_df ,  'user_id' , user_id_list ) 
    user_cid_list      = user_cid_subset_df['user_cid'].values.tolist()
    
    recs_df            = make_recs_from_users_feats(user_cid_list, user_feats_df, res_feats_df, weights_dict, top_n=top_n )
    
    recs_df_repl       = replace_cid_with_id(recs_df, user_cids_df, res_cids_df)\
                           .sort_values( by=['user_id', 'pred_eng_score' ], axis=0, ascending=[True, False] )\
                           .reset_index(drop=True)


    recs_df_repl = (lambda d, c2e : d.reindex(columns=[c for c in d.columns.tolist() if c not in c2e] + c2e))\
                        (recs_df_repl, ['pred_eng_score'])\
                        [['user_id','res_id']]
    
    recs_dict = {}
    
    for user_id in user_id_list:
        recs_dict[user_id] = recs_df_repl[recs_df_repl['user_id'] == user_id]['res_id'].values.tolist()
    
    return recs_dict
```

```python

```

```python
def get_top_n_recs_list_from_ids_standard_paras(user_id_list, top_n):
    
    assert isinstance(top_n, int)
    assert all(isinstance(user_id, int) for user_id in user_id_list)
    assert all( user_id > 0 for user_id in user_id_list)
    assert 0 < len(user_id_list) < 10**5
    assert 0 < top_n < 10**5
    assert len(user_id_list)*top_n <= 10**7
    
    
    return get_top_n_recs_list_from_ids( user_id_list  , 
                                {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }, 
                                user_feats_kc_df, res_feats_kc_df, 
                                top_n=top_n
                            )
```

```python
get_top_n_recs_list_from_ids_standard_paras( [1, 17735], 10)
```

```python

```
