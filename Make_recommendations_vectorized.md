# Make recommendations from a dataframe of useful features - vectorized version

```python
!pip install --upgrade scikit-learn
!pip install --upgrade s3fs
!pip install --upgrade pandas
!pip install --upgrade numpy
!pip install --upgrade openpyxl
```

```python
!pip install --upgrade seaborn
```

```python
import pandas as pd
import s3fs
import numpy as np

from openpyxl.styles import Font
from openpyxl.styles.colors import Color

import matplotlib.pyplot as plt

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
filenames =  [ 'user_cids' ,  'res_cids', 'resources_title' , 'resources_status' ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    
    if 'title' or 'status' in df_name  :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8') 
    else                               :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 

user_cids_df          =   dfs['user_cids']  
res_cids_df           =   dfs['res_cids']   
resources_title_df    =   dfs['resources_title'] 
resources_status_df   =   dfs['resources_status'] 

```

```python

```

### Load res/user table

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
df_ratings_small = df_ratings.sample(n=500000).reset_index(inplace=False, drop=True)
df_ratings_full  = df_ratings

print('df_ratings_small   = {:,}'.format( int( float ('%.3g' % df_ratings_small.shape[0]  ) ) )   )
print('df_ratings_full    = {:,}'.format( int( float ('%.3g' % df_ratings.shape[0]        ) ) )   )
```

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
df_train_full, df_test_full = train_test_split_aug(df_ratings_full, test_size=0.1, random_state=0)
```

###### expect hashes:  
reXXomF$ ,  cmViUA0z

```python
print(f'{hash_df(df_train_full)}, {hash_df(df_test_full)}' )
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
        
user_feats_kc_df = feats_dfs['user']
res_feats_kc_df  = feats_dfs['res']    
```

```python
display(user_feats_kc_df.head(n=2))
print(user_feats_kc_df.shape)
```

```python
display(res_feats_kc_df.head(n=2))
print(res_feats_kc_df.shape)
```

-------------------------------------------------------------------------------

```python
wd_score_only   =   {
                     'favourited'        : 0,
                     'score'             : 1,
                     'clicked_through'   : 0,
                     'previewed'         : 0,
                     'downloaded'        : 0,
                     'used'              : 0,
                     'visited_detail_pg' : 0
                    }
```

--------------------------------------------------------------------------

```python

```

```python
def make_feats_df(user_res_set, user_feats_df, res_feats_df):
    
    ''' 
    Given a dataframe with users/resources and dataframes with user and resource features, 
    output a dataframe with users, resources and features.
    User bias features are currently thrown away since they cannot help with recommendations
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
    
    # Here we throw away the bias tems by not including them:
    combined_df = pd.concat([user_res_set[['res_cid', 'user_cid']].sort_values( by=['user_cid', 'res_cid'] , ascending=[False, False] ), combined_svd_df], axis=1) 
    
    return combined_df
```

```python
# .filter(regex='.*_bias.*').sort_values(by='f__favourited__bias', ascending=False).head(n=10)
```

```python
f_df = make_feats_df(df_train_full.head(n=2)[['res_cid', 'user_cid']], user_feats_kc_df, res_feats_kc_df)
```

```python
f_df
```

```python
df_train_full[['res_cid','user_cid']].head(n=2)
```

```python
def aggregate_feats_df(ua_feats_df, feature_categories):
    '''
    Given an unaggretated features dataframe, aggregate it by a set of eature categories
    '''
    
    aggr_feats_df = ua_feats_df[['res_cid', 'user_cid']].copy()
    
    for fc in feature_categories:
        aggr_feats_df[fc] =     ua_feats_df.filter(regex='f__' + fc + '.*', axis=1).sum(axis=1)
        
        # divide here since scores are on a 1-5 scale
        if fc == 'score': aggr_feats_df[fc] /= 3
            
    return aggr_feats_df
```

```python
agg_f_df = aggregate_feats_df(f_df, TARGETS_N)
agg_f_df.head(n=2)
```

```python
agg_f_df.sort_values(by='score', ascending=False).head(n=2)
```

```python
agg_f_df.sort_values(by='downloaded', ascending=False).head(n=2)
```

```python

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
        np.random.seed(0)
        rand_ser    = pd.Series( np.random.uniform( 0,1, agg_feats_df.shape[0])  , index=agg_feats_df[fc].index  )
        weighted_rs = (0 if total_weight > 0 else 1/(10*len(norm_weights_dict))  ) * rand_ser   
        
        # this is the main path if the total weight is greater than 0
        weighed_feat_sum_df['pred_eng_score'] += norm_weights_dict[fc]*agg_feats_df[fc] + weighted_rs
        
    return weighed_feat_sum_df
```

```python

```

```python

```

```python
equal_weight_final_df = get_weighted_from_agg_feats_df(agg_f_df, {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 } )
equal_weight_final_df
```

```python
def make_preds_from_user_res_set(user_res_id_df, weights_dict=None):
    '''
    Make predictions given just a set of users and resources and weights for the different data types.
    Default weighting is equal weight between the data types, but excluding favourited and score
    Assumes that the data is indexed by id
    '''
    if weights_dict==None:   weights_dict={'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }
        
    user_res_cid_df = replace_id_with_cid(user_res_id_df)
    
    f_df = make_feats_df(user_res_cid_df, user_feats_kc_df, res_feats_kc_df)
    
    agg_f_df = aggregate_feats_df(f_df, TARGETS_N) 

    return  replace_cid_with_id(get_weighted_from_agg_feats_df(agg_f_df, weights_dict ))

```

```python
make_preds_from_user_res_set(  replace_cid_with_id(df_train_full.head(n=2)[['res_cid', 'user_cid']])  )
```

```python

```

```python

```

```python
def make_recs_from_users_feats(user_cid_subset, user_feats_df, res_feats_df,  weights_dict, p_resources_status_df=resources_status_df, top_n=5 ): 
    
    '''
    Given a list of user cids, and user and res features and a weighting dictionary, produce a list of recommendations
    unpublished resources are excluded here
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
    
    weighted_final_df   =  get_weighted_from_agg_feats_df(agg_feats_df_ss, weights_dict )
    
    sorted_df           = weighted_final_df.sort_values( by=[ 'user_cid', 'pred_eng_score'], axis=0, ascending=[True, False] )\
                                           .reset_index(drop=True)\
                                           .groupby('user_cid').head(top_n)
    
    return sorted_df
```

```python
recs_169111 = make_recs_from_users_feats([169111], user_feats_kc_df, res_feats_kc_df, {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 } , top_n=10)\
                            .head(n=2)
recs_169111 
```

```python
def replace_cid_with_id(df, p_user_cids_df=user_cids_df, p_res_cids_df=res_cids_df):
    '''
    replace cids with ids in a dataframe
    '''
    
    df_rep = df.merge(p_user_cids_df, how='left', on='user_cid')\
               .merge(p_res_cids_df, how='left', on='res_cid')\
               .drop(['user_cid', 'res_cid'], axis=1)
    
    df_rep = (lambda d, c2s : d.reindex(columns= c2s + [c for c in d.columns.tolist() if c not in c2s] )) (df_rep, [ 'res_id' , 'user_id'])
    
    return df_rep
```

```python
def replace_id_with_cid(df, p_user_cids_df=user_cids_df, p_res_cids_df=res_cids_df):
    '''
    replace ids with cids in a dataframe
    '''
    
    df_rep = df.merge(p_user_cids_df, how='left', on='user_id')\
               .merge(p_res_cids_df, how='left', on='res_id')\
               .drop(['user_id', 'res_id'], axis=1)
    
    df_rep = (lambda d, c2s : d.reindex(columns= c2s + [c for c in d.columns.tolist() if c not in c2s] )) (df_rep, ['res_cid', 'user_cid'])
    
    return df_rep
```

```python
recs_169111 
```

```python
replace_cid_with_id(recs_169111)
```

```python
replace_id_with_cid(replace_cid_with_id(recs_169111))
```

```python

```

```python
user_cids_df[user_cids_df['user_id']==17735]
```

```python
def get_top_n_recs_df_from_ids(user_id_list, weights_dict, 
                               user_feats_df, res_feats_df, 
                               p_user_cids_df=user_cids_df, p_res_cids_df=res_cids_df, 
                               p_resources_title_df=resources_title_df, return_styled=False , top_n=5           ):
    
    '''
    Get a styled dataframe of recommendations from user ids and weightings and features dataframes for users and resources
    
    '''
    
    user_cid_subset_df = (  lambda d, c, l : d[d[c].isin(l)]  )  (  p_user_cids_df ,  'user_id' , user_id_list ) 
    user_cid_list      = user_cid_subset_df['user_cid'].values.tolist()
    
    recs_df            = make_recs_from_users_feats(user_cid_list, user_feats_df, res_feats_df, weights_dict, top_n=top_n )
    
    recs_w_title_df    = recs_df.merge(p_resources_title_df, how='left', on='res_cid')
    
    recs_df_repl       = replace_cid_with_id(recs_w_title_df, user_cids_df, res_cids_df)\
                           .sort_values( by=['user_id', 'pred_eng_score' ], axis=0, ascending=[True, False] )\
                           .reset_index(drop=True)

    recs_df_repl['user_link'] = recs_df_repl['user_id'].apply(lambda n: 'https://www.klascement.net/profiel/' + str(n)) 
    recs_df_repl['res_link'] = recs_df_repl['res_id'].apply(lambda n: 'https://www.klascement.net/leermiddel/' + str(n)) 
    
    recs_df_repl = (lambda d, c2e : d.reindex(columns=[c for c in d.columns.tolist() if c not in c2e] + c2e))\
                        (recs_df_repl, [ 'user_link',  'title', 'res_link', 'pred_eng_score'])
    
    recs_df_repl.set_index(['user_id', 'res_id'], inplace=True)
    
    def make_clickable(val):  return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    recs_styled_df = recs_df_repl.style.format({'user_link': make_clickable, 'res_link': make_clickable})
    
    return (recs_styled_df if return_styled else recs_df_repl)
```

```python
get_top_n_recs_df_from_ids( [ 17735]  , 
                            {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }, 
                            user_feats_kc_df, res_feats_kc_df, 
                            return_styled=True, 
                            top_n=15
                          )
```

```python

```

```python
def save_recs_to_s3(recs_df, filename, lang='nl'):
        filepath =  filename + '.xlsx'
        
        recommendations_str = {'en': 'Recommendations', 'nl': 'Aanbevelingen' }

        with io.BytesIO() as output:
            with pd.ExcelWriter(output) as writer:
                recs_df.to_excel(writer, encoding='utf-8', index=True,  float_format='%.2f',  sheet_name=recommendations_str[lang])
                
                worksheet = writer.sheets[recommendations_str[lang]]
                
                for col_L_col_W in list(zip(['A', 'B', 'C', 'D', 'E', 'F'], [12,12,45,60,50,16])):
                    worksheet.column_dimensions[col_L_col_W[0]].width = col_L_col_W[1]
                    
                for col_N in [5]:
                    for row_N in range(2, recs_df.shape[0]+1+1):
                        thiscell = worksheet.cell(row=row_N, column=col_N)
                        thiscell.hyperlink = thiscell.value
                        thiscell.style = "Hyperlink"

            data = output.getvalue()
            
        s3 = boto3.resource('s3')
        s3.Bucket(bucket).put_object(Key=filepath, Body=data)  
```

```python

```

```python

```

```python
vec_recs_17735 = get_top_n_recs_df_from_ids( [ 17735]  , 
                                                {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }, 
                                                user_feats_kc_df, res_feats_kc_df, 
                                                return_styled=False, 
                                                top_n=30
                                            )
```

```python

```

```python
if False: save_recs_to_s3(vec_recs_17735, filename='recs_vec_17735_noinvalids')
```

```python

```

```python
vec_recs_536107  = get_top_n_recs_df_from_ids( [ 536107]  , 
                                                    {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }, 
                                                    user_feats_kc_df, res_feats_kc_df, 
                                                    return_styled=False, 
                                                    top_n=20
                                                )
```

```python

```

```python
if False: save_recs_to_s3(vec_recs_536107, filename='vec_recs_536107')
```

```python

```

```python
vec_recs_54457  = get_top_n_recs_df_from_ids( [ 54457]  , 
                                                    {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   'used'  : 1,   'visited_detail_pg' : 1 }, 
                                                    user_feats_kc_df, res_feats_kc_df, 
                                                    return_styled=False, 
                                                    top_n=20
                                                )
```

```python
if False: save_recs_to_s3(vec_recs_54457, filename='vec_recs_54457')
```

```python

```

#### Create the devset 

```python

```

```python
devset_accounts = [5314, 5686, 19432, 131667, 28443, 281580, 20727, 221265, 46565, 248623, 244315, 16877, 
                   1105, 168008, 124884, 70470, 14823, 245142, 336226, 35483, 241023, 115066, 49322, 436326, 22030]
```

```python

```

```python
devset_recs_dl_etc = get_top_n_recs_df_from_ids( devset_accounts  , 
                                            {'favourited' : 0,  'score' : 0,  'clicked_through' : 1,  'previewed' : 1,   'downloaded' : 1,   
                                                  'used'  : 1,   'visited_detail_pg' : 1 }, 
                                            user_feats_kc_df, res_feats_kc_df, 
                                            return_styled=False, 
                                            top_n=6
                                         )
```

```python
devset_recs_score_fav = get_top_n_recs_df_from_ids( devset_accounts  , 
                                                    {'favourited' : 1,  'score' : 1,  'clicked_through' : 0,  'previewed' : 0,   'downloaded' : 0,   
                                                          'used'  : 0.1,   'visited_detail_pg' : 0 }, 
                                                    user_feats_kc_df, res_feats_kc_df, 
                                                    return_styled=False, 
                                                    top_n=4
                                                 )
```

```python
devset_recs_neg = get_top_n_recs_df_from_ids( devset_accounts  , 
                                            {'favourited' : 0,  'score' : 0,  'clicked_through' : 0,  'previewed' : 0,   'downloaded' : 0,   
                                                  'used'  : 0,   'visited_detail_pg' : 0 }, 
                                            user_feats_kc_df, res_feats_kc_df, 
                                            return_styled=False, 
                                            top_n=5
                                         )
```

```python

```

```python
devset_recs = pd.concat([devset_recs_dl_etc, devset_recs_score_fav, devset_recs_neg]).sort_values(by=['user_id', 'res_id'])
```

```python
devset_recs
```

```python
if False: save_recs_to_s3(devset_recs_dl_etc, filename='devset_recs_dl_etc')
```

```python
if False: save_recs_to_s3(devset_recs, filename='devset_recs')
```

```python

```

```python

```

#### Devsets as separate files

```python

```

```python
for user_id in devset_accounts: 
    
    frame_this_acc = devset_recs[devset_recs.index.get_level_values('user_id') == user_id].copy()
    
    frame_this_acc['pred_eng_score'] = ''
    
    frame_this_acc = frame_this_acc.rename(columns={'user_link': 'uw gebruikerspagina', 'title' : 'titel', 'res_link' :'leermiddel' , 
                                           'pred_eng_score': 'kwaliteitsbeoordeling (1 = slechtste, 5 = beste)'  }) 
    
    
    print(frame_this_acc.shape[0])
    
    if(frame_this_acc.shape[0] > 0):
        if False: save_recs_to_s3(frame_this_acc, filename='kc_devset_user=' + str(user_id))
    else:
        print('cannot make recommendations for ' + str(user_id))
```

```python

```

### Test quality of the Devset 

```python

```

```python
df_devset_filled_in = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, 'devset_filled_in_master.csv') )
```

```python
display(df_devset_filled_in.head(n=2))
print(len(df_devset_filled_in))
print( sorted(list(set(df_devset_filled_in['user_id'].values.tolist()) )) )
```

```python

```

```python
devset_preds = make_preds_from_user_res_set( df_devset_filled_in[['user_id', 'res_id']] )
```

```python
display(devset_preds.head(n=2))
print(len(devset_preds))
print( sorted(list(set(devset_preds['user_id'].values.tolist()) )) )
```

```python
devset_preds_ratings = df_devset_filled_in.merge(devset_preds, on=['user_id', 'res_id'], how='left')  
```

```python
display(devset_preds_ratings.head(n=2))
print(len(devset_preds_ratings))
```

```python
devset_preds_ratings.plot.scatter(x='pred_eng_score', y='rating', c='DarkBlue')
```

```python
devset_preds_ratings[['rating', 'pred_eng_score']].corr()
```

```python

```

```python

```
