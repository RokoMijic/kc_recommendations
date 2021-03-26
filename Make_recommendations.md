# Make recommendations from a dataframe of useful features

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

```python
pd.set_option('display.max_rows', 1000)
```

## Load cid/id tables 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
filenames =  [ 'user_cids' ,  'res_cids', 'resources_title'  ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    
    if 'title' in df_name  :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8') 
    else                   :  dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8', dtype='int32') 

    
user_cids_df        =   dfs['user_cids']  
res_cids_df         =   dfs['res_cids']   
resources_title_df  =   dfs['resources_title'] 

```

#### Load features dataframe

```python
feats_dfs = {}

for u_r in ['user', 'res']:
    filename = 'klascement_'+ u_r +'_features_df'
    data_location = 's3://{}/{}'.format(bucket, filename)

    feats_dfs[u_r]  = pd.read_csv(  filepath_or_buffer = 's3://{}/{}'.format(bucket, filename + '.csv')  ,
                                  dtype  = {   u_r+'_cid'      : 'int32'      }                              )
    
    for c in [c for c in feats_dfs[u_r].columns.tolist() if 'f_' in c]:     feats_dfs[u_r][c] = feats_dfs[u_r][c].astype('float32')
        
feats_df_user = feats_dfs['user']
feats_df_res  = feats_dfs['res']    
```

```python
display(feats_df_user.head(n=2))
print(feats_df_user.shape[0])
```

```python
display(feats_df_res.head(n=2))
print(feats_df_res.shape[0])
```

-------------------------------------------------------

```python
use_user_bias=False
use_res_bias=True
```

----------------------------------

```python
best_model_coeffs_dict =    {'f_svd_user0_x_res0': 1.0337291955947876,
                             'f_svd_user1_x_res1': 1.0316495895385742,
                             'f_svd_user2_x_res2': 1.032457709312439,
                             'f_bias_user': 0.98689204454422,
                             'f_bias_res': 0.9960680603981018}

best_model_intercept = 2.1983755

best_model_coeffs_dict['f_bias_user'] *= use_user_bias
best_model_coeffs_dict['f_bias_res']  *= use_res_bias

best_model_coeffs_dict
```

```python
class LinearPredictionModel(LinearRegression):

    def __init__(self, coef=None, intercept=None):
        if coef is not None and intercept is not None:
            coef = np.array(coef)
            intercept = np.array(intercept)
            if len(intercept.shape) > 0 and (not coef.shape[0] == intercept.shape[0]): 
                raise ValueError(f"Coeff shape0 ({coef.shape[0]}) != intercept shape0 ({intercept.shape[0]})")
        else:
            raise ValueError("Please provide both the coefs and intercept")
            
        self.intercept_ = intercept
        self.coef_ = coef

    def fit(self, X, y):  raise NotImplementedError("This model is only for prediction")
```

```python
best_model = LinearPredictionModel(coef= list(best_model_coeffs_dict.values() ), intercept = best_model_intercept )
```

----------------------------------------------

```python
def make_feats_df(user_cid_subset, user_feats_df, res_feats_df, iLinearPredictionModel, model_colnames, top_n=5):
    
    def cartesian_product_basic(left, right): 
        return left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1)
   
    res_feats_df.reset_index(drop=True, inplace=True)

    user_cid_subset_df = pd.DataFrame(   {'user_cid': pd.Series(list(set(user_cid_subset)), dtype='int32')  }  )
    
    user_cid_subset_feats_df = pd.merge(user_feats_df, user_cid_subset_df, how='right', on='user_cid')\
                                    .sort_values( by=['user_cid'], axis=0, ascending=True )\
                                    .reset_index(drop=True).astype({'user_cid':'int32'})
    
    nan_users = (  lambda df: df[df.isnull().any(axis=1)]  ) (user_cid_subset_feats_df)  ['user_cid'].values.tolist()
    
    if len(nan_users) > 0 : 
        users_not_l = '[' + ', '.join([str(x) for x in nan_users[:3] ] ) + ' ...]' if len(nan_users) > 3 else ']'
        print(f"Warning: {len(nan_users)} users {users_not_l} are not in the training dataset; will get bad recs.")
        
    user_cid_subset_features_fillna_df  = (  lambda df :df.fillna(df.mean())  )  (user_cid_subset_feats_df)
    
    feats_df = cartesian_product_basic(user_cid_subset_features_fillna_df , res_feats_df)
    
    for i in range(len(model_colnames)-2): 
        feats_df['f_svd_user'+str(i)+'_x_res'+str(i)] = feats_df['f_svd_user'+str(i)]*feats_df['f_svd_res'+str(i)]
        del feats_df['f_svd_user'+str(i)]
        del feats_df['f_svd_res'+str(i)]
        
    feats_df['pred_eng_score'] = iLinearPredictionModel.predict(feats_df[model_colnames])   

    feats_df = (lambda d, c2e : d.reindex(columns=[c for c in d.columns.tolist() if c not in c2e] + c2e))\
                  (feats_df, ['f_bias_user', 'f_bias_res', 'pred_eng_score'])
    
    feats_df = feats_df.sort_values( by=['user_cid', 'pred_eng_score' ], axis=0, ascending=[True, False] )\
                             .reset_index(drop=True)
    
    features_top_n_df = feats_df.groupby('user_cid').head(top_n)

    return features_top_n_df[['user_cid', 'res_cid', 'pred_eng_score']]
```

```python
%%time
feats_df = make_feats_df(  list(range(10))  , feats_df_user, feats_df_res , best_model , 
                         list(best_model_coeffs_dict.keys()) 
                        )
```

```python
feats_df.head(n=2)
```

```python
def replace_cid_with_id(df, p_user_cids_df, p_res_cids_df):
    
    df_rep = df.merge(p_user_cids_df, how='left', on='user_cid')\
               .merge(p_res_cids_df, how='left', on='res_cid')\
               .drop(['user_cid', 'res_cid'], axis=1)
    
    df_rep = (lambda d, c2e : d.reindex(columns=[c for c in d.columns.tolist() if c not in c2e] + c2e))\
             (df_rep, ['pred_eng_score'])
    
    return df_rep
```

```python
replace_cid_with_id(feats_df, user_cids_df, res_cids_df).head(n=2)
```

```python
def get_top_n_recs(user_id_list, p_user_cids_df=user_cids_df, p_res_cids_df=res_cids_df, 
                   p_resources_title_df=resources_title_df, top_n=1, return_styled=False ):
    
    user_cid_subset_df = (  lambda d, c, l : d[d[c].isin(l)]  )  (  p_user_cids_df ,  'user_id' , user_id_list ) 
    
    user_cid_list      = user_cid_subset_df['user_cid'].values.tolist()
    
    recs_df = make_feats_df(user_cid_list, feats_df_user, feats_df_res , best_model , 
                               list(best_model_coeffs_dict.keys()) , top_n=top_n)
    
    recs_w_title_df = recs_df.merge(p_resources_title_df, how='left', on='res_cid')
    
    recs_df_repl = replace_cid_with_id(recs_w_title_df, user_cids_df, res_cids_df)\
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
get_top_n_recs(list(range(5)))
```

```python

```

```python
def save_recs_to_s3(recs_df, filename):
        filepath =  filename + '.xlsx'

        with io.BytesIO() as output:
            with pd.ExcelWriter(output) as writer:
                recs_df.to_excel(writer, encoding='utf-8', index=True,  float_format='%.2f', 
                                    sheet_name='Recommendations')
                
                worksheet = writer.sheets['Recommendations']
                
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

-------------------------------------

```python
klascement_test_nums = [5553, 17735, 35808, 52975, 54457, 106783, 156568, 174798, 
                        217794, 264895, 298323, 306116, 333274, 480208, 536107]
```

```python
klasc_sec_test_nums = [14317, 44068, 28088, 405934, 3009, 17460, 54420, 23397, 91811, 5995, 4368]
```

```python
save_recs=False
```

```python
if save_recs:
    save_recs_to_s3(get_top_n_recs( [54457], top_n=15 ), filename='recs_for_54457')
    save_recs_to_s3(get_top_n_recs( klascement_test_nums, top_n=15 ), filename='recs_for_first_testset')
    save_recs_to_s3(get_top_n_recs( klasc_sec_test_nums, top_n=15 ), filename='recs_for_second_testset')
else: 
    print('Recommendations not saved')
```

```python

```

---------------------------------------------

```python
get_top_n_recs( [54457], top_n=15 )
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
# (lambda d, c, v: d[d[c]==v] ) (test_recs_df, 'user_id' , 17735  )
```
