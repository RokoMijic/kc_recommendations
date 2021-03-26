# Noninteractive analysis 


Analysis of the user and resouce data without considering interaction

```python
!pip install --upgrade pandas
!pip install --upgrade swifter
!pip install --upgrade s3fs
!pip install --upgrade scipy
!pip install --upgrade numpy 
!pip install --upgrade scikit-learn
!pip install --upgrade numba
!pip install umap-learn
!pip install umap-learn[plot]
```

```python

```

```python
import numpy as np
import pandas as pd
import swifter

import matplotlib.pyplot as plt

import scipy.sparse as sparse

import s3fs
import numba
import umap
import umap.plot
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html


from sklearn import decomposition

from random import random
```

```python
pd.set_option('display.max_rows', 1000)
pd.set_option('max_colwidth', 150)
```

```python
print('The pandas version is {}.'.format(pd.__version__))
print('The numba version is {}.'.format(numba.__version__))
print('The umap version is {}.'.format(umap.__version__))
print('The s3fs version is {}.'.format(s3fs.__version__))
```

### Load data about users and resources

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
filenames =  [  'course_desc' , 'organising_subjects'  , 'course_cats',  'educ_type_desc' ,  'users_courses' ,      'users_edutypes' ,    
              'resources_courses'  ,   'resources_edutypes'  ,   'resources_keywords'  ,   'resources_title'  ,    'keywords_desc' ]  

dfs = {}

for df_name in filenames:

    data_location = 's3://{}/{}'.format(bucket, df_name)
    dfs[df_name] = pd.read_csv( data_location + '.csv' ,  encoding='utf-8') 
    
course_desc_df          =   dfs['course_desc'] 
course_order_df         =   dfs['organising_subjects'].drop(columns='name').rename(columns={'meaningful_course_num':'course_val'})
course_cats_df          =   dfs['course_cats'] 
educ_type_desc_df       =   dfs['educ_type_desc'] 
users_courses_df        =   dfs['users_courses']  
users_edutypes_df       =   dfs['users_edutypes']   
resources_courses_df    =   dfs['resources_courses']   
resources_edutypes_df   =   dfs['resources_edutypes'] 
resources_keywords_df   =   dfs['resources_keywords'] 
resources_title_df      =   dfs['resources_title'] 
keywords_desc_df        =   dfs['keywords_desc']
```

```python
print(f'Number of resources = {resources_title_df.shape[0]}')
```

##### Create meaningful subject values for resources

```python
res_vals_df = resources_courses_df.merge(course_order_df, on='course_id', how='left').merge(course_desc_df, on='course_id', how='left')

general_subjects_val = 128.0

res_course_crsvals_df = pd.DataFrame(  { 'res_crsval'   : res_vals_df.groupby('res_cid')['course_val' ].mean()                                  , 
                                         'res_crsstd'   : res_vals_df.groupby('res_cid')['course_val' ].std().fillna(0)                         , 
                                        }                                                                                                             ).reset_index()

res_course_crsvals_df.loc[res_course_crsvals_df['res_crsval'].isnull() , 'res_crsstd']      =     res_course_crsvals_df['res_crsstd'].max()
res_course_crsvals_df['res_crsval' ]                                                        =     res_course_crsvals_df['res_crsval' ].fillna(general_subjects_val)

res_course_crsvals_df['res_crsval' ] = res_course_crsvals_df['res_crsval'] / res_course_crsvals_df['res_crsval'].max()
res_course_crsvals_df['res_crsstd' ] = res_course_crsvals_df['res_crsstd'] / res_course_crsvals_df['res_crsstd'].max()

```

```python
res_course_crsvals_df.head(n=2)
```

---------------------------------------------------------------------------------


##### Pick only keywords with a lot of uses

```python
resources_keyw_lim_df  = resources_keywords_df.loc[resources_keywords_df['keyword_count'] >= 200].drop(columns=['keyword_count'])
keywords_d_lim_df      = keywords_desc_df.loc[keywords_desc_df['count'] >= 200]
print(f'limited number of keywords (more than 200 uses) = {keywords_d_lim_df.shape[0]}')

resources_keyw_v_lim_df = resources_keywords_df.loc[resources_keywords_df['keyword_count'] >= 400].drop(columns=['keyword_count'])
keywords_d_v_lim_df     = keywords_desc_df.loc[keywords_desc_df['count'] >=400]
print(f'very limited number of keywords (more than 400 uses) = {keywords_d_v_lim_df.shape[0]}')


# forget count data
keywords_desc_df        =   keywords_desc_df.drop(columns=['count'])
```

----------------------------------------------------------


##### Add overall course categories to data, e.g. "physics" will also get the "science" course

```python
res_cats_df = resources_courses_df.merge(course_cats_df, on='course_id', how='left').drop_duplicates()   
res_cats_df = res_cats_df.drop(res_cats_df[res_cats_df["course_id"]==res_cats_df["course_cat_id"]].index).reset_index(drop=True)\
                                                                                                         .drop(columns='course_id')\
                                                                                                         .drop_duplicates()\
                                                                                                         .rename(columns={'course_cat_id':'course_id'})
```

```python
display(res_cats_df.head(n=2))
res_cats_df.shape
```

-------------------------------------------------------------------------------------------


## Analyse education types of users 

```python
def df_to_csr(df, id_col, value_col): 
    return sparse.coo_matrix(( [1]*df.shape[0], (df[id_col], df[value_col]))).tocsr().astype('int64')
```

```python
edutypes_subset_df = users_edutypes_df[users_edutypes_df['user_cid'] < 5000].rename(columns={'user_cid':'cid'})
edutypes_mat  = df_to_csr( edutypes_subset_df, id_col='cid', value_col = 'edutype_id' ).todense()
```

##### Map education levels into 1 dimension using UMAP

```python
edu_embedding_df = edutypes_subset_df[['cid']].drop_duplicates().copy()
```

```python
%%time
# use 500 epochs for good results 
embedding = umap.UMAP( n_components=1,  n_epochs=11 , min_dist=1, n_neighbors=250, metric='euclidean').fit( edutypes_mat  )
```

```python
emb_vals = embedding.transform(edutypes_mat )
edu_embedding_df['embedding_val_umap']  = emb_vals
```

##### Map education levels into 1 dimension using PCA

```python
%%time
pca = decomposition.PCA(n_components=1, whiten=True )
pca.fit(edutypes_mat)
embedding_PCA = pca.transform(edutypes_mat)
edu_embedding_df['embedding_val_pca']  = embedding_PCA
```

##### show embedding values

```python
edutypes_embedding_df  = pd.merge(edutypes_subset_df, edu_embedding_df, on='cid', how='left').drop(columns=['cid']).groupby(['edutype_id']).mean().reset_index(drop=False) 

if edutypes_embedding_df[edutypes_embedding_df['edutype_id']==116]['embedding_val_umap'].values[0] < 0    :    edutypes_embedding_df['embedding_val_umap'] *= -1 

ranks_df = edutypes_embedding_df[["embedding_val_umap", "embedding_val_pca" ]].rank()
edutypes_embedding_df[["embedding_rank_umap", "embedding_rank_pca" ]]  = (ranks_df)/ranks_df.max()
edu_names_embedding_df = pd.merge( educ_type_desc_df  , edutypes_embedding_df, on='edutype_id', how='left').sort_values('embedding_val_umap').dropna().reset_index(drop=True)
```

```python
fig, ax = plt.subplots( figsize=(1, 20) )
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

xcoln  = 'embedding_rank_pca'
ycoln  = 'embedding_rank_umap'
txtoln = 'education_type_name'
edu_names_embedding_df.plot( xcoln , ycoln , kind='scatter',  ax=ax )
for k, v in edu_names_embedding_df.iterrows():    
    ax.annotate(s=v[txtoln], xy = (v[xcoln] , v[ycoln] ), ha='left', rotation=0 )
```

--------------------------------------------------------------------------------


## 2-D Data Visualization with UMAP

```python
def plot_UMAP(data_df, labels_df, titles_df, colors_df, subsetsize, n_epochs=350 , min_dist=0.75, n_neighbors=30, metric='euclidean', plot_noninteractive=False ):
    
    cid_name       =   [x for x in data_df.columns.tolist()   if '_cid' in x    ][0]
    data_id_name   =   [x for x in data_df.columns.tolist()   if '_id'  in x    ][0]
    data_desc_name =   [x for x in labels_df.columns.tolist() if '_id'  not in x][0]
    
    color_non_cid_name = [x for x in colors_df.columns.tolist() if '_cid'  not in x][0]
    
    
    def df_to_csr(df, id_col, value_col): 
        
        if 'weight' in df.columns.tolist():
            return sparse.coo_matrix(  arg1=( df['weight'], (df[id_col], df[value_col])   )  ).tocsr()
        else:
            return sparse.coo_matrix(  arg1=( [1]*df.shape[0], (df[id_col], df[value_col])  )  ).tocsr()
    
    
    data_subset_df = data_df[data_df[cid_name] < subsetsize]
    data_ss_mat  = df_to_csr( data_subset_df , id_col=cid_name, value_col = data_id_name ).todense()  
    
    embedding_df = data_subset_df[[cid_name]].drop_duplicates().copy()
    
    if not embedding_df.shape[0] == data_ss_mat.shape[0]: 
        raise ValueError(  f'Embedding_df shape ({embedding_df.shape[0]}) =/= data_ss_mat shape ({data_ss_mat.shape[0]}) -- likely caused by some missing cids in data_df ' )
        
    embedding = umap.UMAP(n_components=2,  n_epochs=n_epochs , min_dist=min_dist, n_neighbors=n_neighbors, metric=metric).fit( data_ss_mat  )
    emb_vals = embedding.transform(data_ss_mat )
    embedding_df['val_umap_1'] ,  embedding_df['val_umap_2']    =   emb_vals[:, 0]  , emb_vals[:, 1]
    
#     data_last_desc_df =  (pd.merge(data_subset_df.drop_duplicates(subset=cid_name , keep='first') , labels_df, on=data_id_name, how='left'  ).fillna('???').drop(columns=[data_id_name])).sort_values(cid_name).reset_index(drop=True)
    
    tags_or_subjects_concat = {'tags_or_subjects' : pd.merge(data_subset_df  , labels_df, on=data_id_name, how='left'  )\
                                                        .fillna('')\
                                                        .groupby([cid_name])[data_desc_name].apply(lambda x: ' | '.join(x))    
                              } 
    
    data_full_desc_df =  pd.DataFrame( tags_or_subjects_concat ).merge(titles_df, on=cid_name).drop(columns=[cid_name])
    
    if plot_noninteractive: umap.plot.points(embedding, values = colors_df[colors_df[cid_name] < subsetsize][color_non_cid_name]  , cmap='jet'  , height=1000, width=1900, show_legend=False  )

    p = umap.plot.interactive(embedding, values = colors_df[color_non_cid_name] , cmap='jet' ,  hover_data=data_full_desc_df, point_size=6, height=650, width=1300 )
    umap.plot.show(p)

    html = file_html(p, CDN, "my plot")
    
    return embedding_df, html
```

```python
def save_bokeh_plot(filepath, html_content): 
    fs = s3fs.S3FileSystem()
    with fs.open(filepath, 'w') as f:
        f.write(html_content)
    
```

```python
def stack_dfs(df1_in, df2_in, weight1=None, weight2=None, increm_ids=True):
    ''' stack two dataframes together, incrementing id numbers to keep the id column contiguous 
    '''
    
    if (weight1 is None and weight2 is not None) or (weight1 is not None and weight2 is None):
        raise ValueError("Give either both weights or neither")
        
    has_weights = True if weight1 is not None else False
        
    df1 = df1_in.copy()
    df2 = df2_in.copy()
    
    id_name_1       =   [x for x in df1.columns.tolist()   if  '_id' in x    ][0]
    id_name_2       =   [x for x in df2.columns.tolist()   if  '_id' in x    ][0]
    
    combined_id_name = id_name_1 if id_name_1 == id_name_2 else  id_name_1 + '_or_' + id_name_2
    
    other_name_1         =   [x for x in df1.columns.tolist()   if x != id_name_1 and x != 'weight'  ][0]
    other_name_2         =   [x for x in df2.columns.tolist()   if x != id_name_2 and x != 'weight'  ][0]
    
    if has_weights:
        for df in [df1, df2]:
            if 'weight' not in df.columns.tolist():
                df['weight'] = 1.0
        
        df1['weight'] = df1['weight']*weight1
        df2['weight'] = df2['weight']*weight2
    
    combined_other_name = other_name_1 if other_name_1 == other_name_2 else other_name_1 + '_or_' + other_name_2
        
    offset_val = max(df1[id_name_1].values.tolist()) + 1 if increm_ids else 0

    df2[id_name_2] = df2[id_name_2] + offset_val
    
    df1 = df1.rename(columns={id_name_1:combined_id_name, other_name_1:combined_other_name })
    df2 = df2.rename(columns={id_name_2:combined_id_name, other_name_2:combined_other_name })
    
    return df1.append(df2, ignore_index=True)
```

```python
stack_dfs(stack_dfs(resources_courses_df, res_cats_df, 1.0, 0.5, increm_ids=False), resources_keyw_v_lim_df , 1.0 , 0.25 ).sample(n=5)
```

----------------------------------------------------------

```python
output_notebook()
```

---------------------------------------------------------------------

```python

```

```python

```

#### Analyze courses per user

```python
# %%time
# users_courses_embedding, html_users_courses = plot_UMAP(users_courses_df, labels_df=course_desc_df, subsetsize=3000, n_epochs=350 , min_dist=0.90, n_neighbors=30, metric='euclidean')
```

```python

```

```python
# save_bokeh_plot(  's3://{}/{}'.format(bucket, 'bokehplot_users_courses.html') ,   html_users_courses    )
```

```python
# %%time
# users_courses_embedding, html_users_courses = plot_UMAP(users_courses_df, labels_df=course_desc_df, subsetsize=60000, n_epochs=350 , min_dist=0.90, n_neighbors=30, metric='euclidean')
```

```python
# save_bokeh_plot(  's3://{}/{}'.format(bucket, 'bokehplot_users_courses.html') ,   html_users_courses    )
```

```python
# assert False
```

#### Analyze courses per resource

```python
%%time
r_c_k11_emb, html_r_c_k11 = plot_UMAP( stack_dfs(stack_dfs(res_cats_df, resources_courses_df,  1.0, 0.5, increm_ids=False), resources_keyw_lim_df , 1.0 , 0.5 ), 
                                       labels_df=stack_dfs(course_desc_df, keywords_d_lim_df), 
                                       titles_df=resources_title_df,
                                       colors_df=res_course_crsvals_df[['res_cid','res_crsval']],    
                                       subsetsize=76056, 
                                       n_epochs=80 , 
                                       min_dist=0.999, 
                                       n_neighbors=40, 
                                       metric="euclidean", 
                                       plot_noninteractive=False   
                                 )
```

```python

```

```python
# save_bokeh_plot(  's3://{}/{}'.format(bucket, 'bokehplot_res_courses_kw_3.html') ,   html_r_c_k11    )
```

```python

```

#### Analyze courses mixed users and resources

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
# %%time
# embedding_hel = umap.UMAP(n_components=2, n_epochs=50 , min_dist=1.0, n_neighbors=10, metric='hellinger').fit( users_edutypes_sp_csr )
```

```python
# umap.plot.points(embedding_hel, values = embedding_1D.embedding_ , height=1500, width=1500, show_legend=False  )
```

```python
# umap.plot.points(embedding_hel,  labels=users_fil_df.last_edutype_id, height=1500, width=1500, show_legend=False  )
```

```python

```

```python
assert False
```

```python

```

```python
# res_edutypes_subset_df   =  resources_edutypes_df[(5000 <= resources_edutypes_df['res_cid']) & (resources_edutypes_df['res_cid'] < 10000) ].rename(columns={'res_cid':'cid'}) 
# edutypes_subset_df = users_edutypes_subset_df.append(res_edutypes_subset_df, ignore_index=True).reset_index(drop=True)
# edutypes_subset_df = resources_edutypes_df[resources_edutypes_df['res_cid'] < 5000 ].rename(columns={'res_cid':'cid'})
```

```python
# counts_df = users_edutypes_df.groupby(['user_cid']).count().reset_index(drop=False).rename(columns={'edutype_id': 'edutype_id_count'})
# users_edulast_df = users_edutypes_df.drop_duplicates(subset='user_cid' , keep='last')
# users_edulast_df = pd.merge(users_fil_df, educ_type_desc_df, on='edutype_id', how='left').fillna('???').rename(columns={'edutype_id': 'last_edutype_id'})
# display(users_fil_df.head(n=3))
# print(users_fil_df.shape)
```

```python
# if edutypes_embedding_df[edutypes_embedding_df['edutype_id']==116]['embedding_val_umap'].values[0] < 0 : 
#     edutypes_embedding_df['embedding_val_umap'] *= -1 
    
# if edutypes_embedding_df[edutypes_embedding_df['edutype_id']==116]['embedding_val_pca'].values[0] < 0 : 
#     edutypes_embedding_df['embedding_val_pca'] *= -1 
```

```python
# users_fil_df[users_fil_df.isnull().any(axis=1)]
```

```python
# users_edutypes_dense = users_edutypes_sp_csr.todense()
# users_edutypes_dense
# users_edutypes_dense.shape[0]*users_edutypes_dense.shape[1]
```

```python
# courses_subset_df = users_courses_df[users_courses_df['user_cid'] < 35000].rename(columns={'user_cid':'cid'})
# courses_mat  = df_to_csr( courses_subset_df , id_col='cid', value_col = 'course_id' ).todense()
# crs_embedding_df = courses_subset_df[['cid']].drop_duplicates().copy()

# %%time
# embedding_crs = umap.UMAP(n_components=2,  n_epochs=350 , min_dist=0.75, n_neighbors=30, metric='euclidean').fit( courses_mat  )
# emb_vals = embedding_crs.transform(courses_mat )
# crs_embedding_df['val_umap_1'] ,  crs_embedding_df['val_umap_2']    =   emb_vals[:, 0]  , emb_vals[:, 1]

# courses_res_lastdesc_df =  pd.merge(courses_subset_df.drop_duplicates(subset='cid' , keep='last') , course_desc_df, on='course_id', how='left'  ).fillna('???').drop(columns=['course_id'])

# umap.plot.points(embedding_crs, labels=courses_res_lastdesc_df.course_name , height=2200, width=2200, show_legend=False  )
```

```python
# data_subset_df = users_courses_df[users_courses_df['user_cid'] < 5000]
# data_fulldesc_df =  pd.merge(data_subset_df  , course_desc_df, on='course_id', how='left'  ).fillna('???').groupby(['user_cid'])['course_name'].apply(lambda x: ' | '.join(x)).reset_index()
# data_fulldesc_df
```

```python
# 'res_crsnames' : res_vals_df.groupby('res_cid')['course_name'].apply(lambda x: ' | '.join(x))          ,
```
