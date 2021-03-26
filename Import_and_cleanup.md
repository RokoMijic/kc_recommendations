---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python [conda env:python3]
    language: python
    name: conda-env-python3-py
---

# Import and cleaning up of KlasCement Data


This notebook currently has a code path with Dask and a different code path with pandas. The Dask code path is currently crashing with a memoryerror, suspected reason is that dask is not getting access to enough HDD space, or the dask outer merge has to happen in main memory. 

The Pandas code path requres the custom **efficient_outer_merge** function, which merges dataframes without the wasteful conversion to Float64 format which takes 8 bytes per entry as opposed to 1 byte with int8. Using Pandas, the datasets can be merged over about 5 minutes with 32GB or more of RAM. 

To use Pandas, set **compute_now = True** in Cell 14 which will compute the Dask datafames into Pandas dataframes.


##### Set up libraries etc

```python
!pip install --upgrade scikit-learn
!pip install --upgrade "dask[dataframe]"
!pip install --upgrade dask
!pip install --upgrade s3fs
!pip install --upgrade pandas
!pip install --upgrade pympler
!pip install --upgrade fsspec
!pip install googletrans

```

```python

```

##### Imports

```python
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from IPython.display import display, HTML

from functools import reduce
import copy

import re
import math
import random

import s3fs

import xml.etree.ElementTree as ET

from scipy import sparse

import googletrans
```

```python
from z_utilities import assert_nonan, assert_column_subset
```

```python
pd.set_option('display.max_rows', 300)
```

```python
# print versions
print('The pandas version is {}.'.format(pd.__version__))
print('The dask version is {}.'.format(dask.__version__))
print('The s3fs version is {}.'.format(s3fs.__version__))
```

### Guide to the user and resource data 

For users: ID, capacities ('capacities', e.g. student, teaching staff, member of organization), courses, levels of education, date of commencement of membership

For teaching resources: ID, type, subjects, levels of education, keywords, curated recommendations (curated recommendations), total number of days published, date of last review by moderator


### Guide to interaction data: 

<!-- #region -->
##### Standard Columns: 

**Column 1: unique identifier of the learning resource**

**Column 2: unique ID of the user**

**Column 3: date**


##### hits.csv

Visits on the detail page of a learning resource.


##### downloads.csv

Downloads of attachments to a learning resource. If a learning resource has multiple attachments, each will count individually as a download. Downloads by the same user in a short timeframe are likely to be consolidated into one download.


##### previews.csv

Previews of attachments to a learning resource. Previews are possible for some file types (eg pdf, doc, ppt), but not for all attachments. For each preview, we also keep track of whether it led to a download of the attachment. Every download from the previews.csv is also included in downloads.csv. There is no direct link, the download time will be "shortly after" the preview. Conversely, not every download is preceded by a preview, even if the file has that capability.


**Column 4: followed by a download (0 = no, 1 = yes)**


##### urls.csv

Clicked external links on a learning resource. If a learning resource has multiple external links, each will count individually as a click through. This is the equivalent of downloads for externally hosted content.



##### used.csv

"Use" of the learning resource. This means something different for different types of learning materials. For documents it is a download, for websites a click through, for audio or video "play" (with embeds this is undetectable, we use at least x seconds on the page as an indication that you probably played) ... It is a best effort to have a similar metric across the different types.




##### favorites.csv

Learning resources that have been favorite by a user.



##### scores.csv

Learning resources that have been scored by a user.


**Column 4: score given {score ∈ ℕ | 1 ≤ score ≤ 5}**


<!-- #endregion -->

##### Amazon S3 bucket identity: 

```python
bucket='045879944372-sagemaker-ml-dev'
```

# Fetch the user and resource data from S3 


### Course and level descriptors


###### Turn XML schemas into dataframes

```python
subj_translations = pd.read_csv( 's3://{}/{}'.format(bucket,  'subject_translations.csv' ) , names = ["nl", "en"]   )
subj_translations_dict = dict(zip(subj_translations.nl, subj_translations.en)) 
    
```

```python
def norm_langstring(langstring):
    
    langstring = langstring.lower().strip()

    # science and sciences mean the same thing
    langstring = re.sub("sciences", "science", langstring)

    # expand and standardize ICT
    if "information and communications technology" in langstring:  langstring = "information and communication technology"

    if "ict" == langstring:  langstring = "information and communication technology"

    # deal with some synonyms here
    if( 'dutch' in langstring and ('second' in langstring or 'foreign' in langstring or 'alpha' in langstring)) \
    or ('nt2' in langstring)   :
        langstring = 'dutch as a foreign language'

    # deal with more some synonyms here    
    if 'music' in langstring:  langstring = 'music'

    if langstring == "visual arts - art":  langstring = "visual arts"

    langstring = langstring.lower().strip()
    
    return langstring
```

```python
def xml_to_df(xml_s3loc, id_colname, meaningful_colname, trans_dict = subj_translations_dict):
    
    '''turns an XML schema of a particular type into a Pandas DF, including translating some parts to English
    '''
    
    columns = [id_colname, meaningful_colname]
    df_xml = pd.DataFrame(columns=columns)
    
    translator = googletrans.Translator()

    fs = s3fs.S3FileSystem()
    with fs.open(xml_s3loc) as f:
        root = ET.parse(f).getroot()

        for count, node in enumerate(root):
            termIdentifier = None
            langstring     = None
            nl_langstring  = None

            if("term" in node.tag): 
                for childnode in node:
                    if "termIdentifier" in childnode.tag:
                        termIdentifier = int(childnode.text) 
                    elif "caption" in childnode.tag:
                        for gchild in childnode:
                            
                            if gchild.text is not None:
                                # strip out text in brackets as it is often irrelevant for our purposes
                                stripped_text = re.sub("[\(\[].*?[\)\]]", "", gchild.text).lower().strip()

                                if gchild.attrib["language"] == "en":
                                    langstring = stripped_text
                                elif gchild.attrib["language"] == "nl":
                                    nl_langstring = stripped_text

                        # use google translate to translate subjects into English for easy reading!
                        if langstring is None and nl_langstring is not None :
                            
                            if(nl_langstring in trans_dict.keys())     :    langstring = trans_dict[nl_langstring]
                            else                                       :    langstring = nl_langstring   

                    else: pass
                
                langstring = norm_langstring(langstring)

                df_xml = df_xml.append( pd.Series([termIdentifier, langstring], index=columns), ignore_index=True)
                
            else: pass

    df_xml = df_xml.sort_values(id_colname)
    df_xml = df_xml.reset_index(drop=True)
    
    return df_xml
```

```python
def make_replacement_frame(df_in, colname_rep, id_col):
    
    ''' Makes a dataframe that shows which id's represent duplicates of each other based on the colname_rep value
    '''
    
    unique_df = df_in.drop_duplicates(keep='first', subset=[colname_rep] ).reset_index(drop=True)
    repl_df = pd.merge(df_in, unique_df, on=colname_rep, suffixes=('', '_rep') ).drop(columns=[colname_rep])
    
    return repl_df
```

------------------------------------------------------------------


###### Now get the english descriptions of the various XML files

```python
course_d_filename = 's3://{}/{}'.format(bucket, 'courses.xml')
course_desc_incldups_df = xml_to_df(course_d_filename, 'course_id', 'course_name' ).sort_values(by=['course_id'])
print(course_desc_incldups_df.shape)
```

```python
course_id_rep_df = make_replacement_frame(course_desc_incldups_df, colname_rep='course_name', id_col='course_id')
display(course_id_rep_df.head(n=1))
print(course_id_rep_df.shape)
```

```python
course_desc_df = course_desc_incldups_df.drop_duplicates(keep='first', subset=[ 'course_name' ] ).reset_index(drop=True)
display(course_desc_df.head(n=2))
print(course_desc_df.shape)
```

```python
edu_d_filename = 's3://{}/{}'.format(bucket, 'educationtypes.xml')
edu_desc_df = xml_to_df(edu_d_filename, 'edutype_id', 'education_type_name' )
display(edu_desc_df.head(n=3))
print(edu_desc_df.shape)
```

##### utility to explode columns

```python
def expand_col_df(df_in, col_toex, col_keep, sep, expand_to_int=True):
    
    df_exp = pd.DataFrame(df_in[col_toex].str.split(sep).tolist(), index=df_in[col_keep]).stack()
    df_exp = df_exp.reset_index()[[0, col_keep]]  
    df_exp.columns = [col_toex, col_keep] 
    df_exp = df_exp[[col_keep, col_toex]]
    
    if expand_to_int: df_exp[col_toex] = pd.to_numeric(df_exp[col_toex])
    
    return df_exp
```




------------------------------------------------------


### Users

```python
users_filename = 's3://{}/{}'.format(bucket,  'users.csv' )
users_df = pd.read_csv( users_filename , low_memory=False )
users_df = users_df.rename(columns={'courses': 'course_id', 
                                    'education_types': 'edutype_id',
                                   })

# Drop users that either have no object type or no course_id
users_df = users_df.dropna(subset=['course_id', 'edutype_id'])
```

```python
display(users_df.head(n=2))
print(users_df.shape)

```

```python
user_ids = sorted(list(set(users_df['user_id'].values.tolist())))
user_cids = list(range(len(user_ids))) 
random.Random(0).shuffle(user_cids)
user_cids_df = pd.DataFrame( {'user_id': user_ids  , 'user_cid': user_cids } ) 
```

```python
display(user_cids_df.head(n=2))
print(user_cids_df.shape)
```

```python
users_courses_df = expand_col_df(users_df, col_toex='course_id', col_keep='user_id', sep='|' ) \
                                .merge(course_id_rep_df,  on='course_id', how="inner" ) \
                                .drop(columns=['course_id'])  \
                                .rename(columns={'course_id_rep': 'course_id' } ) \
                                .drop_duplicates()
```

```python
display(users_courses_df.sort_values(by=['user_id']).head(n=2))
print(users_courses_df.shape[0])
```

```python
assert_column_subset(users_courses_df, 'course_id', course_desc_df, 'course_id' )
```

```python
users_edutypes_df = expand_col_df(users_df, col_toex="edutype_id", col_keep="user_id", sep='|' )\
                                    .merge(edu_desc_df[['edutype_id']], on='edutype_id', how='inner')
```

```python
display(users_edutypes_df.head(n=2))
print(users_edutypes_df.shape[0])
```

```python
assert_column_subset(users_edutypes_df, 'edutype_id', edu_desc_df, 'edutype_id' )
```

---------------------------------------------------------------------------------------


#### Resources 

```python
resource_colnames = ['res_id', 'type', 'course_id', 'edutype_id','keywords' , 'recommendations' ,
                     'num_days_old', 'date_last_review' ]
resource_filename = 's3://{}/{}'.format(bucket,  'objects.csv' )

resources_df = pd.read_csv( resource_filename ,  low_memory=False  ) 
resources_df = resources_df.rename(columns={'courses': 'course_id', 
                                    'education_types': 'edutype_id',
                                    'object_id': 'res_id',
                                   })

# Drop resources that either have no object type or no course_id
resources_df = resources_df.dropna(subset=['object_type_id', 'course_id'])  
```

```python
display(resources_df.head(n=2))
print(resources_df.shape[0])
```

```python
res_ids = sorted(list(set(resources_df['res_id'].values.tolist())))
res_cids = list(range(len(res_ids))) 
random.Random(1).shuffle(res_cids)
res_cids_df = pd.DataFrame( {'res_id': res_ids  , 'res_cid': res_cids  } )  
```

```python
display(res_cids_df.head(n=2))
print(res_cids_df.shape[0])
```

```python
resources_status_df = resources_df[['res_id', 'status' ]].rename(columns={'status':'status_approved'})
resources_status_df['status_approved'] = resources_status_df['status_approved'].apply(lambda s : True if s=='approved' else False)
```

```python
display(resources_status_df.head(n=2))
print(resources_status_df.shape[0])
```

```python
resources_title_df = resources_df[['res_id', 'title' ]] 
```

```python
display(resources_title_df.head(n=2))
print(resources_title_df.shape[0])
```

```python
resources_courses_df = expand_col_df(resources_df, col_toex="course_id", col_keep="res_id", sep='|' )\
                                .merge(course_id_rep_df,  on='course_id', how="inner" ) \
                                .drop(columns=['course_id'])  \
                                .rename(columns={'course_id_rep': 'course_id' } ) \
                                .drop_duplicates()
```

```python
display(resources_courses_df.head(n=2))
print(resources_courses_df.shape[0])
```

```python
assert_column_subset(resources_courses_df, 'course_id', course_desc_df, 'course_id' )
```

```python
resources_edutypes_df = expand_col_df(resources_df, col_toex="edutype_id", col_keep="res_id", sep='|' )\
                                    .merge(edu_desc_df[['edutype_id']], on='edutype_id', how='inner')
```

```python
display(resources_edutypes_df.head(n=2))
print(resources_edutypes_df.shape[0])
```

```python
assert_column_subset(resources_edutypes_df, 'edutype_id', edu_desc_df, 'edutype_id' )
```

```python
resources_keywords_raw_df = expand_col_df(resources_df, col_toex="keywords", col_keep="res_id", sep='|', expand_to_int=False ).rename(columns={'keywords': 'keyword'}).astype({'keyword': 'str'})
```

```python
keywords_desc_df = pd.DataFrame({'count' : resources_keywords_raw_df[['keyword']]['keyword'].value_counts()} )\
                                                                                           .reset_index(drop=False).rename(columns={'index': 'keyword'})\
                                                                                           .sort_values(by='count', ascending=False )\
                                                                                           .reset_index(drop=False).rename(columns={'index': 'keyword_id'})
```

```python
display(keywords_desc_df.head(n=2))
print(keywords_desc_df.shape[0])
```

```python
resources_keywords_df = resources_keywords_raw_df.merge(keywords_desc_df, on='keyword', how='left' ).drop(columns=['keyword']).rename(columns={'count': 'keyword_count'})
```

```python
display(resources_keywords_df.head(n=2))
print(resources_keywords_df.shape[0])
```

### Save the User and Resource data 

```python
save_ru_data = True 
```

##### utility to add contiguous ids

```python
def add_cids_to(df, cids_df, id_colname):
    cid_colname = [c for c in list(cids_df.columns.values) if ('_cid' in c)][0]
    return cids_df.merge(df , on=id_colname, how='inner').sort_values(cid_colname).reset_index(drop=True).drop(columns=[id_colname])
```

```python
if save_ru_data: 
    
    ru_dfs_dict = {  
                      'course_desc'         :  course_desc_df , 
                      'educ_type_desc'      :  edu_desc_df , 
                      'user_cids'           :  user_cids_df,
                      'users_courses'       :  users_courses_df , 
                      'users_edutypes'      :  users_edutypes_df ,
                      'res_cids'            :  res_cids_df, 
                      'resources_courses'   :  resources_courses_df , 
                      'resources_edutypes'  :  resources_edutypes_df ,
                      'keywords_desc'       :  keywords_desc_df ,   
                      'resources_keywords'  :  resources_keywords_df ,   
                      'resources_title'     :  resources_title_df ,
                      'resources_status'    :  resources_status_df
                  }
    
    for df_name, df in ru_dfs_dict.items():
        
        cids_df = None
    
        if 'cids' not in df_name and 'desc' not in df_name :
            if    'user'      in df_name  :   
                cids_df    = user_cids_df
                id_colname = 'user_id'
            elif  'resources' in df_name  :   
                cids_df = res_cids_df
                id_colname = 'res_id'
                
            if cids_df is not None: df = add_cids_to(df=df, cids_df=cids_df, id_colname=id_colname)
            
        print('\nSaving:  ' + df_name + '.csv' )    
        display(df.head(n=2))
            
        data_location = 's3://{}/{}'.format(bucket, df_name)
        df.to_csv(data_location + '.csv', encoding='utf-8', index=False) 
        
else:
    print("Nothing saved since save_ru_data = False ")
```

----------------------------------------------------------------------------------------


# Fetch interaction data from S3






Warning: Joining the interaction data requires a machine with at least 32GB of RAM 

```python
Fetch_interaction_data = True
```

```python
assert Fetch_interaction_data
```

```python
data_standard_cols = ['res_id', 'user_id', 'date'] 

data_extra_cols =   {'favourites'  :  []                 ,  
                     'scores'      :  ['score']          , 
                     'urls'        :  []                 ,
                     'previews'    :  ['downloaded']     , 
                     'downloads'   :  []                 ,  
                     'used'        :  []                 ,  
                     'hits'        :  []                 ,
                    }

readable_colnames = {'favourites'  :  'favourited'       , 
                     'scores'      :  'gave_score'       , 
                     'urls'        :  'clicked_through'  ,  
                     'previews'    :  'previewed'        , 
                     'downloads'   :  'downloaded'       ,  
                     'used'        :  'used'             , 
                     'hits'        :  'visited_detail_pg',  
                    }

data_names = list(data_extra_cols.keys())
data_cols = {  k : data_standard_cols + v  for (k,v) in data_extra_cols.items()  }
```

###### create dask dataframes containing different types of interaction information

```python
do_all_datas = True

data_names_to_do = data_names if do_all_datas else data_names[:3]
```

```python
dds_list = []

for data_name in data_names_to_do:

    data_key = data_name + '.csv'

    colnames = data_cols[data_name]

    data_location = 's3://{}/{}'.format(bucket, data_key)
    
    this_dd = dd.read_csv(data_location, names=colnames ) 
    
    if(data_name != 'scores' ):      
        this_dd[ 'event|' + readable_colnames[data_name]  ] = 1
        this_dd[ 'event|' + readable_colnames[data_name]  ] = this_dd[ 'event|' + readable_colnames[data_name]  ].astype( 'int8' )
    
    if(data_name == 'scores' ): 
        this_dd[ data_extra_cols['scores'] ] = this_dd[ data_extra_cols['scores'] ].astype( 'int8' )
         
    if(data_name == 'previews' ):    del this_dd['downloaded']
        
    this_dd['date'] = this_dd['date'].apply( lambda s: s[:10], meta=('date', 'str') )      
    
    this_dd = this_dd.rename(columns={'date': 'date|' + readable_colnames[data_name] })
    
    this_dd = this_dd.loc[this_dd['user_id'] > 0]
    this_dd = this_dd.loc[this_dd['res_id'] > 0]
    
    this_dd['user_id']  = this_dd['user_id'].astype( 'int32' )
    this_dd['res_id'] = this_dd['res_id'].astype( 'int32' )

    this_dd = this_dd.reset_index( drop=True)
    
    dds_list.append(this_dd)
```

###### create lists of frames containing either no dates, or no events 

```python
def remove_cols_starting(d, startword):   return d.drop(d.columns[d.columns.str.startswith( startword )], axis=1)
```

```python
dds_nodate_list = [ remove_cols_starting(d, 'date|' ) for d in dds_list ]
```

```python
dds_noevent_list = [ remove_cols_starting(d, 'event|' ) for d in dds_list ]
```

##### function to do an outer merge without using loads of memory 

```python
def efficient_outer_merge(left, right, on, fillna = 0):

    inner_merge =  pd.merge(left, right,  on=on  , how='inner')

    left_nonkey_cols =  list(set(left.columns.tolist()) - set(on))  
    right_nonkey_cols =  list(set(right.columns.tolist()) - set(on))

    right_nonkey_dict = { col:dtype for col,dtype in right.dtypes.to_dict().items() if col in right_nonkey_cols  }
    left_nonkey_dict = { col:dtype for col,dtype in left.dtypes.to_dict().items() if col in left_nonkey_cols  }

    left_extension = pd.DataFrame(  {  n : pd.Series([fillna]*len(left.index), dtype=d)  for n, d in  right_nonkey_dict.items()  } ,  index = left.index  )
    left_extended = pd.concat([left,left_extension], axis=1)

    right_extension = pd.DataFrame(  {  n : pd.Series([fillna]*len(right.index), dtype=d)  for n, d in  left_nonkey_dict.items()  } ,  index = right.index  )
    right_extended = pd.concat([right_extension, right ],  axis=1)

    merged = pd.concat([inner_merge, left_extended, right_extended ],  axis=0)

    merged.drop_duplicates(subset = on, keep = 'first', inplace = True, ignore_index = True )

    merged = merged.reset_index( drop=True)

    return merged
```

```python

```

```python
def merge_and_fix_list(frames_list ):
    
    def merge_frames_pair(left, right, keycols = ['res_id', 'user_id']): 

        print(f'merging with {str(type(left))}')
        
        if('dask' in str(type(left)) )   :   merged_df = dd.merge(left,right,on=keycols, how='outer')
        else                             :   merged_df = efficient_outer_merge(left, right, on=keycols)        
        
        return merged_df
        
    merge_accumulator_frame = frames_list[0]
    
    for frame in frames_list[1:]:
        merge_accumulator_frame = merge_frames_pair(merge_accumulator_frame, frame)
        del frame
        merge_accumulator_frame = merge_accumulator_frame.fillna(0) 
        smallint_col_names = ( lambda d : d.columns[ d.columns.str.startswith( "event|" ) | d.columns.str.startswith( "score" ) ].values.tolist() )  (merge_accumulator_frame)   
        merge_accumulator_frame[smallint_col_names] = merge_accumulator_frame[smallint_col_names].astype("int8")

    merge_accumulator_frame = merge_accumulator_frame.reset_index(drop=True)
    
    return merge_accumulator_frame
```

```python
frame_list_to_process = dds_nodate_list
```

```python
compute_now = True
```

```python
if compute_now:
    if('dask' in str(type(frame_list_to_process[0])) ): 
        with ProgressBar():
            frame_list_to_process = [d.compute().reset_index(drop=True) for d in  frame_list_to_process]
```

---------------------------------------------------------------
The following merge step will fail if you have less than about 32GB of RAM

```python
merged_frame = merge_and_fix_list(frame_list_to_process)
```

```python
if('dask' not in str(type(merged_frame)) ):  
    merged_frame = merged_frame.sort_values(by=['res_id','user_id'] , axis=0, ascending=True, inplace=False, kind='quicksort', ignore_index=True)
```

```python
merged_frame.head(n=5)
```

```python
merged_frame_filtered = merged_frame.merge(user_cids_df, on='user_id', how='inner').merge(res_cids_df, on='res_id', how='inner')
merged_frame_filtered = merged_frame_filtered[ ['res_cid', 'user_cid'] + [ col for col in merged_frame_filtered.columns if col not in ['user_cid', 'res_cid'] ] ]
merged_frame_filtered = merged_frame_filtered.drop(columns=['res_id', 'user_id'])

merged_frame_filtered
```

```python
merged_frame_filtered.shape[0]
```

```python
if('dask' not in str(type(merged_frame_filtered)) ):  
    merged_frame_filtered = merged_frame_filtered.sort_values(by=['res_cid','user_cid'] , axis=0, ascending=True, inplace=False, kind='quicksort', ignore_index=True)
```

```python
merged_frame_filtered
```

##### Save the merged interaction data

```python
save_data = True
```

```python
frame_to_save = merged_frame_filtered
```

```python
if save_data:
    
    filename = 'klascement_no_dates_sorted'
    data_location = 's3://{}/{}'.format(bucket, filename)

    if('dask' in str(type(frame_to_save)) ): 
        with ProgressBar(): 
            frame_to_save.to_csv(data_location, encoding='utf-8', index=False)
    else:
        frame_to_save.to_csv(data_location + '.csv', encoding='utf-8', index=False) 

else:
    if('dask' in str(type(merged_frame)) ): 
        with ProgressBar(): 
            df_to_display = frame_to_save.compute()
    else: 
        df_to_display = frame_to_save
        
    display(df_to_display.head(n=5))    
    display(df_to_display.sample(n=5))
    
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

Stop execution here

```python
assert False 
```

###### utility functions for checking memory use

```python
def memuse_per_row(df):
        
    nrows = df.shape[0]
    memuse = sum(df.memory_usage() )
    bytes_per_row = memuse/nrows
    return bytes_per_row
```

```python
def memuse(df):
            
    memuse = sum(df.memory_usage() )

    return memuse
```

```python
def summarize(df):
    
    print( f'memuse = {np.round(memuse(df), -5):,}' )   
    print( f'rows = {np.round(  df.shape[0], -5) :,}' )   
    print( f'memuse per row = {np.round(memuse_per_row(df), -0):,}' )  
    display(df.head(n=3))
    print(df.dtypes)
```
