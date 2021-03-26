import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

import hashlib
import codecs





def hash_df(df, index=True, small_frame_limit=1000000, num_chars=8):
    
    '''
    Quickly produces a deterministic hash of a dataframe in a nice readable hash format
    Larger DataFrames get sampled for efficiency, but this should usually be OK.
    '''
    
    if df.shape[0] <= small_frame_limit:
        hexdigest = hashlib.sha1(pd.util.hash_pandas_object(df , index=index).values).hexdigest() 
    else:
        combined_hash = hashlib.sha1()
        
        df_sample = df.sample(n=min(1000000, df.shape[0]//3), replace=True, random_state=0)
        digest_sample = hashlib.sha1(pd.util.hash_pandas_object(df_sample, index=index).values).digest() 
        combined_hash.update(digest_sample)  
        sums    = [df[c].sum()  for c in df.columns.tolist() if np.issubdtype(df[c].dtype, np.number)]
        digest_sums  =  hashlib.sha1(   np.array(sums )    ).digest() 
        combined_hash.update(digest_sums)        
        
        hexdigest = combined_hash.hexdigest()
        
    b64digest = codecs.encode(codecs.decode(hexdigest, 'hex'), 'base64').decode().replace("\n", "").replace("=", "").replace("/", "$").replace("+", "&")
    
    return b64digest[:num_chars]




 







def stack_dfs_common_column(df1, df2, common_name:str, diff_name_1:str, diff_name_2:str,  increm_ids:bool=True):
    
    ''' 
    stack two numeric dataframes with a common column together, incrementing numbers in a differing column so that distinct 
    entities are kept distinct. Other columns are allowed, as long as they all have the same names.  
    '''
    
    assert common_name in df1.columns.tolist()
    assert common_name in df2.columns.tolist()
    assert diff_name_1 in df1.columns.tolist()
    assert diff_name_2 in df2.columns.tolist()
    assert np.issubdtype(df1[diff_name_1].dtype, np.integer)
    assert np.issubdtype(df2[diff_name_2].dtype, np.integer) 
    assert set(df1.columns.tolist()) - {diff_name_1} == set(df2.columns.tolist()) - {diff_name_2}
        
    offset_val = max(df1[diff_name_1].values.tolist()) + 1 if increm_ids else 0
    
    diff_name_c = diff_name_1 + '_|_' + diff_name_2 + ('_+_' + str(offset_val) if increm_ids else '')
    
    df2_inc = df2.copy()
    df2_inc[diff_name_2] = df2_inc[diff_name_2] + offset_val
    
    df_c = df1.rename(columns={ diff_name_1 : diff_name_c }).append(df2_inc.rename(columns={ diff_name_2 : diff_name_c }), ignore_index=True)
    
    del df2_inc

    return df_c




def df_to_matix(df, col1, col2, value=1, numeric_type='int8', matrix_type='coo', contig_col=None): 
    
    '''
    Convert a dataframe into a coincidence matrix, where M_ij = 1 iff i and j co-occur in col1 and col2
    '''
    
    if (col1 is None) and (col2 is None): 
        assert len(data_df.columns.tolist()) == 2
        col1, col2 = data_df.columns.tolist()[0] , data_df.columns.tolist()[1] 
        
    for coli in [col1, col2]: 
        if (not np.issubdtype(df[coli].dtype, np.integer)): raise ValueError(f'{df[coli].dtype} should equal {np.integer}')
            
    assert matrix_type in ['dense', 'coo', 'csc', 'csr']
    
    df_to_mz=df
    
    if contig_col is not None:
        # relabel one of the columns to be contiguous integers starting from 0  
        assert contig_col in [col1, col2]
            
        ix_df = df[[contig_col]].drop_duplicates().sort_values(by=contig_col).reset_index(drop=True).reset_index(drop=False)[['index', contig_col]]
        
        df_to_mz = df.merge(ix_df, on=contig_col, how='left')[[(col1 if contig_col==col2 else col2), 'index']].rename(columns={'index':contig_col})


    coo_df_matrix = coo_matrix( ( ( np.array([value]*df_to_mz.shape[0]) ).astype(numeric_type), (df_to_mz[col1], df_to_mz[col2]) )  )
    
    if matrix_type=='coo'      :  return coo_df_matrix
    elif matrix_type=='csr'    :  return coo_df_matrix.tocsr()
    elif matrix_type=='csc'    :  return coo_df_matrix.tocsr()
    elif matrix_type=='dense'  :  return coo_df_matrix.todense().A


def assert_nonan(df):
    if df.isnull().values.any(): 
        nan_df = ( lambda d : d[d.isnull().any(axis=1)]   ) (df) 
        display(  nan_df.head(n=5)  )
        nan_fraction = nan_df.shape[0]  / df.shape[0] 
        raise ValueError(f'Dataframe has {nan_df.shape[0] } nans, which is {nan_df.shape[0]  / df.shape[0] :.2f}% of the rows')

        

def valset(df, col): return set(df[col].unique().tolist() )

def assert_column_subset(df_subset, col_subset, df_superset, col_superset):
    '''
    assert that the set of values in col_subset of df_subset are a subset of the values of col2_superset
    '''
    if not valset(df_subset, col_subset) <= valset(df_superset, col_superset) : 
        val_diff = valset(df_subset, col_subset)-valset(df_superset, col_superset)
        raise ValueError(f'Sets differ between {col_subset} and {col_superset}, difference is {len(val_diff)} elements, listed here:\n {val_diff} ')    

    

# +
# ix_df['index'] += ix_df[contig_col].iloc[0]
# if not ix_df['index'].iloc[0] == ix_df[contig_col].iloc[0] : raise ValueError(f"{ix_df['index'].iloc[0]} must be equal to {x_df[contig_col].iloc[0]}")
