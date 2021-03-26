# Defining Schools Python version 

```python
!pip install --upgrade pandas
!pip install --upgrade scikit-learn
!pip install --upgrade pympler
!pip install --upgrade s3fs
!pip install --upgrade annoy
!pip install --upgrade uncertainties
!pip install --upgrade matplotlib
```

To make jupytext work, the following may be neccessary: 
https://stackoverflow.com/questions/59949130/install-jupytext-plugin-on-aws-sagemaker

```python
import pandas as pd
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

from pympler.asizeof import asizeof

from scipy.sparse import  coo_matrix

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import random_projection
from sklearn.neighbors import NearestNeighbors
import sklearn

from math import log2
import sys

from annoy import AnnoyIndex

from uncertainties import ufloat
```

### Create Dataset

```python
num_students = 500000
num_school_units = 25000
# num_students = 50000
# num_school_units = 1000

# num_students = 10000
# num_school_units = 200


num_years = 10
num_insp_units = num_school_units // num_years
num_rows = num_years*num_students
sch_per_ins = num_school_units // num_insp_units
stu_per_sch = num_students     // num_school_units
stu_per_ins = num_students     // num_insp_units



# 1 in fiu of inspection units will be naturally unpredictable
probability_of_random_inspection_unit_change = 0.00
fiu = int(np.round(1/(0.00001 + probability_of_random_inspection_unit_change)))

# probability_of_switching_outside_school_group   - 0.75 is roughly the maximum where perfect prediction is still possible for the exact algorithms
pog =  0.40

distance_move_from_school_group = int(num_school_units*0.25)
dmf = distance_move_from_school_group

```

```python

```

```python
dataset = pd.DataFrame(  {    "studentID" : pd.Series(   np.repeat(np.array(range(0,num_students)), num_years).tolist()   , dtype="uint32")   ,
                              "year"      : pd.Series(   list(range(0,num_years))*num_students                             , dtype="uint8" )   ,
                         }   
                      )
```

```python
dataset
```

```python
prob_dist =  [pog/(2*dmf)]*dmf + [ (1-pog) ] + [pog/(2*dmf)]*dmf
    
dataset["move"] =  pd.Series(       np.random.choice( a = list(range(-dmf,dmf+1)) ,  size=dataset.shape[0]  , p=prob_dist )      , dtype="int16" ) 
```

```python
dataset["school_unit_u"] = ( ( (dataset["studentID"] // stu_per_ins ) * sch_per_ins + 
                                dataset["year"] + 
                               (dataset["year"]!=0)*dataset["move"]                      ) % num_school_units  
                           
                           ).astype("uint32")

```

```python
# dataset["school_unit"] = dataset["school_unit_u"]
dataset["school_unit"] = ( dataset["school_unit_u"]*17 + 3  ) % num_school_units
```

```python
if num_students <= 10000:
    plt.figure(figsize=(17,14))
    plt.scatter(dataset["studentID"], dataset["school_unit"], s=0.1,  alpha=0.25)
    plt.xlabel("studentID")
    plt.ylabel("school_unit")
    plt.show()
```

```python
dataset["inspection_unit_u"] = (   (    dataset["school_unit_u"] // (sch_per_ins) + 
                                 
                                        ( (dataset["school_unit_u"]+1)  %  fiu == 0)*(    (dataset["school_unit_u"]*77 ) % num_school_units   )    ) % num_insp_units 
                               
                               ).astype("uint32")
```

```python
# dataset["inspection_unit"] = dataset["inspection_unit_u"]
dataset["inspection_unit"] = ( dataset["inspection_unit_u"]*19 + 7   ) % num_insp_units
```

```python
# plt.figure(figsize=(15,12))
# plt.scatter(dataset["studentID"], dataset["inspection_unit_u"], s=0.1,  alpha=0.15)
# plt.xlabel("studentID")
# plt.ylabel("inspection_unit_u")
# plt.show()
```

```python
# plt.figure(figsize=(15,12))
# plt.scatter(dataset["school_unit"], dataset["inspection_unit"], s=0.1,  alpha=0.15)
# plt.xlabel("school_unit")
# plt.ylabel("inspection_unit")
# plt.show()
```

```python
# data_units_u = dataset[ ["school_unit_u", "inspection_unit_u"] ].drop_duplicates( ignore_index = True ).sort_values(by = "school_unit_u",  ignore_index=True)
```

```python
num_school_units_practice = dataset[["school_unit"]].drop_duplicates(ignore_index = True).shape[0]
if num_school_units_practice  != num_school_units:    raise ValueError(num_school_units_practice)
```

```python
num_inspection_units_practice = dataset[["inspection_unit"]].drop_duplicates(ignore_index = True).sort_values(by = "inspection_unit",  ignore_index=True).shape[0]
if num_inspection_units_practice  != num_insp_units:    raise ValueError(num_inspection_units_practice)
```

```python
dataset = dataset[["studentID", "year",  "school_unit", "inspection_unit"]]
```

```python
data_units = dataset[["school_unit", "inspection_unit"]].drop_duplicates(ignore_index = True).sort_values(by = "school_unit",  ignore_index=True)
```

```python
data_units.dtypes
```

```python
# plt.figure(figsize=(4,3))
# plt.scatter(data_units["school_unit"], data_units["inspection_unit"], s=1,  alpha=0.5)
# plt.xlabel("school_unit")
# plt.ylabel("inspection_unit")
# plt.show()
```

```python
display(dataset)
```

```python
print(asizeof(dataset))
```

```python
print(dataset.dtypes)
```

```python
# assert False 
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

 


## Try to predict which schools are in the same inspection unit with SKLearn


 

```python
def to_matrix(df, dense, rows , cols, dtype="int8" ): 
    
    smatrix = coo_matrix(  (  np.full( [df.shape[0]]  , 1, dtype=dtype )  ,   (df[rows] , df[cols])   ) ,  shape=(df[rows].nunique(), df[cols].nunique())  )  
    
    return smatrix.todense() if dense else smatrix
```

```python

```

```python

```

```python
def get_neighbors_lists(selected_school_units, df, n_neighbors, nn_method, nn_args):
    """ Interface for getting nearest neighbors
    """
    if nn_method ==  "sklearn_dense"      :   return get_neighbors_lists_sklearn(selected_school_units, df, n_neighbors, dense=True)
    if nn_method ==  "sklearn_sparse"     :   return get_neighbors_lists_sklearn(selected_school_units, df, n_neighbors, dense=False)
    if nn_method ==  "sklearn_rproj"      :   return get_neighbors_lists_sklearn_rproj(selected_school_units, df, n_neighbors, eps=nn_args['eps'], metric=nn_args['metric'], nn_algorithm=nn_args['nn_algorithm'])
    elif nn_method ==  "annoy"            :   return get_neighbors_lists_annoy(selected_school_units, df, n_neighbors)
    
    else: raise ValueError(nn_method)
```

```python

```

```python
def get_neighbors_lists_sklearn(selected_school_units, df, n_neighbors, dense):
    """ gets a list of lists of nearest neighbors of specified entities base on a dense matrix using sklearn NearestNeighbors
    """
    
    matrix = to_matrix(df, dense=dense, rows="school_unit" , cols="studentID" )
    
    print("Beginning fit")
    model_knn = NearestNeighbors(metric= "jaccard" if dense else "cosine", algorithm= "ball_tree" if dense else "brute", n_neighbors=1, n_jobs=-1)
    model_knn.fit(matrix)
    print("Fitted model")
    
    print("Predicting")
    if dense:      
        points           =   matrix[selected_school_units]  
        neighbors_data   =   model_knn.kneighbors(   points ,  n_neighbors=min(n_neighbors, matrix.shape[0])  ,  return_distance=False  )
        neighbors_lists  =   [sorted(l) for l in neighbors_data.tolist()]
        
    else: 
        neighbors_lists  =  []
        for unit in selected_school_units:
            if unit % 10 == 0 : print(".", end="")
            neighbors_this_unit = model_knn.kneighbors( matrix.getrow(unit) , n_neighbors=min(n_neighbors, matrix.shape[0]) ,  return_distance=False  ).tolist()[0]
            neighbors_lists.append( sorted( neighbors_this_unit )   )
            
    print("\nDone predicting")
    
    return neighbors_lists
```

```python
def get_actual_su_lists(selected_school_units, df):
    """ gets the actual school units with the same inspection unit 
    """
    
    data_units = df[["school_unit", "inspection_unit"]].drop_duplicates(ignore_index = True).sort_values(by = "school_unit",  ignore_index=True)
    
    actual_su_cius_lists = []
    
    for unit in selected_school_units:
    
        corresponding_iu                     = data_units[  data_units["school_unit"]==unit                  ]["inspection_unit"].values.tolist()[0]
        actual_school_units_corresponding_iu = data_units[  data_units["inspection_unit"]==corresponding_iu  ]["school_unit"].values.tolist()
        actual_su_cius_lists.append(  actual_school_units_corresponding_iu  )
        
    return actual_su_cius_lists

```

```python
def calc_frac_correct(pred_lists, actual_lists ):
    
    fracs_correct = []
        
    for predicted_l, actual_l in list(zip(pred_lists, actual_lists)):
        
        intersection = list( set(predicted_l)  &  set(actual_l)  )
        numintersect = len(intersection)
        frac_correct = numintersect/len(actual_l)
        fracs_correct.append(frac_correct)
        
    return fracs_correct
```

```python
def get_frac_correct(selected_school_units, df, n_neighbors, nn_method, nn_args ):   
    
    pred_lists = get_neighbors_lists(selected_school_units, df, n_neighbors=n_neighbors, nn_method=nn_method, nn_args=nn_args)
    
    actual_lists   =   get_actual_su_lists(selected_school_units, df )
    
    fracs_correct = calc_frac_correct( pred_lists=pred_lists, actual_lists=actual_lists )
    
    return ( lambda L : ufloat(np.mean( L ) , np.std( L )/len( L )**0.5  )  )  ( fracs_correct )  
```

Test NN calcs with sklearn dense

```python
if num_students * num_school_units <= 10**8:
    %prun -l5 fc = get_frac_correct( selected_school_units = choice(range(0, num_school_units), size=num_school_units//20,  replace=False), df= dataset,  n_neighbors=10,  nn_method="sklearn_dense", nn_args={} )
    print( fc  )
```

Test NN calcs with sklearn sparse

```python
%prun -l5 fc = get_frac_correct( selected_school_units = choice(range(0, num_school_units), size=20,  replace=False), df= dataset,  n_neighbors=10,  nn_method="sklearn_sparse", nn_args={} )
print( fc  )
```

```python
"time per point = %.4f" % (  4.068/20 )

```

```python

```

#### Try to predict with Annoy


Problem: the /tmp/ folder on AWS maxxes out at 40GB, which is still too small, so annoy will not work even as a mmaped solution.

Also it's slow to make the index, which is not ideal for reproducibility. 

```python
def get_neighbors_lists_annoy(selected_school_units, df, n_neighbors, n_trees=None, filename="/tmp/annoyindex.bin", verbose=True):
    """ gets a list of lists of nearest neighbors of specified entities based on a dense matrix using annoy
    """
    
    sp_matrix = to_matrix(df, dense=False, rows="school_unit" , cols="studentID" )
    
    # A csr matrix is used because row slicing is fast for csr
    csr_matrix = sp_matrix.tocsr()
    
    num_items = sp_matrix.shape[0]
    num_dims = sp_matrix.shape[1]
    dmsize = num_items*num_dims
    fit_in_memory = dmsize <= 25*2**30
    
    if n_trees==None: n_trees = max(10,  num_items//200 )
    
    if verbose: print('Dense matrix size = {:,} Mbytes'.format( int( float ('%.3g' % (dmsize/2**20) ) ) )   )
    
    iannoyindex = AnnoyIndex(num_dims, 'angular')  
    if not fit_in_memory: iannoyindex.on_disk_build(  filename  )
    
    for sch_unit in list(range(num_items)):
        
        dense_row_this_unit = np.squeeze(  np.asarray( csr_matrix.getrow(sch_unit).todense() )  )      
        iannoyindex.add_item(  sch_unit  ,  dense_row_this_unit  )
        if sch_unit % 100 == 0: print(".", end="")

    iannoyindex.build(n_trees=n_trees)

    neighbors_lists = []
    
    for sch_unit in selected_school_units:
        neighbors_lists.append(iannoyindex.get_nns_by_item(sch_unit, n_neighbors , search_k = n_neighbors*n_trees*2 ))
        if sch_unit % 100 == 0: print(".", end="")
            
    
    return neighbors_lists
```

```python
if num_students * num_school_units <= 5*10**8:
    %prun -l5 fc = get_frac_correct( selected_school_units = choice(range(0, num_school_units), size=num_school_units,  replace=False), df= dataset,  n_neighbors=10,  nn_method="annoy", nn_args={} )
    print( fc  )
```

```python

```

```python
# assert False
```

```python

```

```python

```

#### Try to predict with Random Projections

```python
def reduce_matrix_dimension_rproj(matrix_hidim, eps):
    
    transformer = random_projection.SparseRandomProjection(n_components='auto', density='auto', eps=eps, dense_output=True, random_state=314)
    matrix_lowdim = (transformer.fit_transform(matrix_hidim)).astype('float16')
    
    return matrix_lowdim
    
```

```python
def get_neighbors_lists_sklearn_rproj(selected_school_units, df, n_neighbors, eps, metric, nn_algorithm, verbose=True):
    """ gets a list of lists of nearest neighbors of specified entities base on a dense matrix using sklearn NearestNeighbors
    """
    
    matrix_hidim = to_matrix(df, dense=False, rows="school_unit" , cols="studentID" )
    if verbose: print(matrix_hidim.shape)
        
    matrix_lowdim = reduce_matrix_dimension_rproj(matrix_hidim, eps=eps)
    if verbose: print(matrix_lowdim.shape)
    
    if verbose: print("Beginning fit")
    model_knn = NearestNeighbors(metric=metric , algorithm=nn_algorithm, n_neighbors=1, n_jobs=-1)
    model_knn.fit(matrix_lowdim)
    if verbose: print("Fitted model")
    
    if verbose:print("Predicting")
     
    points           =   matrix_lowdim[selected_school_units]
    neighbors_data   =   model_knn.kneighbors(   points ,  n_neighbors=min(n_neighbors, matrix_lowdim.shape[0])  ,  return_distance=False  )
    neighbors_lists  =   [sorted(l) for l in neighbors_data.tolist()]
        
    if verbose:print("\nDone predicting")
    
    return neighbors_lists
```

```python

```

```python

```

```python
# %prun -l8 fc =  get_frac_correct( choice(range(0, num_school_units), size=200,  replace=False), df= dataset,  n_neighbors=10,  nn_method="sklearn_rproj",  nn_args={'eps':0.125, 'metric':'euclidean',  'nn_algorithm':'brute'  } )
# print( fc  )
```

```python
%prun -l8 fc =  get_frac_correct( range(0, num_school_units), df= dataset,  n_neighbors=10,  nn_method="sklearn_rproj",  nn_args={'eps':0.125, 'metric':'euclidean',  'nn_algorithm':'brute'  } )
print( fc  )
```

```python
"time per point = %.4f" % (  253.182/25000 )
```

```python

```

```python
# data_units = dataset[["school_unit", "inspection_unit"]].drop_duplicates(ignore_index = True).sort_values(by = "school_unit",  ignore_index=True)
```

```python
# metrics_ball = sorted(sklearn.neighbors.VALID_METRICS['ball_tree'])
# metrics_kd = sorted(sklearn.neighbors.VALID_METRICS['kd_tree'])
# sorted(sklearn.neighbors.VALID_METRICS_SPARSE['brute'])
# boolean_metrics = ['jaccard', 'matching', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']
# best_boolean_metrics = ['jaccard', 'russellrao', 'dice', 'kulsinski', 'sokalsneath']

# # see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
```

```python

# np.set_printoptions(threshold=sys.maxsize)
# selected_school_units = [0, 1]

# sparse_matrix = to_matrix(dataset, dense=False) 
# sparse_matrix
# points = np.concatenate(   [sparse_matrix.getrow(i).todense() for i in selected_school_units]   , axis=0) 
# points

# dense_matrix = to_matrix(dataset, dense=True) 
# dense_matrix

# points = dense_matrix[selected_school_units]
# points

```

```python
# def NNEQ8(x, y):
#     NNEQ = (  (x != y)  ).sum().astype("int8")
#     return NNEQ
```

```python
# list( map(  lambda L : ufloat(np.mean( L ) , np.std( L )/len( L )**0.5) , [ [1, 2, 3] , [4, 5, 6] , [7, 8, 9] ] ) )
```

```python

```

```python

```
