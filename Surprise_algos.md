---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python [conda env:python3]
    language: python
    name: conda-env-python3-py
---

# Surprise Algos on the KlasCement Data

```python
!pip install --upgrade scikit-learn
!pip install --upgrade scipy
!pip install --upgrade pandas
!pip install --upgrade seaborn
!pip install --upgrade uncertainties
!pip install --upgrade surprise
!pip install --upgrade s3fs
!pip install --upgrade jupytext
```

```python
import numpy as np
import seaborn as sns
import pandas as pd


import scipy
from scipy import sparse

import sklearn
from sklearn.metrics import ndcg_score

import surprise
from surprise import Dataset, accuracy, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne,  CoClustering,  SVD, NMF, SVDpp
from surprise.model_selection import cross_validate, KFold, RepeatedKFold,  train_test_split
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader

import copy
import time
import sys
from itertools import starmap, product
import multiprocessing

import statistics
import uncertainties
from uncertainties import ufloat 
```

```python
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The scipy version is {}.'.format(scipy.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The uncertainties version is {}.'.format(uncertainties.__version__))
print('The surprise version is {}.'.format(surprise.__version__))
print('The pandas version is {}.'.format(pd.__version__))
```

#### Load data 

```python
bucket='045879944372-sagemaker-ml-dev'
```

```python
df_ratings = pd.read_csv(          filepath_or_buffer = 's3://{}/{}'.format(bucket, 'klascement_ratings_int.csv'), 
                                              dtype  = {
                                                        'leermiddel_id': 'int32', 
                                                        'gebruiker_id': 'int32', 
                                                        'eng_score': 'int8', 
                                                       }
                        )
```

```python
df_ratings
```

```python
df_ratings[['eng_score']].hist()
```

```python
nratings = df_ratings.shape[0]
n_users = np.int64(np.max(df_ratings[['gebruiker_id']].values ) )
n_items = np.int64(np.max(df_ratings[['leermiddel_id']].values ) )

print("n_users = {:,}".format( int( float ('%.3g' % (n_users)   ) ) ) )  
print("n_items = {:,}".format( int( float ('%.3g' % (n_items)   ) ) ) )
print("dense matrix size = {:,}".format( int( float ('%.3g' % (n_users*n_items)   ) ) ) )  
print("sparse matrix size = {:,}".format( int( float ('%.3g' % (nratings)   ) ) ) )  
print("sparsity = {:.2g} = 1 in  {:,} ".format(   nratings/(n_users*n_items)  ,  int( float ('%.3g' % (n_users*n_items/nratings)  ) )     ) )  
```

```python

```

```python
df_ratings_small = df_ratings[df_ratings['leermiddel_id'] <= 1000]
df_ratings_small = df_ratings_small[df_ratings_small['gebruiker_id'] <= 10000]

df_ratings_medium = df_ratings[df_ratings['leermiddel_id'] <= 2500]
df_ratings_medium = df_ratings_medium[df_ratings_medium['gebruiker_id'] <= 15000]

df_ratings_large = df_ratings[df_ratings['leermiddel_id'] <= 10000]
df_ratings_large = df_ratings_large[df_ratings_large['gebruiker_id'] <= 30000]



```

```python

```

```python

```

### Make surprise datasets

```python

```

```python
# A reader is still needed but only the rating_scale param is required.
reader = Reader(rating_scale=(0, 10))

# The columns must correspond to user id, item id and ratings (in that order).
sds_small  = Dataset.load_from_df(df_ratings_small[['gebruiker_id', 'leermiddel_id', 'eng_score']], reader)
sds_medium = Dataset.load_from_df(df_ratings_medium[['gebruiker_id', 'leermiddel_id', 'eng_score']], reader)
sds_large = Dataset.load_from_df(df_ratings_large[['gebruiker_id', 'leermiddel_id', 'eng_score']], reader)
sds_huge   = Dataset.load_from_df(df_ratings[['gebruiker_id', 'leermiddel_id', 'eng_score']], reader)
```

```python

```

```python
def dataset_info(dataset):

    trainset = dataset.build_full_trainset()
    
    print("n_ratings = {:,}".format( int( float ('%.3g' % trainset.n_ratings ) ) )      )
    print("n_users = {:,}".format( int( float ('%.3g' % trainset.n_users ) ) )      )
    print("average number of ratings per user = {:,}".format( int( float ('%.3g' % (trainset.n_ratings / trainset.n_users)  ) ) )      )
    

```

```python
dataset_info(sds_small)
```

```python
dataset_info(sds_medium)
```

```python
dataset_info(sds_large)
```

### Train with Surprise

```python

```

```python
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, sds_small, measures=['RMSE'], cv=2, verbose=False)
```

```python

```

```python
# cross_validate(algo, sds_medium, measures=['RMSE'], cv=2, verbose=False)
```

```python
# cross_validate(algo, sds_large, measures=['RMSE'], cv=2, verbose=False)
```

```python
# cross_validate(algo, sds_huge, measures=['RMSE'], cv=2, verbose=False)
```

<!-- #region -->
cross_validate(algo, sds_huge, measures=['RMSE'], cv=2, verbose=False)
{'test_rmse': array([0.9106886 , 0.91098935]),
 'fit_time': (926.4300336837769, 933.39604139328),
 'test_time': (223.18039846420288, 209.75310754776)}
 
 
Takes 933 seconds for Surprise to fit the full dataset with SVD. Ouch. 
<!-- #endregion -->

```python
cross_validate(NormalPredictor(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python
# cross_validate(NormalPredictor(), sds_huge, measures=['RMSE'], cv=2, verbose=True)
```

```python
cross_validate(BaselineOnly(), sds_huge, measures=['RMSE'], cv=2, verbose=True)
```

```python
cross_validate(BaselineOnly(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python
# memoryerror for huge
cross_validate(KNNBasic(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python
# memoryerror for huge 
cross_validate(SlopeOne(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python
# NMF gives zerodivisionerror
# cross_validate(NMF(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python
cross_validate(KNNWithZScore(), sds_large, measures=['RMSE'], cv=2, verbose=True)
```

```python

```

### Train with Funk

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

```python
# !pip install git+https://github.com/gbolmier/funk-svd
```

```python
df_ratings.columns = ['u_id', 'i_id', 'rating' ]
```

```python
train = df_ratings.sample(frac=0.9, random_state=7)
val = df_ratings.drop(train.index.tolist()).sample(n=300000, random_state=8)
test = df_ratings.drop(train.index.tolist()).drop(val.index.tolist())

```

```python
test
```

```python
svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100, n_factors=15, min_rating=0, max_rating=10)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)

```

```python
# mae = mean_absolute_error(test['rating'], pred)
rmse = mean_squared_error(test['rating'], pred)
rmse
```

```python
svd
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
svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100, n_factors=30, min_rating=0, max_rating=10)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
```

```python
rmse = mean_squared_error(test['rating'], pred)
rmse
```

```python

```

```python
svd = SVD(learning_rate=0.001, regularization=0.001, n_epochs=100, n_factors=30, min_rating=0, max_rating=10)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
```

```python
rmse = mean_squared_error(test['rating'], pred)
rmse
```

```python
1+1
```

```python
svd = SVD(learning_rate=0.001, regularization=0.001, n_epochs=100, n_factors=100, min_rating=0, max_rating=10)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
```

```python
rmse = mean_squared_error(test['rating'], pred)
rmse
```

```python
svd = SVD(learning_rate=0.001, regularization=0.001, n_epochs=100, n_factors=300, min_rating=0, max_rating=10)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)

```

```python
rmse = mean_squared_error(test['rating'], pred)
rmse
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
# from scipy.sparse import csc_matrix
# >>> from scipy.sparse.linalg import svds
# >>> A = csc_matrix([[1, 0, 0], [5, 0, 2], [0, -1, 0], [0, 0, 3]], dtype=float)
# >>> u, s, vt = svds(A, k=2) # k is the number of factors
# >>> s
# array([ 2.75193379,  5.6059665 ])
```

```python

```

```python
# NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne,  CoClustering,  SVD, NMF, SVDpp
```

```python

```

```python

```
