# Experiment Runner

```python
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
```

```python
!pip install --upgrade scikit-learn
!pip install --upgrade scipy
!pip install --upgrade pandas
!pip install --upgrade uncertainties
!pip install --upgrade s3fs
!pip install --upgrade joblib
!pip install --upgrade jupytext
!pip install --upgrade more_itertools 

```

```python
from joblib import Parallel, delayed
from joblib import parallel_backend

from itertools import starmap, product
import multiprocessing

import importlib
import numpy as np
import random
from math import log2, cos, erf
from numpy.random import rand

import time

from collections import namedtuple

from experiment_runner import run_experiments
```

-------------------------------------------------------------------------

```python
def hardfunction(d, n):
    temp = 0
    for i in range(n):
        temp = (temp + d**0.57 + log2(abs(d) ) + erf(d) + cos(d)) % 10000
    return temp
```

```python
hard = 1
def plusalgo(dataset, g, h)    :    return sum( [     hardfunction(d, hard) for d in dataset      ]           ) * (1 + h*rand()/(1+g))
def timesalgo(dataset, g  )    :    return 2**(sum( [    hardfunction(d, hard) for d in dataset   ]     ) % 30) * (1 +   rand()/(1+g))
def minusalgo(dataset     )    :    return sum([   hardfunction(d, hard)  for d in  dataset[::2]  ]           ) * (1 +   rand()      )
```

```python
plusalgo.__code__.co_varnames
```

```python
algo_dict = {    'plus':  plusalgo ,
                 'times': timesalgo,
                 'minus': minusalgo     }
```

```python
dataset_dict =  {    'smallnums' :   [5, 2, 4, 3, 1     ]  *300000   , 
                     'mednums'   :   [14, 13, 12, 11, 15]  *300000   , 
                     'bignums'   :   [99, 62, 73, 85, 51]  *300000       }
```

```python
hyperp_dict =  {  'h' : [0.5, 0.1],      'g' : [3,6,9]     }
```

```python
def pmetr(result): return round(abs(result)**0.01   , 5)
def lmetr(result): return round(    log2(abs(log2(abs(log2(abs(result) ) ) ) ) )   , 5)  

metrics_dict = {    'p-met'   :   pmetr    ,
                    'l-met'   :   lmetr         }
```

```python
def experiment_fn(dataset, algorithm, hparams, metrics_dict):
    result = algorithm(dataset=dataset, **hparams)
    return {n: m( result ) for n, m in metrics_dict.items() }
```

##### Run some experiments

```python
%%time
res = run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn, rchoice_tot=8, n_jobs=8  )
```

```python
res
```

```python

```
